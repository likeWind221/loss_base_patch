import argparse
import os
import time

import numpy as np
import optuna
from PIL import Image

from main import Config, attack_process, conf_thre, lambda_total, loader
from yolov3.detect import detect, load_model


"""Bayesian (Optuna) search for lambda_total weights.

Objective: maximize ASR while penalizing residual confidence and poor mask shape
(Ncc > 3 or holes > 0). Score is minimized: lower is better.
"""


def evaluate(threat_model, opt, lamb_sparse, lamb_agg, lamb_attack, max_imgs=5, target_ncc=2):
    # Update global lambda_total used by attack_process
    lambda_total["sparse"] = lamb_sparse
    lambda_total["agg"] = lamb_agg
    lambda_total["attack"] = lamb_attack

    suc = 0
    cnt = 0
    conf_res_sum = 0.0  # residual confidence after attack
    shape_pen_sum = 0.0
    ncc_sum = 0.0
    holes_sum = 0.0

    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mydata")
    img_names = [n for n in os.listdir(folder_path) if not n.startswith(".")]

    for i, name in enumerate(img_names):
        if i >= max_imgs:
            break
        img_path = os.path.join(folder_path, name)
        img = Image.open(img_path)
        input_ts = loader(img)

        # pre-filter: need exactly one confident detection
        results = detect(threat_model, input_ts)
        if len(results) != 1 or results[0]["confidence"].item() < conf_thre:
            continue

        cnt += 1
        ok, conf_after, shape_ret = attack_process(
            opt.height,
            opt.width,
            img,
            threat_model,
            opt.device,
            opt.emp_iterations,
            opt.max_pertubation_mask,
            opt.content,
            opt.save_folder,
            name,
            opt.grad_avg,
            i,
        )

        if ok:
            suc += 1
            # reward lower residual confidence even if already < threshold
            conf_res_sum += max(conf_after, 0.0)
            if shape_ret:
                ncc, holes = shape_ret["ncc"], shape_ret["holes"]
                # Penalize both excessive connected components and holes
                shape_pen = max(0, ncc - target_ncc)
                ncc_sum += ncc
                holes_sum += holes
            else:
                shape_pen = 1.0  # unknown shape, light penalty
            shape_pen_sum += shape_pen
        else:
            # failed attack: penalize strongly on confidence; no shape info counted
            conf_res_sum += max(conf_after - conf_thre, 0.0)
            shape_pen_sum += 5.0

    if cnt == 0:
        return 0.0, 1.0, 10.0  # no valid samples -> bad score

    asr = suc / cnt
    conf_avg = conf_res_sum / cnt
    shape_pen_avg = shape_pen_sum / cnt
    ncc_avg = ncc_sum / cnt
    holes_avg = holes_sum / cnt
    return asr, conf_avg, shape_pen_avg, ncc_avg, holes_avg


def objective(trial, threat_model, opt, max_imgs, target_ncc, shape_weight, holes_weight):
    # Expanded search space
    lamb_sparse = trial.suggest_float("sparse", 0.2, 1.0)
    lamb_agg = trial.suggest_float("agg", 0.5, 8.0)
    lamb_attack = trial.suggest_float("attack", 20.0, 80.0)

    asr, conf_avg, shape_pen, ncc_avg, holes_avg = evaluate(
        threat_model, opt, lamb_sparse, lamb_agg, lamb_attack, max_imgs=max_imgs, target_ncc=target_ncc
    )

    # Hard gates: no success or ncc too high -> discard
    if asr == 0:
        return 1e4
    if ncc_avg > target_ncc:
        return 1e3 + ncc_avg

    # Score: prioritize ASR (scaled), then residual confidence, then shape penalty
    score = (1 - asr) * 10.0 + conf_avg + shape_weight * shape_pen + holes_weight * holes_avg

    # Attach diagnostics
    trial.set_user_attr("asr", asr)
    trial.set_user_attr("conf", conf_avg)
    trial.set_user_attr("shape_pen", shape_pen)
    trial.set_user_attr("ncc_avg", ncc_avg)
    trial.set_user_attr("holes_avg", holes_avg)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials")
    parser.add_argument("--max-imgs", type=int, default=8, help="images per trial")
    parser.add_argument("--target-ncc", type=int, default=3, help="desired max connected components")
    parser.add_argument("--shape-weight", type=float, default=2, help="penalty weight for shape (ncc)")
    parser.add_argument("--holes-weight", type=float, default=1.5, help="penalty weight for holes count")
    args = parser.parse_args()

    threat_model = load_model()
    threat_model.eval()
    opt = Config()

    study = optuna.create_study(direction="minimize")

    def obj(trial):
        return objective(trial, threat_model, opt, args.max_imgs, args.target_ncc, args.shape_weight, args.holes_weight)

    start = time.time()
    study.optimize(obj, n_trials=args.trials)
    elapsed = time.time() - start

    best = study.best_trial
    print("\nBest params:", best.params)
    print(
        "Best diagnostics -> ASR={:.3f}, conf={:.3f}, shape_pen={:.3f}, ncc_avg={:.3f}, holes_avg={:.3f}".format(
            best.user_attrs.get("asr", 0.0),
            best.user_attrs.get("conf", 0.0),
            best.user_attrs.get("shape_pen", 0.0),
            best.user_attrs.get("ncc_avg", 0.0),
            best.user_attrs.get("holes_avg", 0.0),
        )
    )
    print(f"Elapsed: {elapsed:.1f}s for {len(study.trials)} trials")


if __name__ == "__main__":
    main()
