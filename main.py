import os 
import time
import argparse
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from new_attack import shaped_mask_attack
from yolov3.detect import load_model, detect, detect_single_target
from shape_utils import shape_stats
import torch
from ultralytics import YOLO
import numpy as np

class Config(object):
    device = 0
    display = False
    iterations = 10  # 增加总迭代轮数
    width, height = 40, 80  # 增大补丁大小以覆盖更多关键区域
    # width, height = 50, 100 
    emp_iterations = 100  # 增加每次攻击的内部迭代次数
    # max_pertubation_mask = 50
    cover_rate = 0.1
    max_pertubation_mask = int(width*height*cover_rate*3/4)  # 增加允许的最大扰动范围
    content = 0 # range(0, 1, 0.1)
    grad_avg = True # DI
    head_exclude_ratio = 0.16  # 头部排除比例，0.2表示排除顶部20%的区域
    arm_exclude_ratio = 0.1  # 左右边界排除比例，排除胳膊区域，0.25表示左右各排除25%宽度
    knee_exclude_ratio = 0.25  # 下边界排除比例，将下边界上移至膝盖处，0.3表示从底部向上排除30%的高度
    pose_weights = "yolo11n-pose.pt"
    seg_weights = "yolo11n-seg.pt"
    pose_kpt_conf = 0.2
    res_folder = "res"
    attack_dataset_folder = "datasets/pedestrian"
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    save_folder = os.path.join(res_folder, time.asctime( time.localtime(time.time()) ))
    save_folder = save_folder.replace(":","")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    imgs_dir = os.path.join(save_folder, "adv_imgs")
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)
    msks_dir = os.path.join(save_folder, "infrared_masks")
    if not os.path.exists(msks_dir):
        os.mkdir(msks_dir)

loader = transforms.Compose([
    transforms.ToTensor()
])
conf_thre = 0.5

# 独立控制上下左右缩减比例（相对原框高/宽）
shrink_ratio = {
    "top": 0.17,    
    "bottom": 0.2,  
    "left": 0.10,    
    "right": 0.08    
}

# 权重数值
lambda_total = {
    "sparse":  0.8469770773830809,
    "attack": 23.67611660770411,
    "agg": 6.893487201420355
}

POSE_TARGET_SIZE = 416
POSE_IDXS = {
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_knee": 13,
    "r_knee": 14,
}

def pose_bbox_from_keypoints(img, pose_model, min_conf=0.2, target_size=POSE_TARGET_SIZE, return_kpts=False):
    if pose_model is None:
        return None if not return_kpts else (None, None)
    try:
        results = pose_model(img, verbose=False)
    except Exception as exc:
        print(f"pose inference failed: {exc}")
        return None if not return_kpts else (None, None)

    if not results or not hasattr(results[0], "keypoints"):
        return None if not return_kpts else (None, None)

    result = results[0]
    kpts = result.keypoints
    if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
        return None if not return_kpts else (None, None)

    kp_xy = kpts.xy
    kp_conf = kpts.conf

    if not torch.is_tensor(kp_xy):
        kp_xy = torch.as_tensor(kp_xy)
    if kp_conf is not None and not torch.is_tensor(kp_conf):
        kp_conf = torch.as_tensor(kp_conf)

    person_idx = 0
    if hasattr(result, "boxes") and hasattr(result.boxes, "conf") and result.boxes.conf is not None:
        person_idx = int(torch.argmax(result.boxes.conf).item())

    if person_idx >= kp_xy.shape[0]:
        return None if not return_kpts else (None, None)

    pts_xy = kp_xy[person_idx].cpu()
    pts_conf = kp_conf[person_idx].cpu() if kp_conf is not None else None

    need = [POSE_IDXS["l_shoulder"], POSE_IDXS["r_shoulder"], POSE_IDXS["l_elbow"], POSE_IDXS["r_elbow"], POSE_IDXS["l_knee"], POSE_IDXS["r_knee"]]
    for idx in need:
        if pts_xy.shape[0] <= idx:
            return None if not return_kpts else (None, None)
        if pts_conf is not None and pts_conf[idx].item() < min_conf:
            return None if not return_kpts else (None, None)

    w, h = img.size
    if w == 0 or h == 0:
        return None if not return_kpts else (None, None)

    x_scale = float(target_size) / float(w)
    y_scale = float(target_size) / float(h)

    top = int(min(pts_xy[POSE_IDXS["l_shoulder"]][1], pts_xy[POSE_IDXS["r_shoulder"]][1]) * y_scale)
    bottom = int(max(pts_xy[POSE_IDXS["l_knee"]][1], pts_xy[POSE_IDXS["r_knee"]][1]) * y_scale)
    left = int(min(pts_xy[POSE_IDXS["l_elbow"]][0], pts_xy[POSE_IDXS["r_elbow"]][0]) * x_scale)
    right = int(max(pts_xy[POSE_IDXS["l_elbow"]][0], pts_xy[POSE_IDXS["r_elbow"]][0]) * x_scale)

    pts_scaled = []
    for pt in pts_xy:
        pts_scaled.append((float(pt[0]) * x_scale, float(pt[1]) * y_scale))

    top = max(0, min(target_size - 1, top))
    bottom = max(0, min(target_size - 1, bottom))
    left = max(0, min(target_size - 1, left))
    right = max(0, min(target_size - 1, right))

    if bottom <= top or right <= left:
        return None if not return_kpts else (None, None)

    if return_kpts:
        return (top, bottom, left, right), pts_scaled
    return (top, bottom, left, right)


def get_seg_mask(img, seg_model, target_size=POSE_TARGET_SIZE, device="cuda"):
    if seg_model is None:
        return None
    try:
        results = seg_model(img, verbose=False)
    except Exception as exc:
        print(f"seg inference failed: {exc}")
        return None

    if not results:
        return None
    result = results[0]
    if not hasattr(result, "masks") or result.masks is None:
        return None
    if not hasattr(result, "boxes") or result.boxes is None or result.boxes.conf is None:
        return None

    confs = result.boxes.conf
    idx = int(torch.argmax(confs).item())
    if idx >= result.masks.data.shape[0]:
        return None

    mask = result.masks.data[idx].float()  # (h, w)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 1x1xhxw
    mask = torch.nn.functional.interpolate(mask, (target_size, target_size), mode="bilinear", align_corners=False)
    mask = (mask > 0.5).float().to(device)
    return mask

def shrink_bbox(bbox, ratios):
    """按比例收缩 bbox，bbox: (y1, y2, x1, x2)；若收缩后无效则退回原框"""
    y1, y2, x1, x2 = bbox
    h, w = y2 - y1, x2 - x1
    y1n = y1 + int(h * ratios.get("top", 0))
    y2n = y2 - int(h * ratios.get("bottom", 0))
    x1n = x1 + int(w * ratios.get("left", 0))
    x2n = x2 - int(w * ratios.get("right", 0))
    if y2n <= y1n or x2n <= x1n:
        print("收缩越界，使用原框")
        return bbox
    return (y1n, y2n, x1n, x2n)

def attack_process(H, W, img, threat_model, device, emp_iterations, max_pertubation_mask, content, folder_path, name, grad_avg=True, attack_idx=0, pose_model=None, pose_kpt_conf=0.2, seg_model=None):
    input = loader(img)
    bbox, prob = detect_single_target(threat_model, input) # 在攻击前检测原目标的置信度
    if bbox is None:
        print("未检测到目标，跳过攻击")
        return False, 0.0
    prob_before = prob.item() if hasattr(prob, "item") else float(prob)

    # 限制bbox范围或使用姿态点构建的bbox
    pose_bbox, pose_kpts = pose_bbox_from_keypoints(img, pose_model, min_conf=pose_kpt_conf, return_kpts=True)
    if pose_bbox is not None:
        shrinked_bbox = pose_bbox
        print(f"pose-guided bbox: {shrinked_bbox}")
    else:
        shrinked_bbox = shrink_bbox(bbox, shrink_ratio)
        pose_kpts = None

    seg_mask = get_seg_mask(img, seg_model, target_size=POSE_TARGET_SIZE, device=threat_model.device if hasattr(threat_model, "device") else "cuda")
    # 获取权重参数
    lambda_sparse = lambda_total["sparse"]
    lambda_attack = lambda_total["attack"]
    lambda_agg = lambda_total["agg"]
    print("lambda_sparse: {}, lambda_attack: {}, lambda_agg: {}".format(lambda_sparse, lambda_attack, lambda_agg))
    
    begin = time.time()
    adv_img_ts, adv_img, mask = shaped_mask_attack(
        H, W, shrinked_bbox, threat_model, img, device, emp_iterations,
        max_pertubation_mask, content, grad_avg,
        lambda_sparse=lambda_sparse, lambda_attack=lambda_attack, lambda_agg=lambda_agg, verbose=True,
        seg_mask=seg_mask
    ) # 调用攻击函数进行攻击
    bbox_after, prob = detect_single_target(threat_model, adv_img_ts) # 在攻击后检测目标的置信度
    prob_val = prob.item() if hasattr(prob, "item") else float(prob)
    end = time.time()
    print("optimization time: {}".format(end - begin))
    print("obj score after attack: ", prob_val)

    # 可视化bbox与标签
    annotated = adv_img.copy()
    draw = ImageDraw.Draw(annotated)
    # bbox格式: (up, below, left, right) => (y1, y2, x1, x2)
    draw.rectangle([(bbox[2], bbox[0]), (bbox[3], bbox[1])], outline="green", width=3)
    draw.rectangle([(shrinked_bbox[2], shrinked_bbox[0]), (shrinked_bbox[3], shrinked_bbox[1])], outline="red", width=3)
    if pose_kpts:
        r = 3
        use_idxs = [POSE_IDXS["l_shoulder"], POSE_IDXS["r_shoulder"], POSE_IDXS["l_elbow"], POSE_IDXS["r_elbow"], POSE_IDXS["l_knee"], POSE_IDXS["r_knee"]]
        for idx in use_idxs:
            if idx < len(pose_kpts):
                x, y = pose_kpts[idx]
                draw.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="blue")
    seg_mask_vis = None
    if seg_mask is not None:
        try:
            seg_np = (seg_mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
            seg_mask_vis = Image.fromarray(seg_np, mode="L")
        except Exception:
            seg_mask_vis = None
    if seg_mask_vis is not None:
        annotated_rgba = annotated.convert("RGBA")
        overlay = Image.new("RGBA", seg_mask_vis.size, (64, 224, 208, 40))  # 柔和青色，低透明度
        # 将掩码整体降权，避免遮挡补丁/框/关键点
        soft_mask = seg_mask_vis.point(lambda p: int(p * 0.25))
        annotated_rgba.paste(overlay, mask=soft_mask)
        annotated = annotated_rgba.convert("RGB")

    label = f"success:{prob_val < conf_thre} conf:{prob_val:.3f}"
    draw = ImageDraw.Draw(annotated)
    draw.text((bbox[2], max(bbox[0]-10, 0)), label, fill="green")

    imgs_dir = os.path.join(folder_path, "adv_imgs")
    imgs_raw_dir = os.path.join(folder_path, "adv_imgs_raw")
    msks_dir = os.path.join(folder_path, "infrared_masks")  
    if not os.path.exists(imgs_raw_dir):
        os.makedirs(imgs_raw_dir)

    # 仅在攻击成功时保存图像
    success = prob_val < conf_thre or bbox_after is None
    shape_ret = None
    if success and mask is not None:
        mask_np = (np.array(mask) > 127)  # bool
        # 使用更精细的计数：轮廓计数（如有 OpenCV），避免零散小块被合并
        shape_ret = shape_stats(mask_np, connectivity=1, min_area=1, use_cv=True, mode="contour")
        msk_path = os.path.join(msks_dir, f"{attack_idx}_{os.path.splitext(name)[0]}.png")
        mask.save(msk_path, format="PNG")
        raw_path = os.path.join(imgs_raw_dir, f"{attack_idx}_{os.path.splitext(name)[0]}.png")
        img_path = os.path.join(imgs_dir,     f"{attack_idx}_{os.path.splitext(name)[0]}.png")
        adv_img.save(raw_path, format="PNG")
        annotated.save(img_path, format="PNG")
    return success, prob_val, shape_ret

if __name__=="__main__":
    
    ## 加载待攻击模型 ##
    threat_model = load_model()
    threat_model.eval()
    # 加载攻击参数
    opt = Config()

    # 加载姿态模型（仅一次）
    pose_model = None
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pose_weight_path = opt.pose_weights
        if not os.path.isabs(pose_weight_path):
            pose_weight_path = os.path.join(current_dir, pose_weight_path)
        pose_model = YOLO(pose_weight_path)
        pose_device = f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu"
        pose_model.to(pose_device)
    except Exception as exc:
        print(f"failed to load pose model: {exc}")

    seg_model = None
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        seg_weight_path = opt.seg_weights
        if not os.path.isabs(seg_weight_path):
            seg_weight_path = os.path.join(current_dir, seg_weight_path)
        seg_model = YOLO(seg_weight_path)
        seg_device = f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu"
        seg_model.to(seg_device)
    except Exception as exc:
        print(f"failed to load seg model: {exc}")

    # 获取文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, 'mydata')
    # 初始化指标变量
    suc = 0
    sum = 0
    if not os.path.exists(folder_path):
        print("please prepare the dataset correctly")
        assert False
    # 检查数据集目录是否存在图片
    if len(os.listdir(folder_path))==0:
        print("please prepare the dataset correctly")
        assert False
    
    for i,name in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, name)
        img = Image.open(file_path)
        input = loader(img)
        results = detect(threat_model, input) # 在攻击前检测原目标的置信度
        print("{}th image".format(i))
        # 判断是否只有一个检测目标，如果多出来直接跳过
        if len(results) != 1:
            print("detector detect more than one pedestrian in the image")
            continue
        # 获取检测目标的置信度
        prob = results[0]['confidence']
        if prob.item()<0.5: # 本身检测分数较低的跳过
            print("detector cannot detect any pedestrian in the image")
            continue
        else:
            print("obj score before attack: ", prob.item())

        sum += 1

        # 攻击轮数
        for k in range(opt.iterations):
            print("{}th attack".format(k))
            flag, prob_after, shape_ret = attack_process(
                opt.height, opt.width, img, threat_model, opt.device, opt.emp_iterations,
                opt.max_pertubation_mask, opt.content, opt.save_folder, name, opt.grad_avg, k,
                pose_model=pose_model, pose_kpt_conf=opt.pose_kpt_conf, seg_model=seg_model
            )
            if flag:
                if shape_ret:
                    ncc, holes, lcc = shape_ret["ncc"], shape_ret["holes"], shape_ret["lcc_ratio"]
                    print(f"shape: ncc={ncc}, holes={holes}, lcc_ratio={lcc:.3f}")
                suc += 1
                break
    print("ASR: ", float(suc)/sum)