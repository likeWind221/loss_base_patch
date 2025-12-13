import os 
import time
import argparse
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from new_attack import shaped_mask_attack
from yolov3.detect import load_model, detect, detect_single_target

class Config(object):
    device = 0
    display = False
    iterations = 5  # 增加总迭代轮数
    width, height = 40, 80  # 增大补丁大小以覆盖更多关键区域
    # width, height = 50, 100 
    emp_iterations = 200  # 增加每次攻击的内部迭代次数
    # max_pertubation_mask = 50
    cover_rate = 0.1
    max_pertubation_mask = int(width*height*cover_rate*3/4)  # 增加允许的最大扰动范围
    content = 0 # range(0, 1, 0.1)
    grad_avg = True # DI
    head_exclude_ratio = 0.16  # 头部排除比例，0.2表示排除顶部20%的区域
    arm_exclude_ratio = 0.1  # 左右边界排除比例，排除胳膊区域，0.25表示左右各排除25%宽度
    knee_exclude_ratio = 0.25  # 下边界排除比例，将下边界上移至膝盖处，0.3表示从底部向上排除30%的高度
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
    "sparse": 0.5,
    "attack": 50,
    "agg": 2
}

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

def attack_process(H, W, img, threat_model, device, emp_iterations, max_pertubation_mask, content, folder_path, name, grad_avg=True, attack_idx=0):
    input = loader(img)
    bbox, prob = detect_single_target(threat_model, input) # 在攻击前检测原目标的置信度
    if bbox is None:
        print("未检测到目标，跳过攻击")
        return False, 0.0
    prob_before = prob.item() if hasattr(prob, "item") else float(prob)

    # 限制bbox范围
    shrinked_bbox = shrink_bbox(bbox, shrink_ratio)
    # 获取权重参数
    lambda_sparse = lambda_total["sparse"]
    lambda_attack = lambda_total["attack"]
    lambda_agg = lambda_total["agg"]
    print("lambda_sparse: {}, lambda_attack: {}, lambda_agg: {}".format(lambda_sparse, lambda_attack, lambda_agg))
    
    begin = time.time()
    adv_img_ts, adv_img, mask = shaped_mask_attack(
        H, W, shrinked_bbox, threat_model, img, device, emp_iterations,
        max_pertubation_mask, content, grad_avg,
        lambda_sparse=lambda_sparse, lambda_attack=lambda_attack, lambda_agg=lambda_agg, verbose=True
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
    label = f"success:{prob_val < conf_thre} conf:{prob_val:.3f}"
    draw.text((bbox[2], max(bbox[0]-10, 0)), label, fill="green")

    imgs_dir = os.path.join(folder_path, "adv_imgs")
    imgs_raw_dir = os.path.join(folder_path, "adv_imgs_raw")
    msks_dir = os.path.join(folder_path, "infrared_masks")  
    if not os.path.exists(imgs_raw_dir):
        os.makedirs(imgs_raw_dir)

    # 始终保存对抗图（无bbox、有mask）和带可视化版本，文件名前缀标记攻击轮次
    raw_path = os.path.join(imgs_raw_dir, f"{attack_idx}_{os.path.splitext(name)[0]}.png")
    img_path = os.path.join(imgs_dir,     f"{attack_idx}_{os.path.splitext(name)[0]}.png")
    adv_img.save(raw_path, format="PNG")
    annotated.save(img_path, format="PNG")

    success = prob_val < conf_thre or bbox_after is None
    if success and mask is not None:
        msk_path = os.path.join(msks_dir, f"{attack_idx}_{os.path.splitext(name)[0]}.png")
        mask.save(msk_path, format="PNG")
    return success, prob_val

if __name__=="__main__":
    
    ## 加载待攻击模型 ##
    threat_model = load_model()
    threat_model.eval()
    # 加载攻击参数
    opt = Config()
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
            flag, prob_after = attack_process(opt.height, opt.width, img, threat_model, opt.device, opt.emp_iterations, opt.max_pertubation_mask, opt.content, opt.save_folder, name, opt.grad_avg, k)
            if flag:
                suc += 1
                break
    print("ASR: ", float(suc)/sum)