from ultralytics import YOLO
from PIL import Image
import os

# 加载模型
pose_model = YOLO("yolo11n-pose.pt")  
seg_model = YOLO("yolo11n-seg.pt")  

# 获取图片
cur_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(cur_dir, "mydata","random_frame.jpg")

# 获取姿态点估计结果
pose_results = pose_model(img_path)
# 获取分割结果
seg_results = seg_model(img_path)


print("Pose Estimation Results:", pose_results)
print("Segmentation Results:", seg_results)


