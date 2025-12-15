main.py 为攻击程序入口文件
new_attack.py 为攻击逻辑实现主体文件，由main.py调用
range_limit.py 用以测试人体姿态检测与实例分割
shape_utils.py 为统计补丁形态指标工具文件
param_optuna.py 为最优权重参数搜索文件


detect.py 为调用yolov3的检测文件，其中有对相关的detect函数修改，涉及attack_loss