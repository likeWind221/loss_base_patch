项目简介
-
本目录实现红外补丁攻击（infrared patch attack）的主要代码与调用入口。该实现基于 YOLOv3 检测器，包含两套优化器：`attack.py`（旧版）与 `new_attack.py`（新版，默认由 `main.py` 调用）。

参考论文: Physically Adversarial Infrared Patches with Learnable Shapes and Locations

参考项目: https://github.com/shighghyujie/infrared_patch_attack

模型权重下载地址: https://drive.google.com/file/d/1qvcQGhs20eU6DGgQqtGctVXgfJspKa2V/view?usp=sharing

快速指南
-
- **入口**: 请使用 `main.py` 启动默认攻击流程；使用 `new_attack.py` 可运行替代的优化/损失实现。示例：

```bash
cd "infrared_patch/mycode"
python main.py
```

- **输出**: 运行后结果会保存在 `res/<timestamp>/` 下，包含 `adv_imgs/`（对抗图像）和 `infrared_masks/`（补丁二值图）。


目录与关键文件说明
-
- `main.py`: 攻击程序入口，负责加载模型、遍历数据并调用优化器。
- `new_attack.py`: 推荐的攻击逻辑实现（更简单的约束与更新规则）。
- `attack.py`: 旧版攻击实现（可忽略，供参考）。
- `detect.py`: YOLOv3 检测封装，含若干用于攻击计算的 `detect_train` / `detect_train1` 函数。
- `shape_utils.py`: 补丁形态统计工具。
- `param_optuna.py`: 超参数搜索/优化（基于 Optuna）。
- `range_limit.py`: 与人体姿态检测/分割相关的测试脚本。
- `mydata`: 放置测试图片。
- `yolov3`: yolov3库文件。
- `res`: 存放结果文件。


依赖与环境
-
- 推荐 Python 3.8+。
- PyTorch >= 1.7（建议与 CUDA 匹配的官方版本以使用 GPU 加速）。
- 其他 Python 包请参考仓库中的依赖文件：
- 本目录专用依赖：`infrared_patch/mycode/requirements.txt`

安装步骤（示例）
-
1. 使用 conda 创建并激活环境（推荐）：

```bash
conda create -n irpatch python=3.8 -y
conda activate irpatch
```

2. 安装依赖：

```bash
pip install -r "infrared_patch/mycode/requirements.txt"
```

3. 确保模型权重文件 `best.pt` 可用，并放在项目期望的位置：
- 常见位置：`infrared_patch/mycode/best.pt`。


运行与配置要点
-
- 运行目录建议切换到 `infrared_patch/mycode`，因为脚本使用相对路径加载权重与数据。
- 默认数据目录为 `mydata/`（脚本会遍历该目录中的样本）。
- 配置参数（如补丁尺寸、迭代次数、遮盖率等）由 `config.py`（或相应脚本内的 `Config` 类）管理；运行前可修改以适配场景。

注意事项与常见问题
-
- GPU 与 CUDA: 代码中多数地方直接调用 `.cuda()`，若无 GPU 需手动将代码改为可在 CPU 上运行或在运行时使用 GPU。当前实现默认依赖 CUDA，CPU 运行可能报错。
- 检测门槛: 主流程会跳过多目标或置信度过低的样本（默认置信度阈值 0.5），因此部分输入可能被跳过。
- 补丁与像素值: `content` 参数控制热图强度（值为 0 表示热源较暗/冷，根据热图语义可能需要反向理解）。


输出说明
-
- 成功攻击后，`res/<timestamp>/adv_imgs/` 下会保存对抗图，`res/<timestamp>/infrared_masks/` 下会保存 8-bit 灰度补丁掩码（binary）。



