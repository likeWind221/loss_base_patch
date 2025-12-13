import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yolov3.detect import detect_train
import numpy as np
from PIL import Image
import scipy.stats as st
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyThresholdMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        alpha = 1
        ctx.save_for_backward(input)
        return input.clamp(max=alpha)

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 1
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > alpha] = 0
        return grad_input

class ThredOne(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        alpha = 0.2
        one = torch.ones_like(input)
        input = torch.where(input < alpha, one, input)
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 0.2
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < alpha] = 0
        return grad_input

class                     GradModify(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 0.1
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        padding = nn.ZeroPad2d((1, 1, 1, 1))
        mask_padding = padding(input)
        kernel = torch.ones((3,3))
        kernel = kernel/9
        kernel[0][0] = kernel[0][2] = kernel[2][0] = kernel[2][2] = 1/9
        kernel[0][1] = kernel[1][0] = kernel[1][2] = kernel[2][1] = 1/4
        kernel[1][1] = 1/2
        kernel = torch.FloatTensor(kernel)
        kernel = torch.stack([kernel])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        kernel = kernel.unsqueeze(0).to(device)
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1)
        grad_input = grad_input*(msk/msk.max())
        return grad_input

trans = transforms.Compose([
    transforms.ToTensor()
])

inputsize = [416,416]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_gaussian_kernel():
    def gkern(kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    channels = 3                                      # 3通道
    kernel_size = 3                                  # kernel大小
    kernel = gkern(kernel_size, 1).astype(np.float32)      # 3表述kernel内元素值得上下限
    gaussian_kernel = np.stack([kernel])   # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

def kernel_5x5():
    kernel = torch.ones((5,5))  
    kernel[0][0] = kernel[0][4] = kernel[4][0] = kernel[4][4] = 7
    kernel[0][1] = kernel[0][3] = kernel[1][0] = kernel[1][4] = kernel[3][0] = kernel[3][4] = kernel[4][1] = kernel[4][3] = 10
    kernel[0][2] = kernel[2][0] = kernel[2][4] = kernel[4][2] = 13
    kernel[1][1] = kernel[1][3] = kernel[3][1] = kernel[3][3] = 14
    kernel[1][2] = kernel[2][1] = kernel[2][3] = kernel[3][2] = 18
    kernel[2][2] = 0
    kernel = torch.FloatTensor(kernel)
    kernel = torch.stack([kernel])
    kernel = kernel.unsqueeze(0).to(device)
    return kernel

def shaped_mask_attack(H, W, bbox, model, img, device, emp_iterations, max_pertubation_mask = 100, content = 0, grad_avg=False, lambda_sparse=3, lambda_attack=30, lambda_agg=3, verbose=False):
    # 图片预处理
    X_ori = torch.stack([trans(img)]).to(device)
    X_ori = F.interpolate(X_ori, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小

    # 随机生成mask
    objbox = torch.rand((H, W)).to(device)
    mask = Variable(objbox, requires_grad=True)

    # 应用自定义方法
    thre_m = MyThresholdMethod.apply
    thre_o = ThredOne.apply
    grad_m = GradModify.apply

    # 动量
    grad_momentum = 0
    
    # 用于记录loss变化
    losses = []

    # 迭代
    for itr in range(emp_iterations):
        # 处理mask，并生成对抗样本
        mask_extrude = mask 
        mask_extrude = mask_extrude ** 2 / (mask_extrude ** 2).sum() * max_pertubation_mask # 限制mask的范围   
        mask_extrude = thre_m(mask_extrude) # mask中大于1的值置0
        mask_extrude = torch.stack([mask_extrude]) # 将(120, 120)扩充为(1, 120, 120)
        mask_extrude = torch.stack([mask_extrude]) # 将(1, 120, 120)扩充为(1, 1, 120, 120)
        mask_modify = grad_m(mask_extrude) 
        # bbox order: (up, below, left, right) => (y1, y2, x1, x2)
        mask_resize = nn.functional.interpolate(mask_modify, (bbox[1] - bbox[0], bbox[3] - bbox[2]), mode='bilinear', align_corners=False)
        padding_layer = nn.ZeroPad2d((bbox[2], 416 - bbox[3], bbox[0], 416 - bbox[1]))
        mask_pad = padding_layer(mask_resize)

        X_adv_b = X_ori * (1 - mask_pad) + content * mask_pad # 生成计算损失用的对抗样本

        # 计算攻击损失
        loss_attack = detect_train(model, X_adv_b)

        # 值稀疏正则项 
        m = thre_o(mask_extrude) # mask中小于0.2的值置1
        o = torch.ones_like(m)
        loss_sparse = -F.mse_loss(m, o) * 100 + (mask_extrude[0][0] ** 4).sum() / max_pertubation_mask 

        # 集聚正则项
        padding = nn.ZeroPad2d((2, 2, 2, 2)) # 上下左右均添加2dim
        mask_padding = padding(mask_extrude) # 对mask_extrude进行填充
        kernel = kernel_5x5() 
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1) 
        loss_agg = ((msk)*mask_extrude).sum()

        loss_att = loss_attack * lambda_attack
        loss_agg = loss_agg / lambda_agg
        loss_sparse = loss_sparse * lambda_sparse 
        # 总损失函数
        loss_total =  loss_att + loss_sparse + loss_agg
        
        # 记录loss值
        losses.append({
            'iteration': itr,
            'loss_total': loss_total.item(),
            'loss_attack': loss_attack.item(),
            'loss_sparse': loss_sparse.item(),
            'loss_agg': loss_agg.item()
        })
        
        # 打印loss变化（每10轮或最后一轮）
        if verbose and (itr % 10 == 0 or itr == emp_iterations - 1):
            print(f"Iteration {itr}/{emp_iterations-1}: ")
            print(f"  Total Loss: {loss_total.item():.10f}")
            print(f"  Attack Loss: {loss_attack.item():.10f}")
            print(f"  Sparse Loss: {loss_sparse.item():.10f}")
            print(f"  Aggregation Loss: {loss_agg.item():.10f}")

        # 反向传播
        loss_total.backward()

        # 带动量的SGD优化
        grad_c = mask.grad.clone()
        if grad_avg:
            gaussian_kernel = generate_gaussian_kernel()
            grad_c = grad_c.reshape(1, 1, grad_c.shape[0], grad_c.shape[1])
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(1,1), groups=1)[0][0]
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (0, 1), keepdim=True) + 0.9 * grad_momentum   # 增加动量系数
        grad_momentum = grad_a     
        mask.grad.zero_()
        mask.data = mask.data + 0.15 * torch.sign(grad_momentum)  # 增加学习率，加速攻击收敛
        mask.data = mask.data.clamp(0., 1.)

    ## 利用生成的mask生成攻击后的图片 
    one = torch.ones_like(mask_pad)
    zero = torch.zeros_like(mask_pad)
    mask_extrude = torch.where(mask_pad > 0.1, one, zero)
    X_adv = X_ori * (1 - mask_extrude) + mask_extrude * content
    adv_face_ts = X_adv.cpu().detach()
    adv_final = X_adv[0].cpu().detach().numpy()
    adv_final = (adv_final * 255).astype(np.uint8)
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    adv_img = Image.fromarray(adv_x_255)
    mask = mask_extrude.cpu().detach()[0][0].numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    # 打印loss统计信息
    if verbose:
        min_total_loss = min(loss['loss_total'] for loss in losses)
        final_total_loss = losses[-1]['loss_total'] if losses else 0
        print(f"\nLoss summary:")
        print(f"  Minimum Total Loss: {min_total_loss:.6f}")
        print(f"  Final Total Loss: {final_total_loss:.6f}")
        print(f"  Loss Reduction: {(losses[0]['loss_total'] - final_total_loss) / losses[0]['loss_total'] * 100:.2f}%")
    
    return adv_face_ts, adv_img, mask