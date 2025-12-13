import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from yolov3.detect import detect_train1

class MyThresholdMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        alpha = 1
        ctx.save_for_backward(input)
        return input.clamp(max=alpha)

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 1
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input > alpha] = 0
        return grad_input

class ThredOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        alpha = 0.2
        one = torch.ones_like(input)
        output = torch.where(input < alpha, one, input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        alpha = 0.2
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < alpha] = 0
        return grad_input

class GradModify(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        padding = nn.ZeroPad2d((1, 1, 1, 1))
        mask_padding = padding(input)
        kernel = torch.tensor(
            [[1 / 9, 1 / 4, 1 / 9], [1 / 4, 1 / 2, 1 / 4], [1 / 9, 1 / 4, 1 / 9]],
            dtype=torch.float32,
            device=input.device,
        )
        kernel = kernel.expand(1, 1, -1, -1)
        msk = F.conv2d(mask_padding, kernel, bias=None, stride=1)
        grad_input = grad_input * (msk / msk.max())
        return grad_input

trans = transforms.Compose([transforms.ToTensor()])

inputsize = [416, 416]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_gaussian_kernel():
    def gkern(kernlen=21, nsig=5):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        return kernel_raw / kernel_raw.sum()

    kernel_size = 3
    kernel = gkern(kernel_size, 1).astype(np.float32)
    gaussian_kernel = np.expand_dims(np.stack([kernel]), 1)
    return torch.from_numpy(gaussian_kernel).to(device)

def kernel_5x5():
    kernel = torch.ones((5, 5))
    kernel[0][0] = kernel[0][4] = kernel[4][0] = kernel[4][4] = 7
    kernel[0][1] = kernel[0][3] = kernel[1][0] = kernel[1][4] = kernel[3][0] = kernel[3][4] = kernel[4][1] = kernel[4][3] = 10
    kernel[0][2] = kernel[2][0] = kernel[2][4] = kernel[4][2] = 13
    kernel[1][1] = kernel[1][3] = kernel[3][1] = kernel[3][3] = 14
    kernel[1][2] = kernel[2][1] = kernel[2][3] = kernel[3][2] = 18
    kernel[2][2] = 0
    return kernel.float().unsqueeze(0).unsqueeze(0).to(device)

def shaped_mask_attack(
    H,
    W,
    bbox,
    model,
    img,
    device,
    emp_iterations,
    max_pertubation_mask,
    content,
    grad_avg,
    lambda_sparse,
    lambda_attack,
    lambda_agg,
    verbose=False,
):
    X_ori = torch.stack([trans(img)]).to(device)
    X_ori = F.interpolate(X_ori, (inputsize[0], inputsize[1]), mode="bilinear", align_corners=False)

    # mask = Variable(torch.rand((H, W), device=device), requires_grad=True)
    
    # --- 修改后: 初始化为一个中心实心块 ---
    init_mask = torch.zeros((H, W), device=device)
    # 在 BBox 中心生成一个小的初始方块 (比如 1/4 大小)
    c_x = (bbox[2] + bbox[3]) // 2
    c_y = (bbox[0] + bbox[1]) // 2
    h_r = (bbox[1] - bbox[0]) // 4
    w_r = (bbox[3] - bbox[2]) // 4
    
    # 稍微加一点点随机扰动，防止梯度死锁，但主体要是1.0
    init_mask[c_y-h_r:c_y+h_r, c_x-w_r:c_x+w_r] = 1.0 
    init_mask = init_mask + torch.rand_like(init_mask) * 0.05 
    
    mask = Variable(init_mask, requires_grad=True)


    thre_m, thre_o, grad_m = MyThresholdMethod.apply, ThredOne.apply, GradModify.apply
    grad_momentum = 0
    losses = []
    best_total = float("inf")
    best_mask = None
    best_iter = -1

    for itr in range(emp_iterations):
        mask_extrude = mask ** 2 / (mask ** 2).sum() * max_pertubation_mask
        mask_extrude = thre_m(mask_extrude)
        mask_extrude = mask_extrude.unsqueeze(0).unsqueeze(0)
        mask_modify = grad_m(mask_extrude)

        mask_resize = F.interpolate(
            mask_modify,
            (bbox[1] - bbox[0], bbox[3] - bbox[2]),
            mode="bilinear",
            align_corners=False,
        )
        padding_layer = nn.ZeroPad2d((bbox[2], 416 - bbox[3], bbox[0], 416 - bbox[1]))
        mask_pad = padding_layer(mask_resize)

        # STE 二值化
        mask_pad_binary = (mask_pad > 0.5).float()
        mask_pad_for_attack = (mask_pad_binary - mask_pad.detach()) + mask_pad

        X_adv_b = X_ori * (1 - mask_pad_for_attack) + content * mask_pad_for_attack

        # 攻击Loss
        loss_attack = detect_train1(model, X_adv_b)

        # 稀疏Loss
        # m = thre_o(mask_extrude)
        # loss_sparse = -F.mse_loss(m, torch.ones_like(m)) * 100 + (mask_extrude[0][0] ** 4).sum() / max_pertubation_mask
        loss_sparse = torch.mean(torch.abs(mask_extrude)) + torch.mean(mask_extrude * (1 - mask_extrude)) * 4 

        # 聚集Loss
        mask_padding = nn.ZeroPad2d((2, 2, 2, 2))(mask_extrude)
        loss_agg = - (F.conv2d(mask_padding, kernel_5x5(), bias=None, stride=1) * mask_extrude).mean()

        # 总Loss
        loss_attack_final = loss_attack * lambda_attack
        loss_sparse_final = loss_sparse * lambda_sparse
        loss_agg_final = loss_agg * lambda_agg
        total_loss = loss_attack_final + loss_sparse_final + loss_agg_final

        # 追踪最佳loss结果
        if total_loss.item() < best_total and loss_attack.item() < 0.5:
            best_total = total_loss.item()
            best_mask = mask.detach().clone()
            best_iter = itr

        losses.append(
            {
                "iteration": itr,
                "loss_total": total_loss.item(),
                "loss_attack": loss_attack_final.item(),
                "loss_sparse": loss_sparse_final.item(),
                "loss_agg": loss_agg_final.item(),
            }
        )

        if verbose and (itr % 10 == 0 or itr == emp_iterations - 1):
            print(
                f"Iteration {itr}/{emp_iterations - 1}: Total {total_loss.item():.6f}, "
                f"Attack {loss_attack_final.item():.6f}, Sparse {loss_sparse_final.item():.6f}, Agg {loss_agg_final.item():.6f}"
            )

        total_loss.backward()

        grad_c = mask.grad.clone()
        if grad_avg:
            gaussian_kernel = generate_gaussian_kernel()
            grad_c = grad_c.view(1, 1, grad_c.shape[0], grad_c.shape[1])
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(1, 1), groups=1)[0][0]

        grad_a = grad_c / torch.mean(torch.abs(grad_c), (0, 1), keepdim=True) + 0.9 * grad_momentum
        grad_momentum = grad_a
        mask.grad.zero_()
        mask.data = mask.data - 0.15 * torch.sign(grad_momentum)
        mask.data = mask.data.clamp(0.0, 1.0)

    # 迭代结束后用 total_loss 最优的 mask
    if best_mask is not None:
        mask = best_mask

    mask_extrude = mask ** 2 / (mask ** 2).sum() * max_pertubation_mask
    mask_extrude = MyThresholdMethod.apply(mask_extrude)
    mask_extrude = mask_extrude.unsqueeze(0).unsqueeze(0)
    mask_modify = GradModify.apply(mask_extrude)
    mask_resize = F.interpolate(
        mask_modify, (bbox[1] - bbox[0], bbox[3] - bbox[2]), mode="bilinear", align_corners=False
    )
    mask_pad = nn.ZeroPad2d((bbox[2], 416 - bbox[3], bbox[0], 416 - bbox[1]))(mask_resize)

    final_threshold = 0.5
    mask_pad_binary = (mask_pad > final_threshold).float()
    X_adv = X_ori * (1 - mask_pad_binary) + mask_pad_binary * content

    adv_final = (X_adv[0].detach().cpu().numpy() * 255).astype(np.uint8)
    adv_img = Image.fromarray(np.transpose(adv_final, (1, 2, 0)))
    # 生成mask
    mask_extrude_binary = (mask_extrude > final_threshold).float()

    mask_img_numpy = (mask_extrude_binary[0][0].detach().cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_img_numpy)

    if verbose and losses:

        final_total_loss = losses[best_iter]["loss_total"]
        start_total_loss = losses[0]["loss_total"]
        print(
            f"\nLoss summary: best {best_total:.6f} @ iter {best_iter}, "
            f"final {final_total_loss:.6f}, start {start_total_loss:.6f}"
        )

    return X_adv.detach().cpu(), adv_img, mask_img