import os
import argparse
import subprocess
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import OfflineMixedTestDataset  # 你需实现这个，支持传入退化类型列表
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import AdaIR


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
    
    def forward(self, x):
        return self.net(x)


def test_MultiTask(net, dataset, de_types, output_path_base):
    """
    多任务统一测试接口
    de_types: 任务名列表，如 ['gsn', 'gb', 'sp', 'mb', 'jpeg', 'ds']
    """
    # 依赖你的 OfflineMixedTestDataset 返回的 batch 格式：
    # ([img_names, de_types_in_batch], degraded_imgs, clean_imgs)
    device = next(net.parameters()).device

    psnr_meters = {de: AverageMeter() for de in de_types}
    ssim_meters = {de: AverageMeter() for de in de_types}

    testloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    for batch in tqdm(testloader, desc="Testing MultiTask"):
        degraded_imgs = batch['LR']
        clean_imgs = batch['HR']
        names = batch['filename']
        de_task_list = batch['de_type']
        degraded_imgs = degraded_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        with torch.no_grad():
            restored = net(degraded_imgs)

        for i in range(len(names)):
            de_type = de_task_list[i]
            if de_type not in de_types:
                continue
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored[i:i+1], clean_imgs[i:i+1])
            psnr_meters[de_type].update(temp_psnr, N)
            ssim_meters[de_type].update(temp_ssim, N)

            # 保存恢复图像
            out_dir = os.path.join(output_path_base, de_type)
            os.makedirs(out_dir, exist_ok=True)
            # print("before saving:", restored[i].shape)
            save_image_tensor(restored[i], os.path.join(out_dir, f"{names[i]}_{de_type}.png"))

    # 打印各任务指标
    for de_type in de_types:
        print(f"[{de_type}] PSNR: {psnr_meters[de_type].avg:.4f}, SSIM: {ssim_meters[de_type].avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--de_types', nargs='+', default=['gsn', 'gb', 'sp', 'mb', 'jpeg', 'ds'],
                        help='测试任务列表，支持多个，示例：--de_types gsn jpeg sp')
    parser.add_argument('--offline_dir', type=str, required=True, help='离线数据目录，需包含 HR 和 LR 子目录')
    parser.add_argument('--output_path', type=str, default='AdaIR_results/', help='恢复结果保存目录')
    parser.add_argument('--ckpt_name', type=str, required=True, help='模型权重文件名，如 adair.ckpt')
    parser.add_argument('--save_images', action='store_true', help='Whether to save output images')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    testopt = parser.parse_args() 
    ckpt_path = testopt.ckpt_name
    net = AdaIRModel().load_from_checkpoint(ckpt_path, strict=False)
    net = net.to(device)
    net.eval()

    # 加载测试集，传入需要测试的退化任务类型列表
    dataset = OfflineMixedTestDataset(args.offline_dir, de_types=args.de_types)

    # 统一测试
    test_MultiTask(net, dataset, args.de_types, args.output_path)
