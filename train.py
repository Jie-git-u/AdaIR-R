import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import AdaIRTrainDataset, OfflineMixedTrainDataset
from net.model import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class AdaIRModel(pl.LightningModule):
    def __init__(self, pretrained_ckpt=None, freeze_encoder=False):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        
        # 冻结编码器层（在加载权重前）
        if freeze_encoder:
            print("启用冻结机制，冻结比例 =", opt.freeze_ratio)
            self.freeze_encoder_layers(freeze_ratio=opt.freeze_ratio)
        
        # 加载预训练权重
        if pretrained_ckpt is not None:
            print(f"Loading pretrained model from {pretrained_ckpt} ...")
            ckpt = torch.load(pretrained_ckpt, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = {k.replace("net.", ""): v for k, v in ckpt['state_dict'].items() if "net." in k}
            else:
                state_dict = ckpt
            self.net.load_state_dict(state_dict, strict=False)
            
            
    def freeze_encoder_layers(self, freeze_ratio=1.0):
        print(f"启用冻结机制，冻结比例 = {freeze_ratio}")

        # 所有编码器层名按顺序排列
        encoder_layers = ['encoder_level1', 'encoder_level2', 'encoder_level3']
        num_to_freeze = int(len(encoder_layers) * freeze_ratio)

        # 冻结前 num_to_freeze 层
        for name, module in self.net.named_children():
            if name in encoder_layers[:num_to_freeze]:
                print(f"Freezing {name}")
                for param in module.parameters():
                    param.requires_grad = False
            
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id, task_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # 仅优化需要梯度的参数（冻结部分会被跳过）
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(trainable_params, lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=15,
            max_epochs=opt.epochs
        )
        return [optimizer], [scheduler]
    

def main():
    print("Options")
    print(opt)

    # CUDA 优化设置
    torch.set_float32_matmul_precision('high')
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 选择日志记录器
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="AdaIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # ---------------------------
    # 加载训练数据集（根据是否使用 offline 模式）
    # ---------------------------
    if opt.use_offline_dataset:
        trainset = OfflineMixedTrainDataset(opt)
    else:
        trainset = AdaIRTrainDataset(opt)

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=opt.num_workers,
        drop_last=True,
        prefetch_factor=4
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1
    )

    model = AdaIRModel(pretrained_ckpt=opt.pretrained_ckpt, freeze_encoder=opt.freeze_encoder)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16,
        accumulate_grad_batches=4
    )

    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()
