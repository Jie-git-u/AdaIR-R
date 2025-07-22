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
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=opt.epochs)
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
        drop_last=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        every_n_epochs=1,
        save_top_k=-1
    )

    model = AdaIRModel()

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
