import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--gopro_dir', type=str, default='data/Train/Deblur/',
                    help='where clean images of denoising saves.')
parser.add_argument('--enhance_dir', type=str, default='data/Train/Enhance/',
                    help='where clean images of denoising saves.')
parser.add_argument('--superres_dir', type=str, default='data/Train/Superres/',
                    help='Super-resolution dataset directory with subfolders low/ and gt/') # 2025.7.14 zhj添加下采样 确认参数解析部分是否支持 superres 所需参数
parser.add_argument('--jpeg_dir', type=str, default='data/Train/Jpeg/',
                    help='where JPEG compressed and gt images are stored') # 2025.7.14 zhj添加下采样 确认参数解析部分是否支持 jpeg 所需参数
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--use_offline_dataset', action='store_true', help='Use offline degraded dataset') # 2025.7.16 zhj修改
parser.add_argument('--offline_dir', type=str, default='data/Train/', help='Root directory for HR and LR') # 2025.7.16 zhj修改
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="AdaIR",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="AdaIR",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1, help = "Number of GPUs to use for training")
parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to pre-trained model checkpoint')
parser.add_argument('--freeze_encoder', action='store_true', help='Whether to freeze encoder during transfer')
parser.add_argument('--run_name', type=str, default='default_run', help='Name of this training run (used for logging, saving)')
parser.add_argument('--freeze_ratio', type=float, default=1.0, help='冻结encoder的比例，范围为0.0（不冻结）到1.0（全部冻结）')


options = parser.parse_args()

