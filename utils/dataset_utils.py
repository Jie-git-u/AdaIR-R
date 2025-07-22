import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation

    
class AdaIRTrainDataset(Dataset):
    def __init__(self, args):
        super(AdaIRTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6, 'superres': 7, 'jpeg': 8} # 2025.7.14 zhj修改，增加下采样superres

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()
        if 'superres' in self.de_type: # 2025.7.14 zhj添加
            self._init_superres_ids()
        if 'jpeg' in self.de_type: # 2025.7.14 zhj添加
            self._init_jpeg_ids()

        random.shuffle(self.de_type)

    def _init_clean_ids(self):
        ref_file = self.args.data_file_dir + "noisy/denoise.txt"
        temp_ids = []
        temp_ids+= [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.args.denoise_dir)
        clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
        temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    # 2025.7.13 zhj修改 支持两种命名
    def _init_deblur_ids(self):
        blur_dir = os.path.join(self.args.gopro_dir, 'blur/')
        sharp_dir = os.path.join(self.args.gopro_dir, 'sharp/')
        blur_list = os.listdir(blur_dir)

        self.deblur_ids = []

        for blur_name in blur_list:
            # 1. 处理像 '0800_gb.png' → '0800.png'
            if '_gb.' in blur_name:
                clean_name = blur_name.replace('_gb.', '.')
            else:
                # 2. 原样保留
                clean_name = blur_name

            # 检查对应 sharp 是否存在
            if os.path.exists(os.path.join(sharp_dir, clean_name)):
                self.deblur_ids.append({
                    "blur_id": blur_name,
                    "clean_id": clean_name,
                    "de_type": 5
                })
            else:
                print(f"[WARN] sharp image not found for: {blur_name} → {clean_name}")

        self.deblur_ids = self.deblur_ids * 5
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)
        print(f'Total Blur Ids: {self.num_deblur}')

    def _init_enhance_ids(self):
        temp_ids = []
        image_list = os.listdir(os.path.join(self.args.enhance_dir, 'low/'))
        temp_ids = image_list
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids * 20
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))
        
    # 2025.7.14 zhj添加初始化方法 _init_superres_ids
    def _init_superres_ids(self):
        low_dir = os.path.join(self.args.superres_dir, 'low/')
        gt_dir = os.path.join(self.args.superres_dir, 'gt/')
        image_list = os.listdir(low_dir)

        self.superres_ids = []

        for low_name in image_list:
            # 例如 000001x2.png → 000001.png
            if 'x2' in low_name:
                base_name = low_name.replace('x2', '')  # 移除 x2 得到 000001.png
                if os.path.exists(os.path.join(gt_dir, base_name)):
                    self.superres_ids.append({
                        "clean_id": low_name,
                        "gt_name": base_name,
                        "de_type": 7
                    })

        self.superres_ids = self.superres_ids * 20
        self.num_superres = len(self.superres_ids)
        print(f"Total Super-Resolution Ids: {self.num_superres}")
        
    # 2025.7.14 zhj添加初始化方法 _init_jpeg_ids
    def _init_jpeg_ids(self):
        low_dir = os.path.join(self.args.jpeg_dir, 'low/')
        gt_dir = os.path.join(self.args.jpeg_dir, 'gt/')
        image_list = os.listdir(low_dir)

        self.jpeg_ids = []
        for low_name in image_list:
            base_name = os.path.splitext(low_name)[0]
            clean_name = base_name.replace('_jpeg', '') + '.png'  # 例如 0001_jpeg → 0001.png
            if os.path.exists(os.path.join(gt_dir, clean_name)):
                self.jpeg_ids.append({
                    "clean_id": low_name,
                    "gt_id": clean_name,
                    "de_type": 8
                })
        self.jpeg_ids = self.jpeg_ids * 20
        self.num_jpeg = len(self.jpeg_ids)
        print(f"Total JPEG Ids: {self.num_jpeg}")

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.args.data_file_dir + "rainy/rainTrain.txt"
        temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name


    def _get_deblur_name(self, deblur_name):
        gt_name = deblur_name.replace("blur", "sharp")
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")
        return gt_name


    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type and hasattr(self, 's15_ids'):
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type and hasattr(self, 's25_ids'):
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type and hasattr(self, 's50_ids'):
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids
        if "superres" in self.de_type:   # 2025.7.14 zhj 添加
            self.sample_ids += self.superres_ids
        if "jpeg" in self.de_type:       # 2025.7.14 zhj 添加
            self.sample_ids += self.jpeg_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split('.')[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)
            elif de_id == 5:
                # Deblur with Gopro set 2025.7.13 zhj修改
                blur_name = sample["blur_id"]
                clean_name = sample["clean_id"]
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'blur/', blur_name)).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.gopro_dir, 'sharp/', clean_name)).convert('RGB')), base=16)
            elif de_id == 6:
                # Enhancement with LOL training set
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'low/', sample["clean_id"])).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.enhance_dir, 'gt/', sample["clean_id"])).convert('RGB')), base=16)
                clean_name = self._get_enhance_name(sample["clean_id"])
            elif de_id == 7:
                # Super-Resolution task 2025.7.14 zhj添加
                low_name = sample["clean_id"]
                gt_name = sample["gt_name"]
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.superres_dir, 'low/', low_name)).convert('RGB')), base=16)
                clean_img  = crop_img(np.array(Image.open(os.path.join(self.args.superres_dir, 'gt/',  gt_name)).convert('RGB')), base=16)
                clean_name = gt_name
            elif de_id == 8:
                # JPEG Compression Restoration
                low_name = sample["clean_id"]
                gt_name = sample["gt_id"]
                degrad_img = crop_img(np.array(Image.open(os.path.join(self.args.jpeg_dir, 'low/', low_name)).convert('RGB')), base=16)
                clean_img = crop_img(np.array(Image.open(os.path.join(self.args.jpeg_dir, 'gt/', gt_name)).convert('RGB')), base=16)
                clean_name = gt_name

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)


        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)


class DenoiseTestDataset(Dataset):
    def __init__(self, args):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15

        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img)
        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1, 'deblur': 2, 'enhance': 3}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma

        self.set_dataset(task)
    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _init_input_ids(self):
        if self.task_idx == 0:
            self.ids = []
            name_list = os.listdir(self.args.derain_path + 'input/')
            self.ids += [self.args.derain_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 1:
            self.ids = []
            name_list = os.listdir(self.args.dehaze_path + 'input/')
            self.ids += [self.args.dehaze_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 2:
            self.ids = []
            name_list = os.listdir(self.args.gopro_path +'input/')
            self.ids += [self.args.gopro_path + 'input/' + id_ for id_ in name_list]
        elif self.task_idx == 3:
            self.ids = []
            name_list = os.listdir(self.args.enhance_path + 'input/')
            self.ids += [self.args.enhance_path + 'input/' + id_ for id_ in name_list]


        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        if self.task_idx == 0:
            gt_name = degraded_name.replace("input", "target")
        elif self.task_idx == 1:
            dir_name = degraded_name.split("input")[0] + 'target/'
            name = degraded_name.split('/')[-1].split('_')[0] + '.png'
            gt_name = dir_name + name
        elif self.task_idx == 2:
            gt_name = degraded_name.replace("input", "target")

        elif self.task_idx == 3:
            gt_name = degraded_name.replace("input", "target")

        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    
class OfflineMixedTrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gt_dir = os.path.join(args.offline_dir, 'HR')
        self.lr_dir = os.path.join(args.offline_dir, 'LR')

        self.degradation_suffixes = {
            'gsn': '_gsn.png',   # 高斯噪声
            'sp': '_sp.png',     # 椒盐噪声
            'gb': '_gb.png',     # 高斯模糊
            'mb': '_mb.png',     # 运动模糊
            'jpeg': '_jpeg.png', # JPEG 压缩
            'ds': '_ds.png',     # 下采样
        }

        # 只保留 args.de_type 中声明的退化类型，确保传入参数是后缀键名，如 ['gsn', 'sp']
        self.use_types = args.de_type
        self.patch_size = args.patch_size

        # 读取所有 HR 图像名
        self.image_names = sorted([
            f for f in os.listdir(self.gt_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.toTensor = ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        base_name = self.image_names[idx]
        name_no_ext = os.path.splitext(base_name)[0]

        clean_path = os.path.join(self.gt_dir, base_name)

        # 随机选择退化任务
        selected_type = random.choice(self.use_types)
        suffix = self.degradation_suffixes[selected_type]
        degraded_name = name_no_ext + suffix
        degraded_path = os.path.join(self.lr_dir, degraded_name)

        # 读取图像（PIL）
        clean_img = Image.open(clean_path).convert('RGB')
        degraded_img = Image.open(degraded_path).convert('RGB')

        # 确保尺寸相同
        assert clean_img.size == degraded_img.size, \
            f"Size mismatch: {clean_path} vs {degraded_path}"

        width, height = clean_img.size
        crop_size = self.patch_size

        # 随机选裁剪位置，保证合法
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size

        # 对 clean 和 degraded 同位置裁剪
        clean_crop = clean_img.crop((left, top, right, bottom))
        degraded_crop = degraded_img.crop((left, top, right, bottom))

        # 转为 numpy
        clean_np = np.array(clean_crop)
        degraded_np = np.array(degraded_crop)

        # 同时随机增强
        clean_np, degraded_np = random_augmentation(clean_np, degraded_np)

        # 转 tensor
        clean_tensor = self.toTensor(clean_np)
        degraded_tensor = self.toTensor(degraded_np)

        return [name_no_ext, selected_type], degraded_tensor, clean_tensor
    
class OfflineMixedTestDataset(Dataset):
    """
    Offline test dataset for AdaIR.
    Assumes HR images in data/Test/HR
    and corresponding LR images in multiple LR_*/ folders like:
        - LR_gsn
        - LR_sp
        - LR_gb
        - LR_mb
        - LR_jpeg
        - LR_ds
    """
    def __init__(self, offline_dir, de_types=None):
        self.root = offline_dir  # e.g., "data/Test"
        self.hr_dir = os.path.join(self.root, "HR")

        # 支持的退化类型
        self.supported_de_types = ['gsn', 'sp', 'gb', 'mb', 'jpeg', 'ds']

        # 用户选择了哪些退化类型
        if de_types is None:
            self.de_types = self.supported_de_types
        else:
            self.de_types = [de for de in de_types if de in self.supported_de_types]
        
        self.toTensor = ToTensor()

        # 全部 HR 图像名称（不含路径）
        self.filenames = sorted([
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.filenames) * len(self.de_types)

    def __getitem__(self, index):
        # index 转换成图像编号和退化类型索引
        img_idx = index // len(self.de_types)
        de_idx = index % len(self.de_types)

        filename = self.filenames[img_idx]
        de_type = self.de_types[de_idx]

        # HR 路径
        hr_path = os.path.join(self.hr_dir, filename)

        # LR 路径
        lr_dir = os.path.join(self.root, f"LR_{de_type}")
        lr_path = os.path.join(lr_dir, filename)

        # 读取图像并转为 numpy 数组
        hr_img = np.array(Image.open(hr_path).convert("RGB"))
        lr_img = np.array(Image.open(lr_path).convert("RGB"))

        # 使用 crop_img 裁剪图像到 16 的倍数大小
        hr_img = crop_img(hr_img, base=16)
        lr_img = crop_img(lr_img, base=16)

        # 转为 tensor
        hr_tensor = self.toTensor(hr_img)
        lr_tensor = self.toTensor(lr_img)

        return {
            'LR': lr_tensor,
            'HR': hr_tensor,
            'filename': filename,
            'de_type': de_type,
        }

