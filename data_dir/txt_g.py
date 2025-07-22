import os

# clean_dir 为图像所在目录
clean_dir = '../data/Train/Denoise/'

# 输出文件，放在当前 data_dir/ 目录下的 noisy/ 里
output_file = 'noisy/denoise.txt'

# 获取图像列表
img_list = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
img_list.sort()

# 创建输出目录
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 写入 txt 文件
with open(output_file, 'w') as f:
    for img_name in img_list:
        f.write(f"{img_name}\n")

print(f"Success generate {output_file}，total {len(img_list)} imgs")
