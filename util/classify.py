import os
import shutil
import random
from collections import defaultdict

# --- 用户指定路径 ---
images_dir = '../../resource/images'
labels_dir = '../../resource/labels'

# 类别定义
classes = ['box', 'breakage', 'deviation', 'fold', 'normal', 'stain']
num_val_per_class = 5


def run_split():
    # 1. 检查原始路径是否存在
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"❌ 错误：找不到路径 {images_dir} 或 {labels_dir}")
        return

    # 2. 在原有目录下创建子文件夹
    splits = ['train', 'val']
    for s in splits:
        os.makedirs(os.path.join(images_dir, s), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, s), exist_ok=True)

    # 3. 扫描图片并按前缀分组
    # 注意：只扫描根目录下的文件，不扫描已经移入 train/val 的文件
    all_files = [f for f in os.listdir(images_dir)
                 if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    class_groups = defaultdict(list)
    for f in all_files:
        for cls in classes:
            if f.startswith(cls):
                class_groups[cls].append(f)
                break

    # 4. 执行移动操作
    print("开始整理文件...")
    for cls, files in class_groups.items():
        random.shuffle(files)  # 随机打乱以保证验证集的随机性

        for i, img_name in enumerate(files):
            target_split = 'val' if i < num_val_per_class else 'train'

            # 移动图片
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(images_dir, target_split, img_name)
            shutil.move(src_img, dst_img)

            # 移动对应的标签 (.txt)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(labels_dir, target_split, label_name)

            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)
            else:
                print(f"⚠️ 未找到对应的标签文件: {label_name}")

    print(f"✅ 处理完成！文件已存入 {images_dir}/[train,val] 和 {labels_dir}/[train,val]")


if __name__ == "__main__":
    run_split()