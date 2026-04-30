import os
import shutil
import json
import random
from pathlib import Path


def split_and_copy_dataset(source_dir, target_dir, train_ratio=0.9, seed=42):
    """
    将标注后的数据集划分为训练集和验证集，并复制到目标目录

    参数:
        source_dir: 源目录路径（包含 images/, labels/, classes.txt, notes.json）
        target_dir: 目标目录路径（即 ./resource 目录）
        train_ratio: 训练集比例（默认 0.9，即 90% 训练，10% 验证）
        seed: 随机种子（保证可重复性）
    """

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # ==================== 1. 检查源目录结构 ====================
    print("=" * 60)
    print("YOLO 数据集划分工具")
    print("=" * 60)
    print(f"\n源目录: {source_path}")
    print(f"目标目录: {target_path}")

    source_images = source_path / 'images'
    source_labels = source_path / 'labels'
    source_classes = source_path / 'classes.txt'
    source_notes = source_path / 'notes.json'

    if not source_images.exists():
        raise FileNotFoundError(f"❌ 源目录缺少 images/ 文件夹: {source_images}")

    if not source_labels.exists():
        raise FileNotFoundError(f"❌ 源目录缺少 labels/ 文件夹: {source_labels}")

    # ==================== 2. 获取所有图片文件 ====================
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = [
        f for f in source_images.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ]

    if not all_images:
        raise ValueError(f"❌ 在 {source_images} 中未找到任何图片文件")

    print(f"\n📊 数据统计:")
    print(f"  - 总图片数: {len(all_images)}")

    # 检查标签文件匹配情况
    matched_count = 0
    unmatched_images = []
    for img in all_images:
        label_name = img.stem + '.txt'
        label_path = source_labels / label_name
        if label_path.exists():
            matched_count += 1
        else:
            unmatched_images.append(img.name)

    print(f"  - 匹配的标签: {matched_count}")
    if unmatched_images:
        print(f"  - ⚠️  缺少标签的图片: {len(unmatched_images)}")
        for img_name in unmatched_images[:5]:  # 只显示前5个
            print(f"      • {img_name}")
        if len(unmatched_images) > 5:
            print(f"      ... 还有 {len(unmatched_images) - 5} 个")

    # ==================== 3. 划分数据集 ====================
    random.seed(seed)
    shuffled_images = all_images.copy()
    random.shuffle(shuffled_images)

    split_idx = int(len(shuffled_images) * train_ratio)
    train_images = shuffled_images[:split_idx]
    val_images = shuffled_images[split_idx:]

    print(f"\n📈 数据集划分:")
    print(f"  - 训练集: {len(train_images)} 张 ({train_ratio * 100:.0f}%)")
    print(f"  - 验证集: {len(val_images)} 张 ({(1 - train_ratio) * 100:.0f}%)")

    # ==================== 4. 创建目标目录结构 ====================
    target_images_train = target_path / 'images' / 'train'
    target_images_val = target_path / 'images' / 'val'
    target_labels_train = target_path / 'labels' / 'train'
    target_labels_val = target_path / 'labels' / 'val'

    # 清空已存在的目录（避免文件累积）
    if target_path.exists():
        print(f"\n⚠️  目标目录已存在，将清空重新生成...")
        shutil.rmtree(target_path)

    for dir_path in [target_images_train, target_images_val,
                     target_labels_train, target_labels_val]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"✓ 已创建目录结构")

    # ==================== 5. 复制训练集 ====================
    print(f"\n📦 正在复制训练集...")
    train_copied = 0
    for idx, img_path in enumerate(train_images, 1):
        try:
            # 复制图片
            shutil.copy2(img_path, target_images_train / img_path.name)

            # 复制对应的标签文件
            label_name = img_path.stem + '.txt'
            label_path = source_labels / label_name
            if label_path.exists():
                shutil.copy2(label_path, target_labels_train / label_name)
            else:
                print(f"  ⚠️  跳过（无标签）: {img_path.name}")
                continue

            train_copied += 1

            # 显示进度
            if idx % 50 == 0 or idx == len(train_images):
                print(f"  进度: {idx}/{len(train_images)}")

        except Exception as e:
            print(f"  ❌ 复制失败 {img_path.name}: {e}")

    print(f"✓ 训练集复制完成: {train_copied} 对文件")

    # ==================== 6. 复制验证集 ====================
    print(f"\n📦 正在复制验证集...")
    val_copied = 0
    for idx, img_path in enumerate(val_images, 1):
        try:
            # 复制图片
            shutil.copy2(img_path, target_images_val / img_path.name)

            # 复制对应的标签文件
            label_name = img_path.stem + '.txt'
            label_path = source_labels / label_name
            if label_path.exists():
                shutil.copy2(label_path, target_labels_val / label_name)
            else:
                print(f"  ⚠️  跳过（无标签）: {img_path.name}")
                continue

            val_copied += 1

            # 显示进度
            if idx % 20 == 0 or idx == len(val_images):
                print(f"  进度: {idx}/{len(val_images)}")

        except Exception as e:
            print(f"  ❌ 复制失败 {img_path.name}: {e}")

    print(f"✓ 验证集复制完成: {val_copied} 对文件")

    # ==================== 7. 复制配置文件 ====================
    print(f"\n📄 复制配置文件...")

    if source_classes.exists():
        shutil.copy2(source_classes, target_path / 'classes.txt')
        print(f"  ✓ classes.txt")
    else:
        print(f"  ⚠️  未找到 classes.txt")

    if source_notes.exists():
        shutil.copy2(source_notes, target_path / 'notes.json')
        print(f"  ✓ notes.json")
    else:
        print(f"  ⚠️  未找到 notes.json")

    # ==================== 8. 生成 data.yaml ====================
    print(f"\n⚙️  生成配置文件...")
    create_data_yaml(target_path)

    # ==================== 9. 输出总结 ====================
    print("\n" + "=" * 60)
    print("✅ 数据集划分完成！")
    print("=" * 60)
    print(f"📁 目标目录: {target_path.absolute()}")
    print(f"\n📊 最终统计:")
    print(f"  • 训练集: {train_copied} 张图片 + {train_copied} 个标签")
    print(f"  • 验证集: {val_copied} 张图片 + {val_copied} 个标签")
    print(f"  • 总计: {train_copied + val_copied} 张图片")
    print(f"\n📝 生成的文件:")
    print(f"  • data.yaml - YOLO 训练配置文件")
    if (target_path / 'classes.txt').exists():
        print(f"  • classes.txt - 类别列表")
    if (target_path / 'notes.json').exists():
        print(f"  • notes.json - 标注说明")
    print("\n💡 提示: 现在可以使用以下命令开始训练:")
    print(f"   python train.py")
    print("=" * 60)


def create_data_yaml(target_path):
    """
    创建或更新 data.yaml 配置文件
    """
    # 尝试从 classes.txt 读取类别信息
    classes_file = target_path / 'classes.txt'
    class_names = {}

    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name and not class_name.startswith('#'):
                    class_names[idx] = class_name
        print(f"  ✓ 从 classes.txt 读取到 {len(class_names)} 个类别")

    # 如果没有 classes.txt，尝试从 notes.json 读取
    elif (target_path / 'notes.json').exists():
        try:
            with open(target_path / 'notes.json', 'r', encoding='utf-8') as f:
                notes = json.load(f)
                # 尝试多种可能的字段名
                if 'names' in notes:
                    if isinstance(notes['names'], dict):
                        class_names = notes['names']
                    elif isinstance(notes['names'], list):
                        class_names = {i: name for i, name in enumerate(notes['names'])}
                elif 'classes' in notes:
                    if isinstance(notes['classes'], dict):
                        class_names = notes['classes']
                    elif isinstance(notes['classes'], list):
                        class_names = {i: name for i, name in enumerate(notes['classes'])}
                elif 'categories' in notes:
                    if isinstance(notes['categories'], list):
                        class_names = {i: cat.get('name', f'class_{i}')
                                       for i, cat in enumerate(notes['categories'])}
            print(f"  ✓ 从 notes.json 读取到 {len(class_names)} 个类别")
        except Exception as e:
            print(f"  ⚠️  读取 notes.json 失败: {e}")

    # 如果还是没有找到类别信息，给出警告
    if not class_names:
        print(f"  ⚠️  未找到类别信息，请在 data.yaml 中手动配置")
        class_names = {0: "class_0"}

    # 生成 data.yaml
    yaml_content = f"""# data.yaml - YOLO 训练配置文件
# 自动生成于: {Path.cwd()}

path: {target_path.as_posix()}

train: images/train     # 训练集图片目录（相对 path）
val:   images/val       # 验证集图片目录（相对 path）

# 可选 test: images/test

nc: {len(class_names)}                 # 类别数量
names:
"""

    for idx, name in sorted(class_names.items()):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = target_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"  ✓ data.yaml 已生成")
    print(f"    - 类别数量: {len(class_names)}")
    print(f"    - 类别列表: {', '.join(class_names.values())}")


def main():
    """主函数"""
    import sys

    print("\n" + "=" * 60)
    print("YOLO 数据集划分工具 v1.0")
    print("=" * 60 + "\n")

    # 解析命令行参数
    if len(sys.argv) >= 1:
        source_directory = sys.argv[1]
        target_directory = sys.argv[2] if len(sys.argv) > 2 else "./resource"
        ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

        print("使用命令行参数:")
        print(f"  源目录: {source_directory}")
        print(f"  目标目录: {target_directory}")
        print(f"  训练比例: {ratio}")
        print(f"  随机种子: {seed}\n")
    else:
        # 交互式输入
        print("请输入以下信息（或直接回车使用默认值）:\n")

        source_directory = input("1. 源目录路径（包含 images/, labels/ 等）: ").strip()
        if not source_directory:
            # 尝试常见路径
            default_source = "./annotated_data"
            print(f"   使用默认值: {default_source}")
            source_directory = default_source

        target_directory = input("2. 目标目录路径（通常是 ./resource）: ").strip()
        if not target_directory:
            target_directory = "./resource"
            print(f"   使用默认值: {target_directory}")

        ratio_input = input("3. 训练集比例（默认 0.9，范围 0.1-0.99）: ").strip()
        if ratio_input:
            ratio = float(ratio_input)
            if not 0.1 <= ratio <= 0.99:
                print("   ⚠️  比例超出范围，使用默认值 0.9")
                ratio = 0.9
        else:
            ratio = 0.9
            print(f"   使用默认值: {ratio}")

        seed_input = input("4. 随机种子（默认 42，用于复现结果）: ").strip()
        seed = int(seed_input) if seed_input else 42
        if not seed_input:
            print(f"   使用默认值: {seed}")

        print()

    # 验证输入
    if not Path(source_directory).exists():
        print(f"❌ 错误: 源目录不存在: {source_directory}")
        return

    if not 0.1 <= ratio <= 0.99:
        print(f"❌ 错误: 训练比例必须在 0.1 到 0.99 之间")
        return

    # 确认操作
    if not sys.argv:  # 只在交互模式下确认
        print(f"即将执行:")
        print(f"  从 {source_directory} 划分数据")
        print(f"  输出到 {target_directory}")
        print(f"  训练集比例: {ratio * 100:.0f}%")
        confirm = input("\n是否继续？(y/n): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return
        print()

    # 执行划分
    try:
        split_and_copy_dataset(
            source_dir=source_directory,
            target_dir=target_directory,
            train_ratio=ratio,
            seed=seed
        )
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
    except ValueError as e:
        print(f"\n❌ 数据错误: {e}")
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
