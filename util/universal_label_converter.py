import os
import yaml
from pathlib import Path

# ==================== 【用户只需修改这里】 ====================
# 旧类别ID → 新类别名（按你的业务逻辑合并）
MAPPING_CONFIG = {
    0: "energy_arrow",  # 01-grade1
    1: "energy_arrow",  # 02-grade2
    2: "energy_arrow",  # 03-grade3
    3: "energy_arrow",  # 04-grade4
    4: "energy_arrow",  # 05-grade5
    5: "label",  # 06-label
    6: "box",  # 07-box
    7: "stain",  # 08-stain
    8: "fold",  # 09-fold
    9: "brakeage"  # 10-brakeage
}
# ===========================================================

# 使用绝对路径（基于脚本所在位置的上级目录）
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # util 的上级目录即项目根目录
ROOT_DIR = PROJECT_ROOT / "resource"

LABEL_DIRS = [
    ROOT_DIR / "labels" / "train",
    ROOT_DIR / "labels" / "val"
]


def build_class_mapping(config):
    """生成：旧ID → 连续新ID 的映射 + 新类别名列表"""
    # 提取唯一新类别名（按首次出现顺序保持稳定）
    unique_classes = []
    for new_name in config.values():
        if new_name not in unique_classes:
            unique_classes.append(new_name)

    # 新类别名 → 连续ID (0,1,2...)
    name_to_id = {name: idx for idx, name in enumerate(unique_classes)}

    # 旧ID → 连续新ID
    old_to_new_id = {old_id: name_to_id[new_name] for old_id, new_name in config.items()}

    return old_to_new_id, unique_classes


def convert_labels():
    # 检查目录是否存在
    if not ROOT_DIR.exists():
        print(f"❌ 错误: 数据集目录不存在: {ROOT_DIR}")
        return

    old_to_new_id, new_class_names = build_class_mapping(MAPPING_CONFIG)
    print(f"✅ 映射规则: {old_to_new_id}")
    print(f"✅ 新类别 ({len(new_class_names)}类): {new_class_names}\n")

    # 处理所有标注文件
    processed_count = 0
    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            print(f"⚠️  目录不存在，跳过: {label_dir}")
            continue

        txt_files = list(label_dir.glob("*.txt"))
        print(f"📂 处理目录: {label_dir.name} ({len(txt_files)} 个文件)")

        for path in txt_files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls = int(parts[0])
                    if old_cls in old_to_new_id:
                        parts[0] = str(old_to_new_id[old_cls])
                        new_lines.append(" ".join(parts))
                    # else: 跳过无效类别

                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(new_lines))

                processed_count += 1

            except Exception as e:
                print(f"  ❌ 处理失败 {path.name}: {e}")

    print(f"\n✨ 共处理 {processed_count} 个标签文件")

    # 生成新 data.yaml
    new_yaml = {
        "path": str(ROOT_DIR),  # 使用绝对路径
        "train": "images/train",
        "val": "images/val",
        "nc": len(new_class_names),
        "names": new_class_names  # 顺序即ID: 0,1,2...
    }

    yaml_path = ROOT_DIR / "data.yaml"
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_yaml, f, allow_unicode=True, sort_keys=False)

        print(f"✅ 新配置已生成: {yaml_path}")
        print(f"📌 新类别列表:")
        for idx, name in enumerate(new_class_names):
            print(f"   {idx}: {name}")
        print(f"\n💡 提示: 现在可以使用以下命令训练:")
        print(f"   python train.py")
    except Exception as e:
        print(f"❌ 生成 data.yaml 失败: {e}")

    return yaml_path


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO 标签类别转换工具")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据集目录: {ROOT_DIR}")
    print("=" * 60 + "\n")

    convert_labels()
