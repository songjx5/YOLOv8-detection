import os


# 1. 设置你 label 文件（或图片文件）所在的文件夹路径
# 如果脚本就在文件夹里，可以用 '.'
folder_path = r'C:\Users\py_01\Desktop\workSpace\PyCharmCode\Yolo\resource\images'


def remove_prefix(path):
    # 遍历文件夹下的所有文件
    files = os.listdir(path)
    count = 0

    for filename in files:
        # 只处理 .txt 文件（如果要处理图片，改成 .jpg / .png 等或去掉这行判断）
        if not filename.endswith('.jpg'):
            continue

        # 如果文件名中没有下划线 _ ，跳过
        if '_' not in filename:
            print(f"跳过（无下划线）: {filename}")
            continue

        # 按最后一个下划线分割，取最后一部分作为新名字
        # 例如： defect_batch3_img_0729.txt → img_0729.txt
        new_name = filename.split('_')[-1]

        # 如果分割后新名字和原来一样，说明不需要改，也跳过
        if new_name == filename:
            continue

        old_file = os.path.join(path, filename)
        new_file = os.path.join(path, new_name)

        # 检查目标文件名是否已存在（防止覆盖）
        if os.path.exists(new_file):
            print(f"警告：目标文件已存在，跳过 → {new_name} （原文件：{filename}）")
            continue

        try:
            os.rename(old_file, new_file)
            print(f"成功: {filename} → {new_name}")
            count += 1
        except Exception as e:
            print(f"失败: {filename}, 错误原因: {e}")

    print(f"\n处理完成！共重命名了 {count} 个文件。")


if __name__ == "__main__":
    # 安全起见，先打印文件夹路径确认
    print(f"即将处理文件夹：{folder_path}")
    print("按 Enter 继续，或 Ctrl+C 取消...")
    input()  # 按回车才真正开始执行

    remove_prefix(folder_path)