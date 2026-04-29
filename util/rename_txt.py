import os

# 1. 设置你 label 文件所在的文件夹路径
# 如果脚本就在 label 文件夹里，可以用 '.'
folder_path = r'C:\Users\py_01\Desktop\workSpace\PyCharmCode\Yolo\resource\images'


def rename_labels(path):
    # 遍历文件夹下的所有文件
    files = os.listdir(path)
    count = 0

    for filename in files:
        # 只处理 .txt 文件
        if filename.endswith('.jpg') and '-' in filename:
            # 逻辑：按 '-' 分割，取最后一部分
            # 例如 '3becaf80-image_046.txt' 会被分成 ['3becaf80', 'image_046.txt']
            new_name = filename.split('-')[-1]

            old_file = os.path.join(path, filename)
            new_file = os.path.join(path, new_name)

            try:
                os.rename(old_file, new_file)
                print(f"成功: {filename} -> {new_name}")
                count += 1
            except Exception as e:
                print(f"失败: {filename}, 错误原因: {e}")

    print(f"\n处理完成！共重命名了 {count} 个文件。")


if __name__ == "__main__":
    rename_labels(folder_path)