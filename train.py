# src/train.py
from ultralytics import YOLO
import os

# 项目根目录（自动获取）
project_root = os.path.dirname(__file__)

# 数据 yaml 路径
data_yaml = os.path.join(project_root, "resource", "data.yaml")

def main():
    # 加载预训练模型
    model = YOLO("yolo26n.pt")

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=50,  # 提高到 200 轮
        patience=0,  # 【关键】设为 0，禁用自动停止，必须跑满 200 轮
        batch=8,  # 4060 显卡可以跑 8 或 16
        imgsz=640,
        device=0,
        workers=0,  # 【关键修改】将 workers 改为 0，解决内存持续增大问题
        cache=False,  # 【建议】暂时关闭 cache，排除缓存干扰
        lr0=0.01,  # 保持默认学习率
        warmup_epochs=5  # 给模型 5 轮“热身”时间
    )

    print("训练完成！最佳模型：", str(results.save_dir) + "/weights/best.pt")

if __name__ == '__main__':
    main()