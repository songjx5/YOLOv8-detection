from pathlib import Path

from ultralytics import YOLO


project_root = Path(__file__).resolve().parent
model_path = project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
test_image_path = project_root / "resource" / "images" / "val" / "dd396540-A02_L5_T03_STA_004.jpg"
predict_name = "predict_image"


def main() -> None:
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(test_image_path),
        conf=0.25,
        save=True,
        project=str(project_root / "runs"),
        name=predict_name,
        exist_ok=True,
        verbose=False,
    )

    saved_image_path = project_root / "runs" / predict_name / test_image_path.name
    print(f"测试图片: {test_image_path}")
    print(f"结果图片: {saved_image_path}")

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("检测结果: 未检测到目标")
        return

    names = results[0].names
    class_count: dict[str, int] = {}
    for cls_id in boxes.cls.tolist():
        class_name = names[int(cls_id)]
        class_count[class_name] = class_count.get(class_name, 0) + 1

    summary = ", ".join(f"{k}: {v}" for k, v in class_count.items())
    print(f"检测结果: {summary}")


if __name__ == "__main__":
    main()
