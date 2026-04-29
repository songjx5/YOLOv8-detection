from ultralytics import YOLO
import cv2


model_path = r'C:\Users\py_01\Desktop\workSpace\PyCharmCode\Yolo\src\runs\detect\train3\weights\best.pt'
model = YOLO(model_path)

# 2. 打开摄像头 (0 通常是内置摄像头，如果有多个摄像头可以试 1, 2)
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已启动，按 'q' 键退出预览...")

while True:
    # 读取摄像头的一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法接收帧，正在退出...")
        break

    # 3. 执行推理
    # stream=True 模式在处理长视频流时更节省内存
    results = model.predict(source=frame, conf=0.25, show=False, verbose=False)

    # 4. 绘制结果并显示窗口
    # plot() 会返回一个带有检测框的 numpy 数组（BGR 格式）
    annotated_frame = results[0].plot()

    # 显示实时画面
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # 5. 退出机制：按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()