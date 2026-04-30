import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

# 导入 OCR 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.ocr_energy_grade import EnergyGradeOCR


class YOLODetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 能效标签智能检测系统 v2.0")
        self.root.geometry("1600x900")

        # 项目路径配置
        self.project_root = Path(__file__).resolve().parent

        # 查找最新的模型
        self.model_path = self._find_latest_model()

        # 类别名称映射（基于最新的 6 类配置）
        self.class_names = {
            0: 'energy_arrow',
            1: 'label',
            2: 'box',
            3: 'stain',
            4: 'fold',
            5: 'brakeage'
        }

        # 显示名称映射（使用纯文字，避免 Emoji 乱码）
        self.display_names = {
            'energy_arrow': {'type': 'grade', 'display': '能效箭头', 'icon': '[箭头]'},
            'label': {'type': 'label', 'display': '能效标签', 'icon': '[标签]'},
            'box': {'type': 'box', 'display': '产品纸盒', 'icon': '[纸盒]'},
            'stain': {'type': 'defect', 'display': '污渍', 'icon': '[污渍]'},
            'fold': {'type': 'defect', 'display': '褶皱', 'icon': '[褶皱]'},
            'brakeage': {'type': 'defect', 'display': '破损', 'icon': '[破损]'}
        }

        # 加载模型和 OCR
        self.model = None
        self.ocr = None
        self.label_font = self._load_chinese_font(18)
        self.current_image = None
        self.current_image_path = None
        self.detection_results = None

        # 初始化界面
        self._init_ui()

        # 加载模型和 OCR
        self._load_model_and_ocr()

    def _find_latest_model(self):
        """查找最新的训练模型"""
        runs_dir = self.project_root / "runs" / "detect"

        if not runs_dir.exists():
            return None

        # 查找所有 train-* 目录
        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]

        if not train_dirs:
            return None

        # 按修改时间排序，返回最新的
        latest = max(train_dirs, key=lambda d: d.stat().st_mtime)
        model_path = latest / "weights" / "best.pt"

        if model_path.exists():
            print(f"找到最新模型: {model_path}")
            return model_path

        return None

    def _init_ui(self):
        """初始化用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：控制面板和结果
        left_panel = tk.Frame(main_frame, width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # 右侧：图像显示
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 左侧面板内容 ===

        # 标题
        title_label = tk.Label(
            left_panel,
            text="能效标签检测系统",
            font=("Microsoft YaHei", 16, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=(0, 15))

        # 控制按钮框架
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        load_btn = tk.Button(
            btn_frame,
            text="选择图片",
            command=self._load_image,
            font=("Microsoft YaHei", 11),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        load_btn.pack(fill=tk.X, pady=2)

        detect_btn = tk.Button(
            btn_frame,
            text="开始检测",
            command=self._detect,
            font=("Microsoft YaHei", 11),
            bg="#27ae60",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        detect_btn.pack(fill=tk.X, pady=2)

        clear_btn = tk.Button(
            btn_frame,
            text="清除结果",
            command=self._clear,
            font=("Microsoft YaHei", 11),
            bg="#95a5a6",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            pady=5,
            cursor="hand2"
        )
        clear_btn.pack(fill=tk.X, pady=2)

        # 状态标签
        self.status_label = tk.Label(
            left_panel,
            text="状态: 正在加载模型...",
            font=("Microsoft YaHei", 9),
            fg="#7f8c8d",
            anchor=tk.W,
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.X, pady=(10, 5))

        # 分隔线
        separator1 = tk.Frame(left_panel, height=2, bg="#ecf0f1")
        separator1.pack(fill=tk.X, pady=10)

        # 检测结果标题
        result_title = tk.Label(
            left_panel,
            text="检测结果",
            font=("Microsoft YaHei", 13, "bold"),
            fg="#34495e",
            anchor=tk.W
        )
        result_title.pack(fill=tk.X, pady=(0, 5))

        # 结果显示文本框（带滚动条）
        result_frame = tk.Frame(left_panel)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(
            result_frame,
            font=("Microsoft YaHei", 10),
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT,
            bg="#fafafa",
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)

        # === 右侧面板内容 ===

        # 图像显示框架（左右并排）
        image_container = tk.Frame(right_panel)
        image_container.pack(fill=tk.BOTH, expand=True)

        # 原图框架
        original_frame = tk.LabelFrame(
            image_container,
            text="原始图片",
            font=("Microsoft YaHei", 11, "bold"),
            fg="#2c3e50"
        )
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_canvas = tk.Canvas(
            original_frame,
            bg="#2c3e50",
            highlightthickness=0
        )
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 检测结果框架
        result_frame_display = tk.LabelFrame(
            image_container,
            text="检测结果",
            font=("Microsoft YaHei", 11, "bold"),
            fg="#2c3e50"
        )
        result_frame_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.result_canvas = tk.Canvas(
            result_frame_display,
            bg="#2c3e50",
            highlightthickness=0
        )
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _load_model_and_ocr(self):
        """加载 YOLO 模型和 OCR 引擎"""
        status_messages = []

        # 加载 YOLO 模型
        try:
            if self.model_path is None or not self.model_path.exists():
                status_messages.append("模型文件不存在")
                self.status_label.config(text="状态: 模型未找到", fg="#e74c3c")
                messagebox.showwarning(
                    "警告",
                    "未找到训练好的模型文件\n\n请先运行训练或检查模型路径"
                )
                return

            self.model = YOLO(str(self.model_path))
            status_messages.append("YOLO 模型加载成功")
            print(f"模型加载成功: {self.model_path.name}")

        except Exception as e:
            status_messages.append(f"模型加载失败: {str(e)}")
            self.status_label.config(text="状态: 模型加载失败", fg="#e74c3c")
            messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")
            return

        # 加载 OCR 引擎
        try:
            print("\n正在加载 OCR 引擎...")
            self.ocr = EnergyGradeOCR(use_gpu=False)
            if self.ocr.available:
                status_messages.append("OCR 引擎加载成功")
                print("OCR 引擎加载成功")
            else:
                status_messages.append("OCR 功能不可用（将跳过等级识别）")
                print("警告: OCR 功能不可用")
        except Exception as e:
            status_messages.append(f"OCR 加载失败: {str(e)}")
            print(f"OCR 加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.ocr = None

        # 更新状态
        self.status_label.config(
            text="状态: 系统就绪",
            fg="#27ae60"
        )

        print("\n".join(status_messages))

    def _load_chinese_font(self, size=18):
        """加载可用于中文绘制的字体"""
        font_candidates = [
            Path("C:\\Windows\\Fonts\\msyh.ttc"),     # Microsoft YaHei
            Path("C:\\Windows\\Fonts\\msyhbd.ttc"),
            Path("C:\\Windows\\Fonts\\simhei.ttf"),  # SimHei
            Path("C:\\Windows\\Fonts\\simsun.ttc"),  # SimSun
            Path("C:\\Windows\\Fonts\\simkai.ttf"),  # KaiTi
        ]

        for font_path in font_candidates:
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), size=size)
                except OSError:
                    continue

        return ImageFont.load_default()

    def _load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title="选择检测图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("所有文件", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            # 使用 OpenCV 读取图片
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                raise ValueError("无法读取图片")

            self.current_image_path = Path(file_path)

            # 显示原图
            self._display_image(self.current_image, self.original_canvas)

            # 清空结果画布
            self.result_canvas.delete("all")
            self.result_canvas.create_text(
                self.result_canvas.winfo_width() // 2,
                self.result_canvas.winfo_height() // 2,
                text="点击「开始检测」进行分析",
                fill="white",
                font=("Microsoft YaHei", 12)
            )

            # 清空结果文本
            self._update_result_text("")

            self.status_label.config(
                text=f"状态: 已加载 {self.current_image_path.name}",
                fg="#27ae60"
            )

        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败:\n{str(e)}")

    def _detect(self):
        """执行检测"""
        if self.model is None:
            messagebox.showwarning("警告", "模型未加载，请稍候再试")
            return

        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择一张图片")
            return

        try:
            self.status_label.config(text="状态: 正在检测...", fg="#f39c12")
            self.root.update()

            # 执行 YOLO 检测
            results = self.model.predict(
                source=str(self.current_image_path),
                conf=0.25,
                save=False,
                verbose=False
            )

            # 处理结果
            self.detection_results = results[0]

            # 绘制检测结果图
            result_image = self._draw_detections(self.current_image.copy(), self.detection_results)
            self._display_image(result_image, self.result_canvas)

            # OCR 识别能效等级
            ocr_grade = self._ocr_recognize_grade(self.detection_results)

            # 分析并显示结果
            analysis = self._analyze_results(self.detection_results, ocr_grade)
            self._update_result_text(analysis)

            self.status_label.config(text="状态: 检测完成", fg="#27ae60")

        except Exception as e:
            self.status_label.config(text="状态: 检测失败", fg="#e74c3c")
            messagebox.showerror("错误", f"检测失败:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _ocr_recognize_grade(self, results):
        """使用 OCR 识别能效等级"""
        if not self.ocr or not self.ocr.available:
            print("OCR 不可用，跳过等级识别")
            return None

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        names = results.names

        # 查找 energy_arrow 与 label 的边界框
        arrow_boxes = []
        label_boxes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            if class_name == 'energy_arrow':
                arrow_boxes.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'conf': float(box.conf[0])
                })
            elif class_name == 'label':
                label_boxes.append({
                    'box': box.xyxy[0].cpu().numpy(),
                    'conf': float(box.conf[0])
                })

        # 优先识别高置信度箭头
        arrow_boxes.sort(key=lambda item: item['conf'], reverse=True)
        label_boxes.sort(key=lambda item: item['conf'], reverse=True)

        if arrow_boxes:
            print(f"找到 {len(arrow_boxes)} 个箭头区域，开始 OCR 识别...")
        else:
            print("未检测到 energy_arrow，尝试使用 label 区域进行 OCR")

        # 1) 先用 arrow 区域（只尝试高置信度前 2 个，控制耗时）
        for idx, item in enumerate(arrow_boxes[:2]):
            arrow_box = item['box']
            print(f"识别第 {idx + 1} 个箭头区域: {arrow_box}")
            for padding in (25, 55):
                grade = self.ocr.recognize_from_box(self.current_image, arrow_box, padding=padding)
                if grade:
                    print(f"  -> 识别成功: {grade} (padding={padding})")
                    return grade
            print("  -> 识别失败")

        # 2) OCR 失败时，优先用几何位置兜底（速度快且稳定）
        geometry_grade = self._infer_grade_from_geometry(arrow_boxes, label_boxes)
        if geometry_grade:
            return geometry_grade

        # 3) 几何兜底仍失败时，用最高置信度 label 大框兜底 OCR
        if label_boxes:
            print(f"箭头识别失败，尝试 {len(label_boxes)} 个 label 区域 OCR...")
        for idx, item in enumerate(label_boxes[:1]):
            label_box = item['box']
            for padding in (25,):
                grade = self.ocr.recognize_from_box(self.current_image, label_box, padding=padding)
                if grade:
                    print(f"  -> label 识别成功: {grade} (idx={idx + 1}, padding={padding})")
                    return grade

        # 4) 全部失败
        return None

    def _infer_grade_from_geometry(self, arrow_boxes, label_boxes):
        """根据箭头在标签中的垂直位置推断能效等级"""
        if not arrow_boxes or not label_boxes:
            return None

        best_arrow = arrow_boxes[0]['box']
        arrow_cx = (best_arrow[0] + best_arrow[2]) / 2.0
        arrow_cy = (best_arrow[1] + best_arrow[3]) / 2.0

        target_label = None
        for item in label_boxes:
            lx1, ly1, lx2, ly2 = item['box']
            if lx1 <= arrow_cx <= lx2 and ly1 <= arrow_cy <= ly2:
                target_label = item['box']
                break
        if target_label is None:
            target_label = label_boxes[0]['box']

        lx1, ly1, lx2, ly2 = target_label
        label_h = max(1.0, ly2 - ly1)
        rel_y = (arrow_cy - ly1) / label_h
        rel_y = max(0.0, min(0.999, rel_y))

        # 标准能效标签一般为从上到下 1~5 级
        grades = ['一级', '二级', '三级', '四级', '五级']
        idx = int(rel_y * 5.0)
        inferred = grades[idx]
        print(f"  -> OCR 未命中，按箭头位置推断: {inferred} (rel_y={rel_y:.3f})")
        return inferred

    def _draw_detections(self, image, results):
        """在图片上绘制检测结果"""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return image

        # 颜色配置（RGB，供 PIL 绘制）
        colors = {
            'grade': (0, 255, 0),      # 绿色 - 能效相关
            'label': (255, 255, 0),    # 黄色 - 标签
            'box': (0, 255, 255),      # 青色 - 纸盒
            'defect': (255, 0, 0),     # 红色 - 缺陷
        }

        names = results.names
        image_h, image_w = image.shape[:2]

        # 使用 PIL 绘制中文，避免 cv2.putText 中文乱码
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        font = self.label_font if self.label_font else ImageFont.load_default()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names[cls_id]

            # 确定颜色
            display_info = self.display_names.get(class_name, {})
            category_type = display_info.get('type', 'defect')
            color = colors.get(category_type, (255, 255, 255))

            # 绘制边界框
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

            # 绘制标签背景与文字
            display_name = display_info.get('display', class_name)
            label = f"{display_name} {conf:.2f}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            box_left = max(0, x1)
            box_right = min(image_w, box_left + text_w + 10)
            box_top = y1 - text_h - 8
            box_bottom = y1

            if box_top < 0:
                box_top = y1
                box_bottom = min(image_h, y1 + text_h + 8)
                text_y = box_top + 3
            else:
                text_y = box_top + 3

            draw.rectangle([(box_left, box_top), (box_right, box_bottom)], fill=color)
            draw.text((box_left + 5, text_y), label, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _analyze_results(self, results, ocr_grade=None):
        """分析检测结果并生成文字报告"""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return "未检测到任何目标\n\n请确保图片中包含能效标签和产品包装"

        names = results.names
        detected_classes = {}

        # 统计检测到的类别
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names[cls_id]

            if class_name not in detected_classes:
                detected_classes[class_name] = []
            detected_classes[class_name].append({
                'confidence': conf,
                'box': box.xyxy[0].cpu().numpy()
            })

        # 生成分析报告
        report_lines = []
        report_lines.append("=" * 45)
        report_lines.append("能效标签检测报告")
        report_lines.append("=" * 45)
        report_lines.append("")

        # 1. 能耗等级（优先使用 OCR 识别结果）
        report_lines.append("【能耗等级】")
        if ocr_grade:
            grade_display = {
                '一级': '一级能效 (最佳)',
                '二级': '二级能效 (较节能)',
                '三级': '三级能效 (中等)',
                '四级': '四级能效 (较低)',
                '五级': '五级能效 (最低)'
            }
            display = grade_display.get(ocr_grade, ocr_grade)
            report_lines.append(f"OCR 识别结果: {display}")
            report_lines.append(f"(通过箭头区域文字识别)")
        elif 'energy_arrow' in detected_classes:
            max_conf = max([d['confidence'] for d in detected_classes['energy_arrow']])
            report_lines.append(f"检测到能效箭头")
            report_lines.append(f"置信度: {max_conf:.2%}")
            if not self.ocr or not self.ocr.available:
                report_lines.append(f"提示: OCR 未启用，无法识别具体等级")
            else:
                report_lines.append(f"提示: OCR 识别失败，请检查箭头区域文字是否清晰")
        else:
            report_lines.append("未检测到能效箭头")

        report_lines.append("")

        # 2. 标签和纸盒
        report_lines.append("【标签与包装】")
        if 'label' in detected_classes:
            max_conf = max([d['confidence'] for d in detected_classes['label']])
            report_lines.append(f"能效标签: 已检测到")
            report_lines.append(f"置信度: {max_conf:.2%}")
        else:
            report_lines.append("能效标签: 未检测到")

        if 'box' in detected_classes:
            max_conf = max([d['confidence'] for d in detected_classes['box']])
            report_lines.append(f"产品纸盒: 已检测到")
            report_lines.append(f"置信度: {max_conf:.2%}")
        else:
            report_lines.append("产品纸盒: 未检测到")

        report_lines.append("")

        # 3. 缺陷检测
        report_lines.append("【质量检测】")
        defects_found = False

        if 'brakeage' in detected_classes:
            defects_found = True
            count = len(detected_classes['brakeage'])
            max_conf = max([d['confidence'] for d in detected_classes['brakeage']])
            report_lines.append(f"破损: 发现 {count} 处")
            report_lines.append(f"最高置信度: {max_conf:.2%}")

        if 'stain' in detected_classes:
            defects_found = True
            count = len(detected_classes['stain'])
            max_conf = max([d['confidence'] for d in detected_classes['stain']])
            report_lines.append(f"污渍: 发现 {count} 处")
            report_lines.append(f"最高置信度: {max_conf:.2%}")

        if 'fold' in detected_classes:
            defects_found = True
            count = len(detected_classes['fold'])
            max_conf = max([d['confidence'] for d in detected_classes['fold']])
            report_lines.append(f"褶皱: 发现 {count} 处")
            report_lines.append(f"最高置信度: {max_conf:.2%}")

        if not defects_found:
            report_lines.append("未发现明显缺陷")

        report_lines.append("")

        # 4. 偏移检测（基于标签和纸盒的相对位置）
        report_lines.append("【位置分析】")
        if 'label' in detected_classes and 'box' in detected_classes:
            label_box = detected_classes['label'][0]['box']
            box_coords = detected_classes['box'][0]['box']

            # 计算中心点
            label_center_x = (label_box[0] + label_box[2]) / 2
            label_center_y = (label_box[1] + label_box[3]) / 2
            box_center_x = (box_coords[0] + box_coords[2]) / 2
            box_center_y = (box_coords[1] + box_coords[3]) / 2

            # 计算偏移
            offset_x = abs(label_center_x - box_center_x)
            offset_y = abs(label_center_y - box_center_y)
            box_width = box_coords[2] - box_coords[0]
            box_height = box_coords[3] - box_coords[1]

            # 判断是否偏移（偏移超过纸盒宽度的20%视为偏移）
            threshold_x = box_width * 0.2
            threshold_y = box_height * 0.2

            if offset_x > threshold_x or offset_y > threshold_y:
                direction = []
                if label_center_x < box_center_x - threshold_x:
                    direction.append("偏左")
                elif label_center_x > box_center_x + threshold_x:
                    direction.append("偏右")

                if label_center_y < box_center_y - threshold_y:
                    direction.append("偏上")
                elif label_center_y > box_center_y + threshold_y:
                    direction.append("偏下")

                direction_str = "、".join(direction) if direction else "位置异常"
                report_lines.append(f"相较纸盒中心有偏移: {direction_str}")
                report_lines.append(f"水平偏移: {offset_x:.1f}px")
                report_lines.append(f"垂直偏移: {offset_y:.1f}px")
            else:
                report_lines.append("相较纸盒中心: 位置正常")
                report_lines.append(f"水平偏差: {offset_x:.1f}px")
                report_lines.append(f"垂直偏差: {offset_y:.1f}px")
        else:
            report_lines.append("无法分析偏移（需要同时检测到标签和纸盒）")

        report_lines.append("")
        report_lines.append("=" * 45)

        return "\n".join(report_lines)

    def _display_image(self, image, canvas):
        """在 Canvas 上显示图片"""
        # 转换颜色空间 BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为 PIL Image
        pil_image = Image.fromarray(image_rgb)

        # 获取 Canvas 尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400

        # 计算缩放比例（保持宽高比）
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        # 缩放图片
        pil_image_resized = pil_image.resize(new_size, Image.LANCZOS)

        # 转换为 Tkinter PhotoImage
        photo = ImageTk.PhotoImage(pil_image_resized)

        # 清空画布并显示图片
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )

        # 保存引用防止被垃圾回收
        canvas.image = photo

    def _update_result_text(self, text):
        """更新结果文本框"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def _clear(self):
        """清除结果"""
        self.current_image = None
        self.current_image_path = None
        self.detection_results = None

        # 清空画布
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")

        # 清空文本
        self._update_result_text("")

        self.status_label.config(text="状态: 就绪", fg="#7f8c8d")


def main():
    root = tk.Tk()
    app = YOLODetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
