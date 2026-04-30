import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

# 导入颜色分析模块（替代 OCR）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from util.energy_grade_analyzer import EnergyGradeAnalyzer


class YOLODetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 能效标签智能检测系统 v3.0 (颜色识别版)")
        self.root.geometry("1920x1080")

        # 项目路径配置
        self.project_root = Path(__file__).resolve().parent

        # 查找最新的模型
        self.model_path = self._find_latest_model()

        # 类别名称映射
        self.class_names = {
            0: 'energy_arrow',
            1: 'label',
            2: 'box',
            3: 'stain',
            4: 'fold',
            5: 'brakeage'
        }

        # 显示名称映射
        self.display_names = {
            'energy_arrow': {'type': 'grade', 'display': '能效箭头', 'icon': '[箭头]'},
            'label': {'type': 'label', 'display': '能效标签', 'icon': '[标签]'},
            'box': {'type': 'box', 'display': '产品纸盒', 'icon': '[纸盒]'},
            'stain': {'type': 'defect', 'display': '污渍', 'icon': '[污渍]'},
            'fold': {'type': 'defect', 'display': '褶皱', 'icon': '[褶皱]'},
            'brakeage': {'type': 'defect', 'display': '破损', 'icon': '[破损]'}
        }

        # 初始化分析器（开启调试模式可保存中间图）
        self.analyzer = EnergyGradeAnalyzer(debug_mode=False)

        # 系统状态
        self.model = None
        self.label_font = self._load_chinese_font(18)
        self.current_image = None
        self.current_image_path = None
        self.detection_results = None

        # 初始化界面
        self._init_ui()

        # 加载模型
        self._load_model()

    def _find_latest_model(self):
        """查找最新的训练模型"""
        runs_dir = self.project_root / "runs" / "detect"
        if not runs_dir.exists():
            return None

        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
        if not train_dirs:
            return None

        latest = max(train_dirs, key=lambda d: d.stat().st_mtime)
        model_path = latest / "weights" / "best.pt"
        return model_path if model_path.exists() else None

    def _init_ui(self):
        """初始化用户界面"""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：控制面板
        left_panel = tk.Frame(main_frame, width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # 右侧：图像显示
        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # === 左侧面板 ===
        title_label = tk.Label(
            left_panel,
            text="能效标签检测系统\n(颜色识别版)",
            font=("Microsoft YaHei", 16, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=(0, 15))

        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        for text, cmd, color in [
            ("选择图片", self._load_image, "#3498db"),
            ("开始检测", self._detect, "#27ae60"),
            ("清除结果", self._clear, "#95a5a6")
        ]:
            tk.Button(
                btn_frame, text=text, command=cmd,
                font=("Microsoft YaHei", 11), bg=color, fg="white",
                relief=tk.FLAT, padx=10, pady=5, cursor="hand2"
            ).pack(fill=tk.X, pady=2)

        self.status_label = tk.Label(
            left_panel,
            text="状态: 正在加载模型...",
            font=("Microsoft YaHei", 9),
            fg="#7f8c8d",
            anchor=tk.W,
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.X, pady=(10, 5))

        separator1 = tk.Frame(left_panel, height=2, bg="#ecf0f1")
        separator1.pack(fill=tk.X, pady=10)

        result_title = tk.Label(
            left_panel,
            text="检测结果",
            font=("Microsoft YaHei", 13, "bold"),
            fg="#34495e",
            anchor=tk.W
        )
        result_title.pack(fill=tk.X, pady=(0, 5))

        result_frame = tk.Frame(left_panel)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(
            result_frame,
            font=("Microsoft YaHei", 13),
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

        # === 右侧面板 ===
        image_container = tk.Frame(right_panel)
        image_container.pack(fill=tk.BOTH, expand=True)

        # 原图
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

        # 检测结果
        result_frame_display = tk.LabelFrame(
            image_container,
            text="检测结果 (颜色识别)",
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

    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            if self.model_path is None or not self.model_path.exists():
                self.status_label.config(text="状态: 模型未找到", fg="#e74c3c")
                messagebox.showwarning("警告", "未找到训练好的模型文件\n\n请先运行训练或检查模型路径")
                return

            self.model = YOLO(str(self.model_path))
            self.status_label.config(text="状态: 系统就绪 (颜色识别模式)", fg="#27ae60")
            print(f"✅ 模型加载成功: {self.model_path.name}")
            print("✅ 能效等级识别: 基于国家标准颜色编码 (无需 OCR)")

        except Exception as e:
            self.status_label.config(text="状态: 模型加载失败", fg="#e74c3c")
            messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")

    def _load_chinese_font(self, size=18):
        """加载中文字体用于图片标注，避免 OpenCV 中文乱码"""
        font_candidates = [
            Path("C:\\Windows\\Fonts\\msyh.ttc"),
            Path("C:\\Windows\\Fonts\\msyhbd.ttc"),
            Path("C:\\Windows\\Fonts\\simhei.ttf"),
            Path("C:\\Windows\\Fonts\\simsun.ttc"),
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
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                raise ValueError("无法读取图片")

            self.current_image_path = Path(file_path)
            self._display_image(self.current_image, self.original_canvas)

            # 清空结果
            self.result_canvas.delete("all")
            self.result_canvas.create_text(
                self.result_canvas.winfo_width() // 2,
                self.result_canvas.winfo_height() // 2,
                text="点击「开始检测」进行分析",
                fill="white",
                font=("Microsoft YaHei", 12)
            )
            self._update_result_text("")

            self.status_label.config(
                text=f"状态: 已加载 {self.current_image_path.name}",
                fg="#27ae60"
            )
        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败:\n{str(e)}")

    def _detect(self):
        """执行检测（核心：颜色识别替代 OCR）"""
        if self.model is None:
            messagebox.showwarning("警告", "模型未加载，请稍候再试")
            return
        if self.current_image is None:
            messagebox.showwarning("警告", "请先选择一张图片")
            return

        try:
            self.status_label.config(text="状态: 正在检测...", fg="#f39c12")
            self.root.update()

            # YOLO 检测
            results = self.model.predict(
                source=str(self.current_image_path),
                conf=0.25,
                save=False,
                verbose=False
            )
            self.detection_results = results[0]

            # 绘制检测结果
            result_image = self._draw_detections(self.current_image.copy(), self.detection_results)
            self._display_image(result_image, self.result_canvas)

            # === 核心：颜色识别能效等级 ===
            grade_info = self._recognize_grade_by_color(self.detection_results)

            # 生成分析报告
            analysis = self._analyze_results(self.detection_results, grade_info)
            self._update_result_text(analysis)

            self.status_label.config(text="状态: 检测完成 (颜色识别)", fg="#27ae60")

        except Exception as e:
            self.status_label.config(text="状态: 检测失败", fg="#e74c3c")
            messagebox.showerror("错误", f"检测失败:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _recognize_grade_by_color(self, results):
        """
        通过颜色识别能效等级（核心方法）

        返回:
            {
                'grade': '三级',
                'confidence': 0.68,
                'method': 'color',
                'debug_info': '三级 (黄色) 占比: 68.2%'
            } 或 None
        """
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        names = results.names
        arrow_boxes = []

        # 收集所有 energy_arrow 区域
        for box in boxes:
            cls_id = int(box.cls[0])
            if names[cls_id] == 'energy_arrow':
                arrow_boxes.append(box.xyxy[0].cpu().numpy())

        if not arrow_boxes:
            return None

        print(f"\n🔍 颜色分析: 检测到 {len(arrow_boxes)} 个箭头区域")

        # 多区域投票机制（提高鲁棒性）
        votes = []
        confidences = []

        for i, box in enumerate(arrow_boxes):
            grade, conf, debug_info = self.analyzer.analyze_from_box(
                self.current_image,
                box,
                padding=25,
                region_name=f"arrow_{i}"
            )

            print(f"  区域 {i + 1}: {debug_info} (置信度: {conf:.1%})")

            if grade and conf >= self.analyzer.CONFIDENCE_THRESHOLD:
                votes.append(grade)
                confidences.append(conf)

        # 投票决策
        if votes:
            # 统计各等级得票
            from collections import Counter
            vote_count = Counter(votes)
            total_votes = len(votes)

            # 选择得票最高的等级
            best_grade, vote_num = vote_count.most_common(1)[0]
            avg_conf = sum(confidences) / len(confidences)

            # 验证是否达到投票阈值
            if vote_num / total_votes >= self.analyzer.VOTING_THRESHOLD:
                return {
                    'grade': best_grade,
                    'confidence': avg_conf,
                    'method': 'color',
                    'debug_info': f"{best_grade} (得票: {vote_num}/{total_votes}, 平均置信度: {avg_conf:.1%})"
                }

        # 单区域 fallback
        if arrow_boxes:
            grade, conf, debug_info = self.analyzer.analyze_from_box(
                self.current_image,
                arrow_boxes[0],
                padding=30,
                region_name="fallback"
            )
            if grade and conf >= self.analyzer.CONFIDENCE_THRESHOLD * 0.6:  # 允许更低置信度输出
                return {
                    'grade': grade,
                    'confidence': conf,
                    'method': 'color_fallback',
                    'debug_info': debug_info
                }

        return None

    def _draw_detections(self, image, results):
        """绘制检测结果（增强：在箭头区域标注识别到的等级）"""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return image

        image_h, image_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        font = self.label_font if self.label_font else ImageFont.load_default()

        colors = {
            'grade': (0, 255, 0),  # 绿色
            'label': (255, 255, 0),  # 黄色
            'box': (0, 255, 255),  # 青色
            'defect': (255, 0, 0),  # 红色
        }

        names = results.names
        grade_info = self._recognize_grade_by_color(self.detection_results) if self.detection_results is not None else None

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names[cls_id]

            display_info = self.display_names.get(class_name, {})
            category_type = display_info.get('type', 'defect')
            color = colors.get(category_type, (255, 255, 255))

            # 绘制边界框
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

            # 绘制标签
            display_name = display_info.get('display', class_name)
            label = f"{display_name} {conf:.2f}"

            # 如果是 energy_arrow 且已识别等级，添加等级标注
            if class_name == 'energy_arrow' and grade_info is not None:
                if grade_info and grade_info.get('grade'):
                    grade_desc = EnergyGradeAnalyzer.get_grade_description(grade_info['grade'])
                    label += f"\n{grade_desc}"

            lines = label.split('\n')
            line_heights = []
            text_w_max = 0
            line_gap = 2

            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_w_max = max(text_w_max, text_w)
                line_heights.append(text_h)

            text_h_total = sum(line_heights) + line_gap * (len(lines) - 1)
            box_left = max(0, x1)
            box_right = min(image_w, box_left + text_w_max + 10)
            box_top = y1 - text_h_total - 10
            box_bottom = y1

            if box_top < 0:
                box_top = y1
                box_bottom = min(image_h, y1 + text_h_total + 10)

            draw.rectangle([(box_left, box_top), (box_right, box_bottom)], fill=color)

            y_cursor = box_top + 5
            for line, line_h in zip(lines, line_heights):
                draw.text((box_left + 5, y_cursor), line, fill=(0, 0, 0), font=font)
                y_cursor += line_h + line_gap

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _analyze_results(self, results, grade_info=None):
        """生成检测报告"""
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return "未检测到任何目标\n\n请确保图片中包含能效标签和产品包装"

        names = results.names
        detected_classes = {}

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names[cls_id]
            if class_name not in detected_classes:
                detected_classes[class_name] = []
            detected_classes[class_name].append({'confidence': conf, 'box': box.xyxy[0].cpu().numpy()})

        report = []
        report.append("=" * 45)
        report.append("能效标签检测报告 (颜色识别版)")
        report.append("=" * 45)
        report.append("")

        # 1. 能耗等级（颜色识别结果）
        report.append("【能耗等级】")
        if grade_info and grade_info.get('grade'):
            grade = grade_info['grade']
            grade_desc = EnergyGradeAnalyzer.get_grade_description(grade)
            report.append(f"识别结果: {grade_desc}")
            report.append(f"识别方式: 颜色分析 (国家标准 GB 21455)")
            report.append(f"置信度: {grade_info['confidence']:.1%}")
            report.append(f"详细: {grade_info.get('debug_info', '')}")
        elif 'energy_arrow' in detected_classes:
            report.append("检测到能效箭头区域")
            max_conf = max(d['confidence'] for d in detected_classes['energy_arrow'])
            report.append(f"箭头检测置信度: {max_conf:.2%}")
            report.append("提示: 未识别到明确等级颜色（可能光照/遮挡影响）")
        else:
            report.append("未检测到能效箭头区域")
        report.append("")

        # 2. 标签与包装
        report.append("【标签与包装】")
        if 'label' in detected_classes:
            max_conf = max(d['confidence'] for d in detected_classes['label'])
            report.append(f"能效标签: 已检测到 (置信度: {max_conf:.2%})")
        else:
            report.append("能效标签: 未检测到")

        if 'box' in detected_classes:
            max_conf = max(d['confidence'] for d in detected_classes['box'])
            report.append(f"产品纸盒: 已检测到 (置信度: {max_conf:.2%})")
        else:
            report.append("产品纸盒: 未检测到")
        report.append("")

        # 3. 质量检测
        report.append("【质量检测】")
        defects_found = False
        for defect_type in ['brakeage', 'stain', 'fold']:
            if defect_type in detected_classes:
                defects_found = True
                count = len(detected_classes[defect_type])
                max_conf = max(d['confidence'] for d in detected_classes[defect_type])
                defect_name = self.display_names.get(defect_type, {}).get('display', defect_type)
                report.append(f"{defect_name}: 发现 {count} 处 (最高置信度: {max_conf:.2%})")

        if not defects_found:
            report.append("未发现明显缺陷")
        report.append("")

        # 4. 位置分析
        report.append("【位置分析】")
        if 'label' in detected_classes and 'box' in detected_classes:
            label_box = detected_classes['label'][0]['box']
            box_coords = detected_classes['box'][0]['box']

            label_center_x = (label_box[0] + label_box[2]) / 2
            label_center_y = (label_box[1] + label_box[3]) / 2
            box_center_x = (box_coords[0] + box_coords[2]) / 2
            box_center_y = (box_coords[1] + box_coords[3]) / 2

            offset_x = abs(label_center_x - box_center_x)
            offset_y = abs(label_center_y - box_center_y)
            box_width = box_coords[2] - box_coords[0]
            box_height = box_coords[3] - box_coords[1]

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
                report.append(f"标签位置: {'、'.join(direction) if direction else '异常'}")
                report.append(f"水平偏移: {offset_x:.1f}px | 垂直偏移: {offset_y:.1f}px")
            else:
                report.append("标签位置: 正常")
                report.append(f"水平偏差: {offset_x:.1f}px | 垂直偏差: {offset_y:.1f}px")
        else:
            report.append("无法分析位置（需同时检测到标签和纸盒）")

        report.append("")
        report.append("=" * 45)
        report.append("💡 说明: 能效等级通过国家标准颜色编码识别")
        report.append("   一级: 深绿 | 二级: 浅绿 | 三级: 黄 | 四级: 橙 | 五级: 红")

        return "\n".join(report)

    def _display_image(self, image, canvas):
        """在 Canvas 上显示图片"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 860, 620

        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        pil_image_resized = pil_image.resize(new_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image_resized)

        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # 防止垃圾回收

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

        self.original_canvas.delete("all")
        self.result_canvas.delete("all")
        self._update_result_text("")

        self.status_label.config(text="状态: 就绪 (颜色识别模式)", fg="#7f8c8d")


def main():
    root = tk.Tk()
    # 设置高 DPI 支持 (Windows)
    if sys.platform == 'win32':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)

    app = YOLODetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
