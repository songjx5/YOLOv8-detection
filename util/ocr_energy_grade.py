"""
OCR 能效等级识别模块
使用 PaddleOCR 识别能效标签箭头区域的文字（一级、二级等）
"""

import cv2
import numpy as np
import re
import os


class EnergyGradeOCR:
    def __init__(self, use_gpu=False):
        """
        初始化 OCR 引擎

        参数:
            use_gpu: 是否使用 GPU 加速
        """
        self._fatal_backend_error = False
        self._fatal_error_message = None

        # 避免部分 Paddle CPU 环境触发 oneDNN + PIR 兼容问题
        if not use_gpu:
            os.environ.setdefault("FLAGS_use_mkldnn", "0")

        try:
            from paddleocr import PaddleOCR

            # 适配 PaddleOCR 3.x 新版本
            if use_gpu:
                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang='ch',
                    show_log=False,
                    device='gpu'
                )
            else:
                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang='ch',
                    show_log=False,
                    device='cpu'
                )

            self.available = True
            print("✅ PaddleOCR 初始化成功")
        except ImportError as e:
            missing_module = getattr(e, "name", "") or ""
            if missing_module in ("paddleocr", "paddle"):
                print("⚠️  PaddleOCR 未安装，OCR 功能不可用")
                print("   安装命令: pip install paddlepaddle paddleocr")
            else:
                print(f"⚠️  PaddleOCR 依赖缺失: {missing_module or str(e)}")
                print("   请安装缺失依赖后重试（例如: pip install setuptools）")
            self.available = False
            self.ocr = None
        except Exception as e:
            print(f"⚠️  PaddleOCR 初始化失败: {e}")
            self.available = False
            self.ocr = None

    def recognize_grade(self, image_region):
        """
        识别能效等级文字

        参数:
            image_region: 箭头区域的图像（numpy array）

        返回:
            str: 识别到的等级文字（如 "一级", "二级" 等），未识别到返回 None
        """
        if not self.available or self.ocr is None or self._fatal_backend_error:
            return None

        if image_region is None or image_region.size == 0:
            return None

        try:
            # 快速路径：优先保证速度
            variants = self._build_variants(image_region, aggressive=False)
            collected_texts = []

            for img in variants:
                result = self._run_ocr(img)
                texts = self._collect_texts(result, min_confidence=0.3)
                if not texts:
                    continue

                full_text = "".join(texts)
                grade = self._extract_grade(full_text)
                if grade:
                    return grade

                collected_texts.extend(texts)

            # 兜底路径：快速路径失败后再走更强预处理
            aggressive_variants = self._build_variants(image_region, aggressive=True)
            for img in aggressive_variants:
                result = self._run_ocr(img)
                texts = self._collect_texts(result, min_confidence=0.3)
                if not texts:
                    continue
                full_text = "".join(texts)
                grade = self._extract_grade(full_text)
                if grade:
                    return grade
                collected_texts.extend(texts)

            if collected_texts:
                return self._extract_grade("".join(collected_texts))

            return None

        except Exception as e:
            print(f"OCR 识别错误: {e}")
            return None

    def _run_ocr(self, image):
        """兼容 PaddleOCR 不同版本接口"""
        try:
            if hasattr(self.ocr, "ocr"):
                try:
                    return self.ocr.ocr(image, cls=False)
                except TypeError:
                    return self.ocr.ocr(image)

            if hasattr(self.ocr, "predict"):
                return self.ocr.predict(image)

            raise RuntimeError("当前 PaddleOCR 版本不支持 ocr/predict 接口")
        except Exception as e:
            if self._is_fatal_backend_error(e):
                self._fatal_backend_error = True
                self._fatal_error_message = str(e)
                self.available = False
                print("⚠️  OCR 后端与当前 Paddle 运行时不兼容，已自动禁用 OCR。")
                print("   建议升级/对齐 paddlepaddle 与 paddleocr 版本。")
            raise

    def _is_fatal_backend_error(self, err):
        msg = str(err)
        return (
            "ConvertPirAttribute2RuntimeAttribute" in msg or
            ("onednn_instruction.cc" in msg and "Unimplemented" in msg)
        )

    def _build_variants(self, image_region, aggressive=False):
        """构造多种预处理图像，提升文字识别率"""
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        # 反锐化，提升轻微模糊场景的边缘清晰度
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        sharpen = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu)
        _, clahe_otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        adaptive_dilate = cv2.dilate(adaptive, kernel, iterations=1)

        # PaddleOCR 对 3 通道图像更稳定，统一转为 BGR
        variants = []
        if aggressive:
            base_candidates = [
                clahe,
                sharpen,
                adaptive,
                otsu,
            ]
        else:
            base_candidates = [
                image_region,
                clahe,
                adaptive,
            ]

        for candidate in base_candidates:
            if len(candidate.shape) == 2:
                candidate = cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR)

            h, w = candidate.shape[:2]
            scales = [1.0]
            # 小字区域优先放大，显著提升 OCR 召回
            if min(h, w) < 220:
                scales.append(2.0)
                if aggressive and min(h, w) < 160:
                    scales.append(3.0)
            elif min(h, w) < 360 and aggressive:
                scales.append(2.0)

            for s in scales:
                if s == 1.0:
                    variants.append(candidate)
                    continue
                new_w = int(w * s)
                new_h = int(h * s)
                # 控制上限，避免过大图像拖慢 OCR
                if max(new_w, new_h) > 1800:
                    continue
                resized = cv2.resize(candidate, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                variants.append(resized)

        return variants

    def _collect_texts(self, result, min_confidence=0.5):
        """从不同 OCR 返回结构中提取文本"""
        texts = []

        def add_text(text, confidence=None):
            if text is None:
                return
            if confidence is not None and confidence < min_confidence:
                return
            text = str(text).strip()
            if text:
                texts.append(text)

        def walk(node):
            if node is None:
                return

            if isinstance(node, dict):
                if isinstance(node.get("rec_text"), str):
                    add_text(node.get("rec_text"), node.get("rec_score"))
                if isinstance(node.get("text"), str):
                    add_text(node.get("text"), node.get("score"))

                rec_texts = node.get("rec_texts")
                rec_scores = node.get("rec_scores")
                if isinstance(rec_texts, (list, tuple)):
                    for i, txt in enumerate(rec_texts):
                        score = None
                        if isinstance(rec_scores, (list, tuple)) and i < len(rec_scores):
                            score = rec_scores[i]
                        add_text(txt, score)

                for value in node.values():
                    if isinstance(value, (list, tuple, dict)):
                        walk(value)
                return

            if isinstance(node, (list, tuple)):
                # 兼容经典结构: [bbox, (text, score)]
                if (
                    len(node) >= 2 and
                    isinstance(node[1], (list, tuple)) and
                    len(node[1]) >= 2 and
                    isinstance(node[1][0], str)
                ):
                    add_text(node[1][0], node[1][1])
                    return

                for item in node:
                    walk(item)

        walk(result)
        return texts

    def _extract_grade(self, text):
        """
        从 OCR 结果中提取能效等级

        参数:
            text: OCR 识别的完整文字

        返回:
            str: 标准化后的等级文字
        """
        if not text:
            return None

        normalized = str(text)
        normalized = normalized.replace(" ", "").replace("\n", "")
        normalized = normalized.replace("壹", "一").replace("贰", "二").replace("叁", "三")
        normalized = normalized.replace("肆", "四").replace("伍", "五")
        normalized = normalized.replace("l", "1").replace("I", "1")

        digit_grade_map = {'1': '一级', '2': '二级', '3': '三级', '4': '四级', '5': '五级'}
        cn_grade_map = {'一': '一级', '二': '二级', '三': '三级', '四': '四级', '五': '五级'}

        match = re.search(r'([1-5])级', normalized)
        if match:
            return digit_grade_map[match.group(1)]

        match = re.search(r'([一二三四五])级', normalized)
        if match:
            return cn_grade_map[match.group(1)]

        match = re.search(r'能效([1-5])', normalized)
        if match:
            return digit_grade_map[match.group(1)]

        match = re.search(r'能效([一二三四五])', normalized)
        if match:
            return cn_grade_map[match.group(1)]

        # 常见等级表述模式
        grade_patterns = {
            '一级': ['一级', '1级', '能效1', '能效一级'],
            '二级': ['二级', '2级', '能效2', '能效二级'],
            '三级': ['三级', '3级', '能效3', '能效三级'],
            '四级': ['四级', '4级', '能效4', '能效四级'],
            '五级': ['五级', '5级', '能效5', '能效五级'],
        }

        for grade_name, patterns in grade_patterns.items():
            for pattern in patterns:
                if pattern in normalized:
                    return grade_name

        return None

    def recognize_from_box(self, full_image, box_coords, padding=10):
        """
        从完整图片中根据边界框裁剪并识别

        参数:
            full_image: 完整图片（numpy array）
            box_coords: 边界框 [x1, y1, x2, y2]
            padding: 裁剪时的额外边距

        返回:
            str: 识别到的等级
        """
        x1, y1, x2, y2 = map(int, box_coords)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        dynamic_padding = max(int(max(box_w, box_h) * 0.8), padding)
        dynamic_padding = min(dynamic_padding, 120)

        # 添加边距并防止越界
        h, w = full_image.shape[:2]
        x1 = max(0, x1 - dynamic_padding)
        y1 = max(0, y1 - dynamic_padding)
        x2 = min(w, x2 + dynamic_padding)
        y2 = min(h, y2 + dynamic_padding)

        # 裁剪区域
        region = full_image[y1:y2, x1:x2]
        if region.size == 0:
            return None

        # 小区域先放大，缓解分辨率不足问题
        rh, rw = region.shape[:2]
        if min(rh, rw) < 200:
            scale = max(2.0, 220.0 / max(1.0, min(rh, rw)))
            new_w = min(2200, int(rw * scale))
            new_h = min(2200, int(rh * scale))
            region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 识别
        return self.recognize_grade(region)


def test_ocr():
    """测试 OCR 功能"""
    print("="*60)
    print("OCR 能效等级识别测试")
    print("="*60)

    ocr = EnergyGradeOCR(use_gpu=False)

    if not ocr.available:
        print("\n❌ OCR 功能不可用，请先安装 PaddleOCR")
        return

    # 测试文字提取
    test_cases = [
        "中国能效标识 一级",
        "能效等级: 2级",
        "叁级能效",
        "能效4级产品",
    ]

    print("\n📝 文字提取测试:")
    for text in test_cases:
        grade = ocr._extract_grade(text)
        print(f"  '{text}' → {grade}")

    print("\n" + "="*60)


if __name__ == "__main__":
    test_ocr()
