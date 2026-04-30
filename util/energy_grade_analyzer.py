"""
能效等级颜色分析模块
基于国家标准 GB 21455 的颜色编码识别能效等级
无需 OCR，抗模糊/破损/光照干扰
"""

import cv2
import numpy as np
from pathlib import Path


class EnergyGradeAnalyzer:
    """能效等级颜色分析器"""

    # 国家标准颜色定义 (HSV 色彩空间)
    GRADE_COLORS = {
        '一级': {
            'hsv_ranges': [
                (np.array([40, 40, 40]), np.array([80, 255, 255]))  # 深绿
            ],
            'rgb_example': (34, 139, 34),
            'description': '深绿色'
        },
        '二级': {
            'hsv_ranges': [
                (np.array([50, 30, 100]), np.array([85, 150, 255]))  # 浅绿
            ],
            'rgb_example': (144, 238, 144),
            'description': '浅绿色'
        },
        '三级': {
            'hsv_ranges': [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))  # 黄色
            ],
            'rgb_example': (255, 215, 0),
            'description': '黄色'
        },
        '四级': {
            'hsv_ranges': [
                (np.array([10, 100, 100]), np.array([25, 255, 255]))  # 橙色
            ],
            'rgb_example': (255, 165, 0),
            'description': '橙色'
        },
        '五级': {
            'hsv_ranges': [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),   # 红1
                (np.array([170, 100, 100]), np.array([180, 255, 255])) # 红2 (HSV 红色环绕)
            ],
            'rgb_example': (255, 0, 0),
            'description': '红色'
        }
    }

    # 置信度阈值 (区域中目标颜色占比)
    CONFIDENCE_THRESHOLD = 0.08
    # 多区域投票时的最小支持比例
    VOTING_THRESHOLD = 0.5

    def __init__(self, debug_mode=False):
        """
        初始化分析器

        参数:
            debug_mode: 是否保存调试图像（裁剪区域、掩码等）
        """
        self.debug_mode = debug_mode
        self.debug_dir = Path(__file__).parent / "debug_color"
        if debug_mode and not self.debug_dir.exists():
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def analyze_from_box(self, image, box_coords, padding=25, region_name="arrow"):
        """
        从边界框分析能效等级

        参数:
            image: 原始图像 (BGR)
            box_coords: [x1, y1, x2, y2]
            padding: 裁剪扩展像素
            region_name: 区域名称（用于调试文件名）

        返回:
            (等级, 置信度, 调试信息) 或 (None, 0, "")
        """
        # 裁剪区域
        region = self._crop_region(image, box_coords, padding)
        if region is None or region.size == 0:
            return None, 0, "裁剪区域无效"

        # 保存调试图
        if self.debug_mode:
            cv2.imwrite(str(self.debug_dir / f"{region_name}_crop.jpg"), region)

        # 分析主色调
        grade, confidence, debug_info = self._analyze_region_color(region)

        if self.debug_mode and grade:
            self._save_debug_visualization(region, grade, confidence, region_name)

        return grade, confidence, debug_info

    def _crop_region(self, image, box_coords, padding):
        """安全裁剪区域"""
        x1, y1, x2, y2 = map(int, box_coords)
        h, w = image.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2].copy()

    def _analyze_region_color(self, region):
        """
        分析区域主色调

        返回:
            (等级, 置信度, 调试信息)
        """
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # 计算各等级颜色占比
        grade_scores = {}
        for grade, config in self.GRADE_COLORS.items():
            mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

            # 合并多个 HSV 范围（如红色有两个范围）
            for lower, upper in config['hsv_ranges']:
                mask = cv2.inRange(hsv, lower, upper)
                mask_total = cv2.bitwise_or(mask_total, mask)

            # 形态学处理：去除噪点
            kernel = np.ones((3, 3), np.uint8)
            mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
            mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

            # 计算占比
            color_pixels = np.sum(mask_total > 0)
            ratio = color_pixels / total_pixels if total_pixels > 0 else 0
            grade_scores[grade] = ratio

            # 保存掩码调试图
            if self.debug_mode:
                mask_bgr = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(self.debug_dir / f"mask_{grade}.jpg"), mask_bgr)

        # 选择最高占比且超过阈值的等级
        best_grade = None
        max_ratio = 0
        for grade, ratio in grade_scores.items():
            if ratio > max_ratio and ratio >= self.CONFIDENCE_THRESHOLD:
                max_ratio = ratio
                best_grade = grade

        if best_grade:
            debug_info = f"{best_grade} ({self.GRADE_COLORS[best_grade]['description']}) 占比: {max_ratio:.1%}"
            return best_grade, max_ratio, debug_info
        else:
            # 返回最高占比（即使低于阈值）用于调试
            if grade_scores:
                fallback_grade = max(grade_scores, key=grade_scores.get)
                debug_info = f"无明确等级 (最高: {fallback_grade} {grade_scores[fallback_grade]:.1%})"
                return None, grade_scores[fallback_grade], debug_info
            return None, 0, "区域无有效颜色"

    def _save_debug_visualization(self, region, grade, confidence, region_name):
        """保存带标注的调试可视化图"""
        vis = region.copy()
        h, w = vis.shape[:2]

        # 绘制等级标签
        label = f"{grade} ({confidence:.0%})"
        font_scale = min(1.0, w / 200)
        thickness = max(1, int(w / 300))

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(vis, (5, 5), (text_w + 15, text_h + 15), (0, 0, 0), -1)
        cv2.putText(vis, label, (10, text_h + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 绘制颜色示例块
        color_bgr = tuple(reversed(self.GRADE_COLORS[grade]['rgb_example']))
        cv2.rectangle(vis, (w - 60, 10), (w - 10, 60), color_bgr, -1)

        cv2.imwrite(str(self.debug_dir / f"{region_name}_result.jpg"), vis)

    @staticmethod
    def get_grade_description(grade):
        """获取等级描述文本"""
        descriptions = {
            '一级': '一级能效 (最佳)',
            '二级': '二级能效 (较节能)',
            '三级': '三级能效 (中等)',
            '四级': '四级能效 (较低)',
            '五级': '五级能效 (最低)'
        }
        return descriptions.get(grade, grade)
