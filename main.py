import os
import math
import random
import time

import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

from PIL import Image, ImageDraw, ImageFont



CAM_W, CAM_H = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

detector = HandDetector(detectionCon=0.75, maxHands=1)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def is_v_sign_by_landmarks(lmList):
    if lmList is None or len(lmList) < 21:
        return False

    wrist = np.array(lmList[0][0:2], dtype=np.float32)

    idx_pip = np.array(lmList[6][0:2], dtype=np.float32)
    idx_tip = np.array(lmList[8][0:2], dtype=np.float32)

    mid_pip = np.array(lmList[10][0:2], dtype=np.float32)
    mid_tip = np.array(lmList[12][0:2], dtype=np.float32)

    margin = 10.0
    idx_extended = np.linalg.norm(idx_tip - wrist) > np.linalg.norm(idx_pip - wrist) + margin
    mid_extended = np.linalg.norm(mid_tip - wrist) > np.linalg.norm(mid_pip - wrist) + margin

    tip_dist = np.linalg.norm(idx_tip - mid_tip)
    separated = tip_dist > 25.0

    return bool(idx_extended and mid_extended and separated)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def draw_hand_landmarks(img, lmList):
    if lmList is None or len(lmList) < 21:
        return
    for a, b in HAND_CONNECTIONS:
        pa = tuple(map(int, lmList[a][0:2]))
        pb = tuple(map(int, lmList[b][0:2]))
        cv2.line(img, pa, pb, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    for i in range(21):
        p = tuple(map(int, lmList[i][0:2]))
        cv2.circle(img, p, 4, (255, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA)



class FilterMode:
    NONE = 0
    CONV_SHARP = 1
    CONV_EDGE = 2
    MORPH_OPEN = 3
    MORPH_GRADIENT = 4
    COLOR_HSV = 5
    COLOR_LAB_CLAHE = 6


FILTER_NAMES_CN = {
    FilterMode.NONE: "滤镜：无",
    FilterMode.CONV_SHARP: "滤镜：锐化",
    FilterMode.CONV_EDGE: "滤镜：边缘增强",
    FilterMode.MORPH_OPEN: "滤镜：开运算",
    FilterMode.MORPH_GRADIENT: "滤镜：形态梯度",
    FilterMode.COLOR_HSV: "滤镜：HSV 视图",
    FilterMode.COLOR_LAB_CLAHE: "滤镜：CLAHE 增强",
}


def apply_filter(img, mode: int):
    if mode == FilterMode.NONE:
        return img

    if mode == FilterMode.CONV_SHARP:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    if mode == FilterMode.CONV_EDGE:
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)
        edge = cv2.filter2D(img, -1, kernel)
        return cv2.addWeighted(img, 0.75, edge, 0.55, 0)

    if mode == FilterMode.MORPH_OPEN:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, k)

    if mode == FilterMode.MORPH_GRADIENT:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)
        return cv2.addWeighted(img, 0.8, grad, 0.8, 0)

    if mode == FilterMode.COLOR_HSV:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        pseudo = cv2.merge([h, s, v])
        return cv2.cvtColor(pseudo, cv2.COLOR_HSV2BGR)

    if mode == FilterMode.COLOR_LAB_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return img


C_INK = (18, 18, 18)          
C_VERMILION = (20, 30, 220)    
C_GOLD = (0, 215, 255)         
C_JADE = (90, 190, 130)        
C_WHITE = (245, 245, 245)


PANEL_BG = (210, 230, 245)     
PANEL_EDGE = (40, 60, 180)     
PANEL_SHADOW = (10, 10, 10)



def find_windows_font():
    fonts_dir = r"C:\Windows\Fonts"
    candidates = [
        
        "simkai.ttf",     
        "simfang.ttf",    
        "simli.ttf",      
        "simsun.ttc",     
        "simsun.ttf",
        "msyh.ttc",      
        "msyh.ttf",
        "msyhbd.ttc",
    ]
    for fn in candidates:
        p = os.path.join(fonts_dir, fn)
        if os.path.exists(p):
            return p
    return None


class ChineseTextRenderer:
    def __init__(self):
        self.font_path = find_windows_font()
        if self.font_path is None:
            raise FileNotFoundError("找不到 Windows 字体文件：请确认 C:\\Windows\\Fonts 存在。")

        self._font_cache = {}

    def font(self, size):
        size = int(size)
        if size not in self._font_cache:
            self._font_cache[size] = ImageFont.truetype(self.font_path, size=size)
        return self._font_cache[size]

    @staticmethod
    def _bgr_to_pil(img_bgr):
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _pil_to_bgr(img_pil):
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_text(self, draw: ImageDraw.ImageDraw, x, y, text, size=36,
                  fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=4):
        # PIL 颜色是 RGB
        font = self.font(size)
        draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)

    def draw_panel(self, img_bgr, x1, y1, x2, y2, alpha=0.82):
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL_BG, -1, lineType=cv2.LINE_AA)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), PANEL_EDGE, 3, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x1 + 18, y1 + 18), 6, C_GOLD, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x2 - 18, y1 + 18), 6, C_GOLD, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x1 + 18, y2 - 18), 6, C_GOLD, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x2 - 18, y2 - 18), 6, C_GOLD, -1, lineType=cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    def draw_hud_run(self, img_bgr, game, filter_mode, bg_enabled, hand_ok):
        now = time.time()
        shield_text = ""
        if game.shield_active:
            shield_text = f"护盾：生效中（{max(0.0, game.shield_until - now):.1f}s）"
        else:
            cd = max(0.0, game.skill_cooldown_until - now)
            if cd > 0:
                shield_text = f"护盾冷却：{cd:.1f}s"
            else:
                shield_text = "护盾"

        lines = [
            f"分数：{game.score}",
            f"手部特效：{game.roi_modes_cn()[game.roi_mode_idx]}",
            ("手势提示：请把手放入镜头范围" if not hand_ok else ""),
            shield_text,
            f"{FILTER_NAMES_CN.get(filter_mode, '滤镜：?')} ｜ 背景：{'开' if bg_enabled else '关'}",
        ]
        lines = [s for s in lines if s.strip() != ""]

        x1, y1, x2, y2 = 28, 22, 890, 240
        img_bgr = self.draw_panel(img_bgr, x1, y1, x2, y2, alpha=0.55)

        pil = self._bgr_to_pil(img_bgr)
        draw = ImageDraw.Draw(pil)

        y = y1 + 18
        for i, s in enumerate(lines):
            size = 46 if i == 0 else 34
            self.draw_text(draw, x1 + 22, y, s, size=size,
                           fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=5)
            y += 52 if i == 0 else 40

        return self._pil_to_bgr(pil)

    def draw_menu(self, img_bgr, game):
        x1, y1 = 140, 90
        x2, y2 = CAM_W - 140, CAM_H - 120
        img_bgr = self.draw_panel(img_bgr, x1, y1, x2, y2, alpha=0.86)

        pil = self._bgr_to_pil(img_bgr)
        draw = ImageDraw.Draw(pil)

        self.draw_text(draw, x1 + 60, y1 + 30, "菜单", size=64,
                       fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=6)

        cur = game.roi_modes_cn()[game.roi_mode_idx]
        self.draw_text(draw, x1 + 60, y1 + 120, f"当前特效：{cur}", size=48,
                       fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=5)

        tips = [
            "✌（食指+中指）切换特效",
            "握拳 返回游戏",
            "五指张开 可再次进入菜单",
        ]
        yy = y1 + 200
        for t in tips:
            self.draw_text(draw, x1 + 60, yy, t, size=40,
                           fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=5)
            yy += 60

        modes = game.roi_modes_cn()
        col_x = x1 + 520
        col_y = y1 + 120
        self.draw_text(draw, col_x, col_y - 70, "可选特效：", size=40,
                       fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=5)

        for i, m in enumerate(modes):
            is_cur = (i == game.roi_mode_idx)
            fill = (255, 215, 0) if is_cur else (255, 255, 255)
            stroke = (80, 30, 0) if is_cur else (0, 0, 0)
            self.draw_text(draw, col_x, col_y + i * 46, f"{'▶ ' if is_cur else '   '}{m}",
                           size=34, fill=fill, stroke=stroke, stroke_w=4)

        self.draw_text(draw, x1 + 60, y2 - 80, f"分数：{game.score}    （按 R 可重开）",
                       size=42, fill=(255, 255, 255), stroke=(0, 0, 0), stroke_w=5)

        return self._pil_to_bgr(pil)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            raise FileNotFoundError(f"[ERROR] Food image not found: {pathFood}")
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = (0, 0)
        self.randomFoodLocation()

        self.score = 0
        self.state = "RUN"  # RUN / MENU

        self.last_fingers = None
        self.hold_frames = 0
        self.trigger_frames = 6
        self.cooldown = 0

        self.roi_modes = ["NONE", "BLUR", "EDGE", "MOSAIC", "GRAY", "SHARPEN"]
        self.roi_mode_idx = 0

        self.smoothHead = None
        self.ema_alpha = 0.35

        self.shield_active = False
        self.shield_until = 0.0
        self.skill_cooldown_until = 0.0
        self.skill_duration = 5.0
        self.skill_cooldown = 7.0
        self.flash_until = 0.0
        self.collision_ignore_until = 0.0

        self.snake_thickness = 18
        self.head_radius = 22
        self.collision_ignore_tail = 12

        self.scale_step = 22
        self.max_scales_per_frame = 160

    def roi_modes_cn(self):
        mapping = {
            "NONE": "无",
            "BLUR": "柔化",
            "EDGE": "边缘",
            "MOSAIC": "马赛克",
            "GRAY": "灰度",
            "SHARPEN": "锐化",
        }
        return [mapping.get(x, x) for x in self.roi_modes]

    def randomFoodLocation(self):
        self.foodPoint = (random.randint(100, 1000), random.randint(100, 600))

    def reset(self):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)
        self.score = 0

        self.smoothHead = None
        self.shield_active = False
        self.shield_until = 0.0
        self.skill_cooldown_until = 0.0
        self.flash_until = 0.0
        self.collision_ignore_until = 0.0

        self.state = "RUN"
        self.last_fingers = None
        self.hold_frames = 0
        self.cooldown = 0

        self.randomFoodLocation()

    
    @staticmethod
    def _safe_fingers(fingers):
        if fingers is None or not isinstance(fingers, (list, tuple)) or len(fingers) < 5:
            return None
        return list(map(int, fingers[:5]))

    @staticmethod
    def is_fist(fingers):
        f = SnakeGameClass._safe_fingers(fingers)
        return False if f is None else (sum(f) <= 1)

    @staticmethod
    def is_open_palm(fingers):
        f = SnakeGameClass._safe_fingers(fingers)
        return False if f is None else (sum(f) >= 4)

    @staticmethod
    def is_v_sign_for_cycle(fingers):
        f = SnakeGameClass._safe_fingers(fingers)
        if f is None:
            return False
        return (f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0)

    def apply_effect_to_roi(self, img, bbox):
        if bbox is None:
            return img
        x, y, w, h = bbox
        H, W = img.shape[:2]

        margin = 25
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(W, x + w + margin)
        y2 = min(H, y + h + margin)
        if x2 <= x1 or y2 <= y1:
            return img

        roi = img[y1:y2, x1:x2].copy()
        mode = self.roi_modes[self.roi_mode_idx]

        if mode == "NONE":
            return img
        elif mode == "BLUR":
            roi2 = cv2.GaussianBlur(roi, (21, 21), 0)
        elif mode == "EDGE":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            roi2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == "MOSAIC":
            scale = 0.10
            small = cv2.resize(
                roi,
                (max(1, int((x2 - x1) * scale)), max(1, int((y2 - y1) * scale))),
                interpolation=cv2.INTER_LINEAR
            )
            roi2 = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        elif mode == "GRAY":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif mode == "SHARPEN":
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            roi2 = cv2.filter2D(roi, -1, kernel)
        else:
            roi2 = roi

        img[y1:y2, x1:x2] = roi2
        return img

    def handle_menu_gesture(self, fingers):
        f = self._safe_fingers(fingers)
        if f is None:
            return

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        ft = tuple(f)
        if ft == self.last_fingers:
            self.hold_frames += 1
        else:
            self.last_fingers = ft
            self.hold_frames = 1

        if self.hold_frames < self.trigger_frames:
            return

        if self.state == "RUN" and self.is_open_palm(f):
            self.state = "MENU"
            self.cooldown = 12
            return

        if self.state == "MENU" and self.is_v_sign_for_cycle(f):
            self.roi_mode_idx = (self.roi_mode_idx + 1) % len(self.roi_modes)
            self.cooldown = 10
            return

        if self.state == "MENU" and self.is_fist(f):
            self.state = "RUN"
            self.cooldown = 12
            return

    def trigger_shield(self):
        now = time.time()
        if now < self.skill_cooldown_until:
            return
        self.shield_active = True
        self.shield_until = now + self.skill_duration
        self.skill_cooldown_until = now + self.skill_cooldown
        self.flash_until = now + 0.12

    def _smooth_head(self, currentHead):
        cx, cy = currentHead
        if self.smoothHead is None:
            self.smoothHead = (float(cx), float(cy))
        sx, sy = self.smoothHead
        a = self.ema_alpha
        sx = a * cx + (1 - a) * sx
        sy = a * cy + (1 - a) * sy
        self.smoothHead = (sx, sy)
        return int(sx), int(sy)

    
    def _draw_food_glow(self, imgMain, rx, ry):
        overlay = imgMain.copy()
        cv2.circle(
            overlay, (rx, ry),
            int(max(self.wFood, self.hFood) * 0.9),
            (255, 255, 255), -1, lineType=cv2.LINE_AA
        )
        return cv2.addWeighted(overlay, 0.10, imgMain, 0.90, 0)

    def _apply_flash(self, imgMain, now):
        if now < self.flash_until:
            overlay = imgMain.copy()
            cv2.rectangle(overlay, (0, 0), (imgMain.shape[1], imgMain.shape[0]), (255, 255, 255), -1)
            imgMain = cv2.addWeighted(overlay, 0.07, imgMain, 0.93, 0)
        return imgMain

    
    def _draw_chinese_body(self, img):
        if len(self.points) < 2:
            return

        pts = [tuple(p) for p in self.points]

        
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], C_INK, self.snake_thickness + 6, lineType=cv2.LINE_AA)

        
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], C_VERMILION, self.snake_thickness, lineType=cv2.LINE_AA)

       
        scale_color = C_GOLD if self.shield_active else (40, 190, 230)
        outline = (30, 60, 160)

        remain = float(self.scale_step)
        flip = 0
        drawn = 0

        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue

            ux, uy = dx / seg_len, dy / seg_len
            nx, ny = -uy, ux

            dist = seg_len
            sx, sy = float(x0), float(y0)

            while dist >= remain:
                if drawn >= self.max_scales_per_frame:
                    return

                px = sx + ux * remain
                py = sy + uy * remain

                offset = (self.snake_thickness * 0.20) * (1 if flip == 0 else -1)
                cx = int(px + nx * offset)
                cy = int(py + ny * offset)

                r = max(6, int(self.snake_thickness * 0.40))
                ang = int(math.degrees(math.atan2(dy, dx)))
                start, end = ((200, 340) if flip == 0 else (20, 160))

                cv2.ellipse(img, (cx, cy), (r, int(r * 0.78)), ang, start, end,
                            scale_color, 2, lineType=cv2.LINE_AA)
                cv2.ellipse(img, (cx, cy), (r, int(r * 0.78)), ang, start, end,
                            outline, 1, lineType=cv2.LINE_AA)

                if self.shield_active:
                    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

                drawn += 1
                flip ^= 1

                sx, sy = px, py
                dist -= remain
                remain = float(self.scale_step)

            remain -= dist

        if self.shield_active:
            overlay = img.copy()
            for i in range(1, len(pts)):
                cv2.line(overlay, pts[i - 1], pts[i], (0, 255, 255),
                         self.snake_thickness + 14, lineType=cv2.LINE_AA)
            img[:] = cv2.addWeighted(overlay, 0.04, img, 0.96, 0)

    
    def _draw_chinese_head(self, img, center):
        x, y = center
        r = self.head_radius

        if self.shield_active:
            glow = img.copy()
            cv2.circle(glow, (x, y), r + 26, (0, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(glow, (x, y), r + 14, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            img[:] = cv2.addWeighted(glow, 0.06, img, 0.94, 0)

        cv2.circle(img, (x, y), r, C_VERMILION, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), r, C_GOLD, 3, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), r + 3, C_INK, 2, lineType=cv2.LINE_AA)

        cv2.ellipse(img, (x, y), (r - 4, r - 4), 0, 35, 215, C_WHITE, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (r - 4, r - 4), 0, 215, 395, C_INK, 2, lineType=cv2.LINE_AA)
        cv2.circle(img, (x - r // 4, y), 4, C_INK, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x + r // 4, y), 4, C_WHITE, -1, lineType=cv2.LINE_AA)

        cv2.ellipse(img, (x - r // 2, y - r // 3), (r // 3, r // 5), 0, 200, 20, C_GOLD, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x - r // 2 + 8, y - r // 3 + 2), (r // 4, r // 6), 0, 200, 30, C_GOLD, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x + r // 2, y - r // 3), (r // 3, r // 5), 0, 160, 340, C_GOLD, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x + r // 2 - 8, y - r // 3 + 2), (r // 4, r // 6), 0, 150, 330, C_GOLD, 2, lineType=cv2.LINE_AA)

        cv2.circle(img, (x, y + r // 2), 3, C_JADE, -1, lineType=cv2.LINE_AA)

        if self.shield_active:
            cv2.circle(img, (x, y), r + 10, (0, 255, 255), 3, lineType=cv2.LINE_AA)

    def update(self, imgMain, currentHead, fingers=None, bbox=None):
        now = time.time()
        self.handle_menu_gesture(fingers)

        if self.shield_active and now > self.shield_until:
            self.shield_active = False

        
        if self.state == "MENU":
            imgMain = self.apply_effect_to_roi(imgMain, bbox)
            return imgMain

       
        imgMain = self.apply_effect_to_roi(imgMain, bbox)

        if currentHead is None:
            return imgMain

        cx, cy = self._smooth_head(currentHead)
        if self.previousHead == (0, 0) and len(self.points) == 0:
            self.previousHead = (cx, cy)

        px, py = self.previousHead
        self.points.append([cx, cy])
        dist = math.hypot(cx - px, cy - py)
        self.lengths.append(dist)
        self.currentLength += dist
        self.previousHead = (cx, cy)

        if self.currentLength > self.allowedLength:
            for i, l in enumerate(self.lengths):
                self.currentLength -= l
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break

        rx, ry = self.foodPoint
        if (rx - self.wFood // 2 < cx < rx + self.wFood // 2) and (ry - self.hFood // 2 < cy < ry + self.hFood // 2):
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1

        self._draw_chinese_body(imgMain)
        if self.points:
            self._draw_chinese_head(imgMain, tuple(self.points[-1]))

        imgMain = self._draw_food_glow(imgMain, rx, ry)
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

       
        if now >= self.collision_ignore_until:
            body = np.array(self.points[:-self.collision_ignore_tail], np.int32)
            if len(body) >= 3:
                body = body.reshape((-1, 1, 2))
                minDist = cv2.pointPolygonTest(body, (cx, cy), True)
                hit_thresh = self.snake_thickness // 2
                if -hit_thresh <= minDist <= hit_thresh:
                    if self.shield_active:
                        self.shield_active = False
                        self.flash_until = now + 0.12
                        self.collision_ignore_until = now + 0.6
                    else:
                        self.reset()

        imgMain = self._apply_flash(imgMain, now)
        return imgMain


def load_background(path="bg.jpg", w=CAM_W, h=CAM_H):
    bg = cv2.imread(path)
    if bg is None:
        return None
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)



game = SnakeGameClass("donut.png")
bg_img = load_background("bg.jpg")
bg_enabled = True
filter_mode = FilterMode.NONE

text = ChineseTextRenderer()

last_vsign_time = 0.0
VSIGN_DEBOUNCE = 0.25

while True:
    ok, img = cap.read()
    if not ok:
        continue
    img = cv2.flip(img, 1)

    raw_img = img.copy()

    
    if bg_enabled and bg_img is not None:
        render_img = bg_img.copy()
    else:
        render_img = img.copy()

    
    hands, _ = detector.findHands(raw_img, flipType=False, draw=False)

    current_head = None
    fingers = None
    bbox = None
    hand_ok = False

    if hands:
        hand_ok = True
        hand = hands[0]
        lmList = hand.get("lmList", None)
        bbox = hand.get("bbox", None)
        try:
            fingers = detector.fingersUp(hand)
        except Exception:
            fingers = None

        if lmList is not None and len(lmList) >= 21:
            
            draw_hand_landmarks(render_img, lmList)

            current_head = tuple(lmList[8][0:2])
            cv2.circle(render_img, current_head, 10, (255, 0, 255), cv2.FILLED, lineType=cv2.LINE_AA)

            now = time.time()
           
            if game.state == "RUN":
                if is_v_sign_by_landmarks(lmList) and (now - last_vsign_time) > VSIGN_DEBOUNCE:
                    game.trigger_shield()
                    last_vsign_time = now

    
    render_img = game.update(render_img, current_head, fingers=fingers, bbox=bbox)

    
    render_img = apply_filter(render_img, filter_mode)

    
    if game.state == "MENU":
        render_img = text.draw_menu(render_img, game)
    else:
        render_img = text.draw_hud_run(render_img, game, filter_mode, (bg_enabled and bg_img is not None), hand_ok)

    cv2.imshow("Image", render_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        game.reset()
    elif key == ord("b"):
        bg_enabled = not bg_enabled
    elif key == ord("f"):
        filter_mode = (filter_mode + 1) % 7

cap.release()
cv2.destroyAllWindows()
