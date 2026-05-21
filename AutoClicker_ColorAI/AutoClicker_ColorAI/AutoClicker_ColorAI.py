
"""
Reward-Trained AutoClicker (Color-Blob + Money-ROI Reward)
---------------------------------------------------------
Detects clickables (coins/gems/chests/crates/presents) via HSV color blobs.
Chooses a candidate, clicks it, waits 2 seconds, and gives reward=1 if:
  - the money HUD ROI changed (pHash Hamming distance >= threshold), OR
  - histogram correlation shows change.

It prints debug output each click so you can see when it rewarded/trained.

Hotkeys:
- 8: Toggle bot on/off
- 7: Calibrate money ROI (9 = top-left, 0 = bottom-right)
- ESC: Quit

Requirements:
pip install dxcam opencv-python numpy pyautogui pywin32 keyboard scikit-learn pydirectinput
"""

import time
import math
from dataclasses import dataclass
from collections import deque

import cv2
import dxcam
import keyboard
import numpy as np
import pyautogui
import win32gui
import win32con
import os
import joblib
import pydirectinput as pdi

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

pyautogui.FAILSAFE = False
pdi.FAILSAFE = False

# ---------------------------- Window / Capture ----------------------------

def find_window_rect(title: str):
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        raise RuntimeError(f'Window "{title}" not found. Open it first.')
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    return hwnd, (left, top, right, bot)

class ScreenGrabber:
    """
    Uses full-screen dxcam capture, then crops to the Roblox window.
    """
    def __init__(self, rect):
        self.left, self.top, self.right, self.bot = rect
        self.w = self.right - self.left
        self.h = self.bot - self.top
        self.cam = dxcam.create(output_color="BGR")
        self.cam.start(target_fps=60)
        # Give camera a moment to initialize
        time.sleep(0.1)

    def grab(self):
        try:
            frame = self.cam.get_latest_frame()
            if frame is None:
                return None

            H, W = frame.shape[:2]
            crop_left = max(0, min(self.left, W))
            crop_top = max(0, min(self.top, H))
            crop_right = max(crop_left, min(self.right, W))
            crop_bot = max(crop_top, min(self.bot, H))

            if crop_right > crop_left and crop_bot > crop_top:
                return frame[crop_top:crop_bot, crop_left:crop_right]
            return None
        except KeyboardInterrupt:
            raise
        except Exception:
            return None

    def stop(self):
        """Stop the camera capture"""
        try:
            self.cam.stop()
        except:
            pass

# ---------------------------- Candidate Generation (Color Blobs) ----------------------------

@dataclass
class Candidate:
    x: int
    y: int
    hint: float
    patch: np.ndarray

def _cc_candidates(mask, frame_bgr, patch_size=64, area_min=120, area_max=50000, hint_boost=1.0):
    h, w = frame_bgr.shape[:2]
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < area_min or area > area_max:
            continue

        cx, cy = cents[i]
        cx, cy = int(cx), int(cy)

        if cx < patch_size // 2 or cy < patch_size // 2 or cx > w - patch_size // 2 or cy > h - patch_size // 2:
            continue

        patch = frame_bgr[
            cy - patch_size // 2:cy + patch_size // 2,
            cx - patch_size // 2:cx + patch_size // 2
        ].copy()

        # Hint favors bigger + squarer-ish blobs
        squareness = 1.0 - abs(1.0 - (ww / (hh + 1e-6)))
        hint = hint_boost * (0.6 * (area / 1500.0) + 0.4 * squareness)
        out.append(Candidate(cx, cy, float(hint), patch))

    return out

def generate_candidates(frame_bgr, patch_size=64, max_candidates=35):
    """
    Loot detection by HSV color masks - tuned for specific items:
    - Bluish-green coins: cyan/teal colored coins
    - Pinkish-purple crates: magenta/pink rectangular blocks
    - Pinkish-purple presents: magenta/pink gift boxes
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Bluish-green coins - cyan/teal colors (Hue 80-100, high saturation and brightness)
    mask_coins = cv2.inRange(hsv, (80, 100, 150), (100, 255, 255))

    # Pinkish-purple crates and presents - magenta/pink colors (Hue 140-170)
    mask_chests = cv2.inRange(hsv, (140, 100, 140), (170, 255, 255))  # Pinkish-purple crates
    mask_presents = cv2.inRange(hsv, (145, 110, 150), (165, 255, 255))  # Pinkish-purple presents
    
    # Combine all pinkish-purple items (crates and presents)
    mask_purple = cv2.bitwise_or(mask_chests, mask_presents)

    # Clean masks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_coins = cv2.morphologyEx(mask_coins, cv2.MORPH_OPEN, k, iterations=1)
    mask_coins = cv2.morphologyEx(mask_coins, cv2.MORPH_CLOSE, k, iterations=1)

    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, k, iterations=1)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, k, iterations=1)

    cands = []
    # Bluish-green coins - smaller, more numerous
    cands += _cc_candidates(mask_coins, frame_bgr, patch_size=patch_size, area_min=80, area_max=20000, hint_boost=1.2)
    # Pinkish-purple items (crates and presents) - various sizes, can be large
    cands += _cc_candidates(mask_purple, frame_bgr, patch_size=patch_size, area_min=100, area_max=50000, hint_boost=1.0)

    # Dedup close centers (lots of blobs overlap)
    cands.sort(key=lambda c: c.hint, reverse=True)
    picked = []
    for c in cands:
        if all((c.x - p.x) ** 2 + (c.y - p.y) ** 2 > 22 ** 2 for p in picked):
            picked.append(c)
        if len(picked) >= max_candidates:
            break

    return picked

# ---------------------------- Features + Online Learner ----------------------------

def extract_features(patch_bgr):
    patch = cv2.resize(patch_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_feat = hog.compute(gray).reshape(-1).astype(np.float32)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).reshape(-1)
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).reshape(-1)
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).reshape(-1)
    color_feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    color_feat /= (color_feat.sum() + 1e-6)

    return np.concatenate([hog_feat, color_feat])

class OnlineClickModel:
    def __init__(self):
        self.scaler = StandardScaler(with_mean=False)
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-5, penalty="l2")
        self.is_fit = False
        self._warmup_X = []
        self._warmup_y = []

    def predict_proba(self, X):
        if not self.is_fit:
            return np.full((X.shape[0],), 0.5, dtype=np.float32)
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1].astype(np.float32)

    def update(self, X, y):
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)

        if not self.is_fit:
            self._warmup_X.append(Xs)
            self._warmup_y.append(y)
            ys = np.concatenate(self._warmup_y)
            if len(np.unique(ys)) < 2:
                return
            Xbuf = np.vstack(self._warmup_X)
            self.clf.partial_fit(Xbuf, ys, classes=np.array([0, 1]))
            self.is_fit = True
            self._warmup_X.clear()
            self._warmup_y.clear()
        else:
            self.clf.partial_fit(Xs, y)

    def save(self, path="model_state.joblib"):
        joblib.dump({
            "scaler": self.scaler,
            "clf": self.clf,
            "is_fit": self.is_fit
        }, path)

    def load(self, path="model_state.joblib"):
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.clf = data["clf"]
        self.is_fit = data.get("is_fit", True)
        # warmup buffers don’t need to persist
        self._warmup_X.clear()
        self._warmup_y.clear()
        return True

# ---------------------------- Reward: Money ROI Change (+ Local Change) ----------------------------

def phash64(gray_roi):
    """Perceptual hash - more sensitive to change"""
    g = cv2.resize(gray_roi, (32, 32), interpolation=cv2.INTER_AREA)
    g = np.float32(g)
    d = cv2.dct(g)
    d8 = d[:8, :8]
    # Use mean instead of median for more sensitivity to changes
    # Exclude DC component (0,0) which is just average brightness
    mean_val = np.mean(d8[1:, 1:])
    return (d8 > mean_val).reshape(-1)

def hamming(a, b):
    return int(np.count_nonzero(a != b))

def get_roi(frame_bgr, roi):
    x0, y0, x1, y1 = roi
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(frame_bgr.shape[1], x1); y1 = min(frame_bgr.shape[0], y1)
    return frame_bgr[y0:y1, x0:x1]

def sample_money_sig(grabber, money_roi, samples=6, dt=0.03):
    acc = None
    got = 0
    for _ in range(samples):
        fr = grabber.grab()
        if fr is None:
            time.sleep(dt)
            continue
        roi = get_roi(fr, money_roi)
        if roi.size == 0:
            time.sleep(dt)
            continue
        # Ensure ROI has valid dimensions
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            time.sleep(dt)
            continue
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        acc = g.astype(np.float32) if acc is None else (acc + g.astype(np.float32))
        got += 1
        time.sleep(dt)

    if got == 0:
        return None

    avg = (acc / got).astype(np.uint8)
    return phash64(avg)

def local_change_score(frame_before, frame_after, cx, cy, box=70):
    h, w = frame_before.shape[:2]
    x0 = max(0, cx - box // 2); x1 = min(w, cx + box // 2)
    y0 = max(0, cy - box // 2); y1 = min(h, cy + box // 2)

    b = cv2.cvtColor(frame_before[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    a = cv2.cvtColor(frame_after[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    return float(cv2.absdiff(a, b).mean())

def reward_money_plus_local(grabber, money_roi, frame_before, clicked_xy,
                            wait_seconds=2.0, money_ham_thresh=3, local_diff_thresh=3.0):
    """
    Check if money changed after clicking.
    Timing:
    1. Baseline: Uses frame_before (captured BEFORE the click)
    2. Waits wait_seconds (default 2.0)
    3. After wait: Grabs new frame and compares to baseline
    So it's checking: "Money before click" vs "Money after 2 second wait"
    """
    # Get baseline from frame_before (captured BEFORE click)
    roi_before = get_roi(frame_before, money_roi)
    
    # pHash baseline from frame_before
    sig_before = None
    if roi_before.size > 0 and roi_before.shape[0] >= 10 and roi_before.shape[1] >= 10:
        gray_before = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY)
        gray_before = cv2.GaussianBlur(gray_before, (3, 3), 0)
        sig_before = phash64(gray_before)
    
    # Histogram baseline from frame_before
    hist_before = None
    if roi_before.size > 0 and roi_before.shape[0] >= 10 and roi_before.shape[1] >= 10:
        gray_before = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY)
        hist_before = cv2.calcHist([gray_before], [0], None, [256], [0, 256])
    
    if sig_before is None:
        time.sleep(wait_seconds)
        return 0, {"money_ham": None, "local_diff": None, "money_changed": False, "local_changed": False}

    # Wait the specified time (money should update during this wait)
    time.sleep(wait_seconds)

    frame_after = grabber.grab()
    if frame_after is None:
        return 0, {"money_ham": None, "local_diff": None, "money_changed": False, "local_changed": False}

    sig_after = sample_money_sig(grabber, money_roi)
    if sig_after is None:
        return 0, {"money_ham": None, "local_diff": None, "money_changed": False, "local_changed": False}

    money_ham = hamming(sig_before, sig_after)
    
    # Also check histogram correlation as secondary method
    hist_changed = False
    if hist_before is not None:
        roi_after = get_roi(frame_after, money_roi)
        if roi_after.size > 0 and roi_after.shape[0] >= 10 and roi_after.shape[1] >= 10:
            gray_after = cv2.cvtColor(roi_after, cv2.COLOR_BGR2GRAY)
            hist_after = cv2.calcHist([gray_after], [0], None, [256], [0, 256])
            # Correlation: 1.0 = identical, lower = different
            hist_corr = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CORREL)
            # If correlation drops below 0.95, money likely changed
            hist_changed = (hist_corr < 0.95)
    
    # Money changed if pHash OR histogram shows change
    money_changed = (money_ham >= money_ham_thresh) or hist_changed

    cx, cy = clicked_xy
    local_diff = local_change_score(frame_before, frame_after, cx, cy)
    local_changed = (local_diff >= local_diff_thresh)

    # Reward = 1 if money changed (no longer requires local change)
    reward = 1 if money_changed else 0
    dbg = {
        "money_ham": money_ham,
        "hist_changed": hist_changed,
        "local_diff": local_diff,
        "money_changed": money_changed,
        "local_changed": local_changed,
        "money_thresh": money_ham_thresh,
        "local_thresh": local_diff_thresh
    }
    return reward, dbg

# ---------------------------- ROI Calibration ----------------------------

def calibrate_roi(grabber):
    print("ROI calibration: hover TOP-LEFT of gold money digits, press 9.")
    print("Then hover BOTTOM-RIGHT of gold money digits, press 0.")
    tl = None
    while True:
        if keyboard.is_pressed("esc"):
            raise SystemExit

        if keyboard.is_pressed("9"):
            mx, my = pyautogui.position()
            tl = (mx - grabber.left, my - grabber.top)
            print("Top-left set:", tl)
            time.sleep(0.25)

        if keyboard.is_pressed("0") and tl is not None:
            mx, my = pyautogui.position()
            br = (mx - grabber.left, my - grabber.top)
            x0, y0 = tl
            x1, y1 = br
            roi = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            print("Money ROI:", roi)
            time.sleep(0.25)
            return roi

        time.sleep(0.01)

# ---------------------------- Game Click (Roblox-Optimized) ----------------------------

def focus_window(hwnd):
    if win32gui.IsIconic(hwnd):  # If minimized, restore
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)

def is_diamond_pile(patch_bgr):
    """
    Check if a candidate patch is a bluish-green coin pile.
    Returns True if the patch matches bluish-green coin color range.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return False
    
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    # Bluish-green coin range: (80, 100, 150) to (100, 255, 255) - cyan/teal colors
    mask_coin = cv2.inRange(hsv, (80, 100, 150), (100, 255, 255))
    # If more than 20% of the patch matches coin color, it's likely a coin pile
    coin_ratio = np.sum(mask_coin > 0) / (patch_bgr.shape[0] * patch_bgr.shape[1])
    return coin_ratio > 0.20

def game_click_cluster(hwnd, abs_x, abs_y, grid_size=3, spacing=4):
    """
    Cluster click function for bluish-green coin piles - clicks in a small grid pattern.
    """
    focus_window(hwnd)
    
    # Calculate grid offsets
    grid_half = (grid_size - 1) // 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            offset_x = (i - grid_half) * spacing
            offset_y = (j - grid_half) * spacing
            
            click_x = abs_x + offset_x
            click_y = abs_y + offset_y
            
            # Move to position
            pdi.moveTo(click_x, click_y, _pause=False)
            
            # Jiggle
            pdi.moveRel(3, 0, _pause=False)
            time.sleep(0.01)
            pdi.moveRel(-3, 0, _pause=False)
            time.sleep(0.01)
            
            # Click
            pdi.mouseDown()
            time.sleep(0.05)
            pdi.mouseUp()
            time.sleep(0.02)  # Small delay between cluster clicks

def game_click(hwnd, abs_x, abs_y):
    """
    Roblox-optimized click function using DirectInput for better registration.
    """
    focus_window(hwnd)
    
    # 1. Warp to target
    pdi.moveTo(abs_x, abs_y, _pause=False)

    # 2. Jiggle to update Roblox's raw cursor (make it more pronounced)
    pdi.moveRel(5, 0, _pause=False)  # Increased to 5px for visibility
    time.sleep(0.02)  # Slightly longer delay
    pdi.moveRel(-5, 0, _pause=False)
    time.sleep(0.02)

    # 3. Slightly long press for shaky hitboxes
    pdi.mouseDown()
    time.sleep(0.06)  # Increased to 60ms for reliability
    pdi.mouseUp()

# ---------------------------- Main Loop ----------------------------

def main():
    WIN_TITLE = "Roblox"
    hwnd, rect = find_window_rect(WIN_TITLE)
    grabber = ScreenGrabber(rect)
    model = OnlineClickModel()

    MODEL_PATH = "model_state.joblib"
    if model.load(MODEL_PATH):
        print("Loaded saved model_state.joblib")
    else:
        print("No saved model yet (starting fresh)")

    enabled = False

    # Backup ROI from your screenshot (relative to Roblox window crop).
    # You SHOULD calibrate with 7 for your exact setup.
    money_roi = (1759, 634, 1992, 688)

    click_cooldown = 0.75
    next_click_time = 0.0

    # Track recent clicks to avoid clicking same area without rewards
    recent_clicks = deque(maxlen=15)  # Track last 15 clicks: (x, y, reward, time)
    DIVERSITY_RADIUS = 80  # Pixels - if clicking within this radius without rewards, force diversity
    UNREWARDED_PENALTY = 0.5  # Penalty multiplier for candidates near unrewarded clicks
    MIN_DIVERSITY_DISTANCE = 150  # Minimum distance to force when stuck

    # Avoid clicking on UI area (right HUD, bottom icons, and inventory)
    def in_ui_zone(x, y):
        # Right side HUD
        if x > grabber.w * 0.80:
            return True
        # Bottom area (inventory and icons) - expanded to bottom 15%
        if y > grabber.h * 0.85:
            return True
        # Left side bottom area
        if x < grabber.w * 0.05 and y > grabber.h * 0.60:
            return True
        # Bottom-center area (inventory zone) - 30% to 70% width, bottom 20%
        if (grabber.w * 0.30 < x < grabber.w * 0.70) and y > grabber.h * 0.80:
            return True
        return False

    print("7 calibrates money ROI (recommended). 8 toggles bot. ESC quits.")

    try:
        while True:
            if keyboard.is_pressed("esc"):
                break

            if keyboard.is_pressed("7"):
                money_roi = calibrate_roi(grabber)
                time.sleep(0.25)

            if keyboard.is_pressed("8"):
                enabled = not enabled
                print("ENABLED" if enabled else "DISABLED")
                time.sleep(0.25)

            frame = grabber.grab()
            if frame is None or not enabled:
                time.sleep(0.01)
                continue

            now = time.time()
            if now < next_click_time:
                time.sleep(0.01)
                continue

            cands = generate_candidates(frame, patch_size=64, max_candidates=35)
            cands = [c for c in cands if not in_ui_zone(c.x, c.y)]
            if not cands:
                next_click_time = time.time() + 0.25
                continue

            X = np.vstack([extract_features(c.patch) for c in cands])
            proba = model.predict_proba(X)

            hint = np.array([c.hint for c in cands], dtype=np.float32)
            hint = (hint - hint.min()) / (np.ptp(hint) + 1e-6)

            # Calculate center proximity bonus (STRONGLY favor candidates closer to screen center)
            screen_center_x = grabber.w / 2.0
            screen_center_y = grabber.h / 2.0
            center_distances = np.array([
                math.sqrt((c.x - screen_center_x)**2 + (c.y - screen_center_y)**2) 
                for c in cands
            ], dtype=np.float32)
            
            # Strong center bias: candidates far from center get heavily penalized
            # Use exponential decay - candidates far from center get much lower scores
            max_possible_dist = math.sqrt(screen_center_x**2 + screen_center_y**2)
            # Normalize to 0-1, then apply exponential (closer = much higher)
            normalized_dist = center_distances / (max_possible_dist + 1e-6)
            center_proximity = np.exp(-3.0 * normalized_dist)  # Exponential decay - strong center bias
            center_proximity = (center_proximity - center_proximity.min()) / (np.ptp(center_proximity) + 1e-6)
            
            # Additional penalty for candidates in outer 30% of screen (likely rocks/side items)
            edge_threshold = max_possible_dist * 0.8  # Outer 30% of screen
            edge_penalty = np.ones(len(cands), dtype=np.float32)
            for i, dist in enumerate(center_distances):
                if dist > edge_threshold:
                    edge_penalty[i] = 0.3  # Heavy penalty for edge candidates

            # Check if we're stuck clicking in same area without rewards
            unrewarded_clicks = [(x, y) for x, y, r, t in recent_clicks if r == 0]
            stuck_in_area = len(unrewarded_clicks) >= 3  # 3+ unrewarded clicks in recent history
            
            # Apply diversity penalty to candidates near unrewarded clicks
            diversity_scores = np.ones(len(cands), dtype=np.float32)
            if unrewarded_clicks:
                for i, c in enumerate(cands):
                    min_dist_to_unrewarded = float('inf')
                    for ux, uy in unrewarded_clicks:
                        dist = math.sqrt((c.x - ux)**2 + (c.y - uy)**2)
                        min_dist_to_unrewarded = min(min_dist_to_unrewarded, dist)
                    
                    # Penalize if too close to unrewarded clicks
                    if min_dist_to_unrewarded < DIVERSITY_RADIUS:
                        diversity_scores[i] = UNREWARDED_PENALTY * 0.5  # Even stronger penalty
                    # If stuck, heavily favor candidates far from unrewarded area AND close to center
                    elif stuck_in_area and min_dist_to_unrewarded < MIN_DIVERSITY_DISTANCE:
                        diversity_scores[i] = 0.2  # Very strong penalty for nearby candidates when stuck
                    elif stuck_in_area:
                        # Only give bonus if also close to center (avoid side rocks)
                        dist_to_center = math.sqrt((c.x - screen_center_x)**2 + (c.y - screen_center_y)**2)
                        if dist_to_center < max_possible_dist * 0.5:  # Within inner 50% of screen
                            diversity_scores[i] = 1.5  # Bonus for far from unrewarded AND close to center
                        else:
                            diversity_scores[i] = 0.5  # Less bonus if far from center

            # Mix "learned" score + heuristic + center proximity + diversity
            # Prioritize: ML score (60%) + hint (10%) + center proximity (30%) - STRONG center bias
            final = 0.60 * proba + 0.10 * hint + 0.30 * center_proximity
            final = final * diversity_scores  # Apply diversity multiplier
            final = final * edge_penalty  # Apply edge penalty (heavily penalize side rocks)
            best = cands[int(np.argmax(final))]

            # Click using Roblox-optimized game_click function
            abs_x = grabber.left + best.x
            abs_y = grabber.top + best.y
            
            # Check if it's a diamond pile - use cluster clicking for better reliability
            if is_diamond_pile(best.patch):
                game_click_cluster(hwnd, abs_x, abs_y, grid_size=3, spacing=4)
            else:
                game_click(hwnd, abs_x, abs_y)

            # Reward after 2 seconds
            reward, dbg = reward_money_plus_local(
                grabber, money_roi, frame, (best.x, best.y),
                wait_seconds=2.0,
                money_ham_thresh=3,  # Very sensitive - detects small money changes
                local_diff_thresh=3.0  # More lenient
            )

            # Print debug so you KNOW when it rewarded
            money_ham = dbg["money_ham"]
            hist_changed = dbg.get("hist_changed", False)
            local_diff = dbg["local_diff"]
            money_thresh = dbg.get("money_thresh", 3)
            local_thresh = dbg.get("local_thresh", 3.0)

            money_ham_str = "None" if money_ham is None else f"{money_ham}/{money_thresh}"
            hist_str = "hist✓" if hist_changed else "hist✗"
            local_diff_str = "None" if local_diff is None else f"{local_diff:.2f}/{local_thresh:.1f}"

            # Show why reward was given/denied
            reward_reason = []
            if dbg['money_changed']:
                reward_reason.append("money")
            if dbg['local_changed']:
                reward_reason.append("local")
            reason_str = "+".join(reward_reason) if reward_reason else "none"

            print(
                f"Click @({best.x},{best.y}) "
                f"reward={reward} ({reason_str}) | "
                f"money_ham={money_ham_str} {hist_str} | "
                f"local_diff={local_diff_str} | "
                f"cooldown={click_cooldown:.1f}s"
            )

            # Track this click for diversity enforcement
            recent_clicks.append((best.x, best.y, reward, time.time()))
            
            # Show diversity status if stuck (recalculate after adding this click)
            current_unrewarded = [(x, y) for x, y, r, t in recent_clicks if r == 0]
            if len(current_unrewarded) >= 3:
                print(f"  [DIVERSITY] {len(current_unrewarded)} recent clicks had no reward - forcing diversity")

            # Update model
            x1 = extract_features(best.patch).reshape(1, -1)
            model.update(x1, np.array([reward], dtype=np.int64))

            next_click_time = time.time() + click_cooldown
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        grabber.stop()
        print("Saving model...")
        model.save(MODEL_PATH)
        print("Exiting.")

if __name__ == "__main__":
    main()
