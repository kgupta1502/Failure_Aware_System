"""
Optimized Multi-Pillar Failure-Aware Safety Supervisory System  v2
===================================================================
Changelog vs v1:
  - Pillar 4 (Proximity Alert) added to fix pedestrian detection lag
  - Hard SRI override when a pedestrian occupies > 60% hazard proximity
  - Sudden-appearance detector: pedestrian count jumps 0 → N in 1 frame
  - Danger-zone bounding box highlights drawn over YOLO annotation
  - Updated weights: 0.25 V + 0.40 E + 0.20 T + 0.15 P  (sum = 1.0)
  - GPU batch inference hint for Pillar 2 dual stream

Root cause of the original lag (fixed here):
  Pillars 1-3 are model HEALTH metrics, not scene DANGER metrics.
  In good daylight with a clear model detection (high confidence),
  all three pillars read LOW → SRI stays LOW → system says SAFE even
  while a pedestrian stands directly in the vehicle's path.
  Pillar 4 measures the pedestrian's proximity directly.
"""

import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO


class OptimizedSafetySystem:
    """
    Four-pillar System Risk Index (SRI ∈ [0, 1]).

    Pillar 1 – Visibility Hazard     : RMS contrast of the scene.
    Pillar 2 – Epistemic Uncertainty  : Model-output variance under perturbation.
    Pillar 3 – Temporal Instability   : SRI volatility over recent frames.
    Pillar 4 – Proximity Alert  (NEW) : Pedestrian bbox area + centre-lane zone.

    SRI > safety_threshold  →  CRITICAL RISK, recommend driver handover.
    """

    # ─── Danger zone: centre 50 % of frame width ─────────────────────────
    ZONE_LEFT  = 0.25
    ZONE_RIGHT = 0.75

    # ─── Proximity calibration ────────────────────────────────────────────
    # BBox area < PROX_MIN_AREA (% of frame) → no alert
    # BBox area > PROX_MAX_AREA (% of frame) → full hazard
    PROX_MIN_AREA = 0.01   # ≈ pedestrian at 15 m
    PROX_MAX_AREA = 0.10   # ≈ pedestrian at 3–5 m

    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, model_path: str, safety_threshold: float = 0.45,
                 history_len: int = 60):
        self.model            = YOLO(model_path)
        self.safety_threshold = safety_threshold
        self.sri_history      = deque(maxlen=history_len)
        self.fps_buffer       = deque(maxlen=30)
        self.frame_count      = 0

        # Pedestrian count history (for sudden-appearance detection)
        self._ped_history: deque = deque(maxlen=10)

        # Discover pedestrian class IDs once at startup
        self._person_ids: set = {
            idx for idx, name in self.model.names.items()
            if name.lower() in ('person', 'pedestrian', 'ped', 'people')
        }

        print("=" * 62)
        print("  Optimized Safety Supervisory System  v2")
        print(f"  Model          : {model_path}")
        print(f"  Threshold      : {safety_threshold}")
        print(f"  History        : {history_len} frames")
        if self._person_ids:
            names = [self.model.names[i] for i in self._person_ids]
            print(f"  Pedestrian IDs : {self._person_ids}  {names}")
        else:
            print("  WARNING: no 'person'/'pedestrian' class found in model")
        print("=" * 62)

    # ─── Pillar 1 ─────────────────────────────────────────────────────────
    def _pillar_visibility(self, frame: np.ndarray) -> float:
        """
        RMS contrast of the grayscale frame.
        Low contrast (fog / darkness / occlusion) → high hazard.

        Calibration:
          gray.std() < 15  → near-blackout → hazard = 1.0
          gray.std() > 60  → crystal-clear → hazard = 0.0
        """
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = float(gray.std())
        hazard   = 1.0 - (np.clip(contrast, 15.0, 60.0) - 15.0) / 45.0
        return float(hazard)

    # ─── Pillar 2 ─────────────────────────────────────────────────────────
    def _pillar_epistemic(self, conf_orig: np.ndarray,
                          conf_pert: np.ndarray) -> float:
        """
        Prediction variance between a clean and a gamma-shifted frame.

        High divergence → model operates near its decision boundary →
        unsafe to rely on.

        Fixes Code-2's un-normalised count-delta bug by dividing by
        max(n1, n2) instead of a hardcoded constant.
        """
        has_orig = len(conf_orig) > 0
        has_pert = len(conf_pert) > 0

        if has_orig != has_pert:     # one stream saw nothing → high alarm
            return 0.65
        if not has_orig:             # both empty → agree on clear scene
            return 0.0

        count_ratio = (abs(len(conf_orig) - len(conf_pert))
                       / max(len(conf_orig), len(conf_pert)))
        conf_delta  = abs(float(np.mean(conf_orig)) - float(np.mean(conf_pert)))
        return float(np.clip(0.5 * count_ratio + 0.5 * conf_delta, 0.0, 1.0))

    # ─── Pillar 3 ─────────────────────────────────────────────────────────
    def _pillar_temporal(self) -> float:
        """
        Std deviation of the SRI ring buffer, scaled to [0, 1].
        High volatility = rapidly changing scene or model instability.
        Requires ≥ 5 frames of history; returns 0 otherwise.
        """
        if len(self.sri_history) < 5:
            return 0.0
        return float(np.clip(np.std(self.sri_history) * 4.0, 0.0, 1.0))

    # ─── Pillar 4 (NEW) ───────────────────────────────────────────────────
    def _pillar_proximity(self, frame: np.ndarray, results) -> tuple:
        """
        Directly measures pedestrian danger from bounding box geometry.

        WHY THIS EXISTS:
          Pillars 1-3 measure MODEL HEALTH, not SCENE DANGER. In good
          daylight with a confident detection the model can be working
          perfectly while a pedestrian stands 2 m ahead.  The existing
          pillars all read LOW → SRI stays LOW → false SAFE.

        LOGIC:
          1. For each person/pedestrian detection above conf threshold:
             a. area_ratio  = bbox_area / frame_area  (distance proxy)
             b. area_score  = scaled to [0, 1] using PROX calibration
             c. centre_zone = is the pedestrian in the vehicle's path?
                              → multiplier ×1.35 if yes
          2. Sudden appearance: pedestrian count jumps 0 → N in 1 frame
             → floor hazard at 0.50 even if bbox is still small
          3. Return (max_hazard, list_of_danger_boxes) for HUD rendering.

        Returns:
            (proximity_hazard: float, danger_boxes: list[tuple])
        """
        H, W    = frame.shape[:2]
        frame_area = H * W

        if not self._person_ids or len(results.boxes) == 0:
            self._ped_history.append(0)
            return 0.0, []

        classes    = results.boxes.cls.cpu().numpy().astype(int)
        boxes      = results.boxes.xyxy.cpu().numpy()
        confs      = results.boxes.conf.cpu().numpy()

        max_hazard  = 0.0
        ped_count   = 0
        danger_boxes = []

        for i, cls in enumerate(classes):
            if cls not in self._person_ids:
                continue
            if confs[i] < 0.35:               # filter very uncertain hits
                continue

            ped_count += 1
            x1, y1, x2, y2 = boxes[i]

            # 1. Proximity from bbox area
            area_ratio  = (x2 - x1) * (y2 - y1) / frame_area
            area_score  = float(np.clip(
                (area_ratio - self.PROX_MIN_AREA) /
                (self.PROX_MAX_AREA - self.PROX_MIN_AREA),
                0.0, 1.0
            ))

            # 2. Centre-lane zone multiplier
            cx = (x1 + x2) / 2.0
            in_zone = (W * self.ZONE_LEFT <= cx <= W * self.ZONE_RIGHT)
            if in_zone:
                area_score = min(area_score * 1.35, 1.0)

            if area_score > max_hazard:
                max_hazard = area_score

            # Collect boxes above a minimal threshold for HUD highlights
            if area_score > 0.10 or in_zone:
                danger_boxes.append((int(x1), int(y1), int(x2), int(y2),
                                     area_score, in_zone))

        # 3. Sudden-appearance override
        prev_avg = float(np.mean(self._ped_history)) if self._ped_history else 0.0
        if prev_avg < 0.5 and ped_count > 0:
            max_hazard = max(max_hazard, 0.50)   # floor for sudden entry

        self._ped_history.append(ped_count)
        return float(np.clip(max_hazard, 0.0, 1.0)), danger_boxes

    # ─── SRI ──────────────────────────────────────────────────────────────
    def _compute_sri(self, vis: float, epist: float, temporal: float,
                     mean_conf: float, proximity: float = 0.0) -> float:
        """
        Unified System Risk Index from four pillars.

        Weights (sum to 1.0):
          0.25 × visibility hazard
          0.40 × max(epistemic uncertainty, confidence deficit)
          0.20 × temporal instability
          0.15 × proximity hazard

        Hard override:
          When a pedestrian is critically close (proximity > 0.60), the
          weighted average is not enough — other 'healthy' pillars dilute
          it. The override forces  SRI = max(SRI, proximity × 0.90)
          so a close pedestrian ALWAYS triggers CRITICAL regardless of
          how good the model's other metrics look.
        """
        conf_deficit = 1.0 - mean_conf
        sri = (0.25 * vis
               + 0.40 * max(epist, conf_deficit)
               + 0.20 * temporal
               + 0.15 * proximity)

        # Hard proximity override — cannot be diluted by healthy pillars
        if proximity > 0.60:
            sri = max(sri, proximity * 0.90)

        return float(np.clip(sri, 0.0, 1.0))

    # ─── Colour helper ────────────────────────────────────────────────────
    def _sri_color(self, sri: float) -> tuple:
        """BGR gradient: green → amber → red."""
        if sri < 0.30:
            return (30, 210, 30)
        if sri < self.safety_threshold:
            return (0, 165, 255)
        return (40, 40, 220)

    # ─── HUD helpers ──────────────────────────────────────────────────────
    def _draw_metric_bar(self, img, x, y, w, h, value, color, label):
        """Labelled progress bar."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
        fw = int(w * np.clip(value, 0.0, 1.0))
        if fw > 0:
            cv2.rectangle(img, (x, y), (x + fw, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (110, 110, 110), 1)
        cv2.putText(img, f"{label}  {value:.3f}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_sparkline(self, img, x, y, w, h):
        """SRI trend chart with dashed threshold line."""
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
        cv2.putText(img, "SRI trend (recent frames)",
                    (x + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (130, 130, 130), 1, cv2.LINE_AA)

        th_y = int(y + h - 4 - self.safety_threshold * (h - 22))
        for sx in range(x + 6, x + w - 6, 9):
            cv2.line(img, (sx, th_y),
                     (min(sx + 5, x + w - 6), th_y), (60, 60, 200), 1)
        cv2.putText(img, f"thr={self.safety_threshold:.2f}",
                    (x + w - 60, th_y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (80, 80, 200), 1, cv2.LINE_AA)

        hist = list(self.sri_history)
        if len(hist) < 2:
            return
        pts = []
        for i, v in enumerate(hist):
            px = x + 5 + int(i * (w - 10) / (len(hist) - 1))
            py = int(y + h - 4 - np.clip(v, 0.0, 1.0) * (h - 22))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i],
                     self._sri_color(hist[i]), 2, cv2.LINE_AA)

    def _draw_proximity_highlights(self, img, danger_boxes):
        """
        Draw an orange/red bounding box around pedestrians in the danger zone.
        This is drawn ON TOP of the standard YOLO boxes so it's clearly visible.
        """
        for (x1, y1, x2, y2, score, in_zone) in danger_boxes:
            # Colour: red if in centre lane, orange otherwise
            color   = (0, 60, 255) if in_zone else (0, 140, 255)
            thickness = 3 if in_zone else 2

            # Outer warning box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Proximity score badge above the box
            label   = f"PED {score:.0%}"
            tw      = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            bx1, by1 = x1, max(y1 - 22, 0)
            bx2, by2 = x1 + tw + 8, max(y1 - 4, 18)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(img, label, (bx1 + 4, by2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Corner-marker lines for "locked on" visual
            d = 12
            for (cx, cy), (dx, dy) in [
                ((x1, y1), (d, d)), ((x2, y1), (-d, d)),
                ((x1, y2), (d, -d)), ((x2, y2), (-d, -d))
            ]:
                cv2.line(img, (cx, cy), (cx + dx, cy), color, thickness + 1)
                cv2.line(img, (cx, cy), (cx, cy + dy), color, thickness + 1)

    def _draw_hud(self, img, m):
        """
        Full HUD renderer — four zones:
          A. Top status bar (status text + FPS)
          B. Left telemetry panel (SRI value + four metric bars)
          C. Pedestrian count badge (new)
          D. SRI sparkline trend chart
        """
        H, W  = img.shape[:2]
        color = self._sri_color(m['sri'])

        # ── A. Top bar ────────────────────────────────────────────────────
        cv2.rectangle(img, (0, 0), (W, 54), (18, 18, 18), -1)
        badge  = " CRITICAL — HANDOVER CONTROL " if m['is_critical'] else " OPERATIONAL — SAFE TO DRIVE "
        prefix = "[!]" if m['is_critical'] else "[OK]"
        cv2.putText(img, f"{prefix}{badge}",
                    (16, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.82, color, 2, cv2.LINE_AA)
        fps_str = f"FPS {m['fps']:.1f}  |  Frame {m['frame']:04d}"
        tw = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(img, fps_str,
                    (W - tw - 14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (150, 150, 150), 1, cv2.LINE_AA)

        # ── B. Left telemetry panel ───────────────────────────────────────
        px, py, pw, ph = 10, 62, 300, 285
        overlay = img.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (70, 70, 70), 1)

        cv2.putText(img, "SAFETY TELEMETRY",
                    (px + 10, py + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(img, (px + 8, py + 30), (px + pw - 8, py + 30),
                 (55, 55, 55), 1)

        # Large SRI value
        cv2.putText(img, f"{m['sri']:.3f}",
                    (px + 12, py + 90), cv2.FONT_HERSHEY_DUPLEX,
                    2.0, color, 2, cv2.LINE_AA)
        cv2.putText(img, "System Risk Index",
                    (px + 12, py + 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (110, 110, 110), 1, cv2.LINE_AA)

        # Four metric bars (including Proximity)
        bx, bw, bh = px + 12, pw - 24, 11
        self._draw_metric_bar(img, bx, py + 135, bw, bh,
                               m['vis'],   (30, 200, 255),  "Visibility hazard    ")
        self._draw_metric_bar(img, bx, py + 168, bw, bh,
                               m['epist'], (100, 80, 255),  "Epistemic uncert.    ")
        self._draw_metric_bar(img, bx, py + 201, bw, bh,
                               m['temp'],  (50, 220, 150),  "Temporal instability ")
        self._draw_metric_bar(img, bx, py + 234, bw, bh,
                               m['prox'],  (0, 140, 255),   "Proximity alert  NEW ")

        # Stats row
        cv2.putText(img, f"Obj: {m['objs']:2d}   Ped: {m['peds']:2d}   Conf: {m['mean_conf']:.2f}",
                    (bx, py + 272), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (180, 180, 180), 1, cv2.LINE_AA)

        # ── C. Pedestrian proximity badge (visible when peds detected) ────
        if m['peds'] > 0:
            ped_color = (0, 60, 255) if m['prox'] > self.safety_threshold else (0, 140, 255)
            badge_text = f"PED x{m['peds']}  PROX {m['prox']:.2f}"
            tw = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            bx1, by1 = W - tw - 30, 62
            bx2, by2 = W - 10, 90
            cv2.rectangle(img, (bx1, by1), (bx2, by2), ped_color, -1)
            cv2.putText(img, badge_text, (bx1 + 8, by1 + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── D. SRI sparkline ─────────────────────────────────────────────
        spark_y = py + ph + 8
        if spark_y + 70 < H - 5:
            self._draw_sparkline(img, px, spark_y, pw, 70)

        return img

    # ─── Main per-frame pipeline ──────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Full pipeline (2 model calls, not 3):
          1. Dual inference: original + gamma-perturbed
          2. Pillar 1: visibility on raw frame
          3. Pillar 2: epistemic from extracted confidence arrays
          4. Pillar 3: temporal from SRI ring buffer
          5. Pillar 4: proximity from original detection boxes (NEW)
          6. Compute SRI with hard override
          7. Annotate frame: YOLO boxes + proximity highlights + HUD
        """
        self.frame_count += 1

        # ── Dual inference ────────────────────────────────────────────────
        perturbed = cv2.convertScaleAbs(frame, alpha=0.8, beta=-10)
        r_orig    = self.model(frame,     verbose=False)[0]
        r_pert    = self.model(perturbed, verbose=False)[0]

        conf_orig = (r_orig.boxes.conf.cpu().numpy()
                     if len(r_orig.boxes) > 0 else np.array([]))
        conf_pert = (r_pert.boxes.conf.cpu().numpy()
                     if len(r_pert.boxes) > 0 else np.array([]))

        # ── Four pillars ──────────────────────────────────────────────────
        vis       = self._pillar_visibility(frame)
        epist     = self._pillar_epistemic(conf_orig, conf_pert)
        temp      = self._pillar_temporal()
        prox, danger_boxes = self._pillar_proximity(frame, r_orig)

        mean_conf = float(np.mean(conf_orig)) if len(conf_orig) > 0 else 0.0
        ped_count = len(danger_boxes)

        # ── SRI ───────────────────────────────────────────────────────────
        sri = self._compute_sri(vis, epist, temp, mean_conf, prox)
        self.sri_history.append(sri)

        # ── FPS ───────────────────────────────────────────────────────────
        self.fps_buffer.append(time.time())
        fps = ((len(self.fps_buffer) - 1)
               / max(self.fps_buffer[-1] - self.fps_buffer[0], 1e-6)
               if len(self.fps_buffer) > 1 else 0.0)

        # ── Annotate ──────────────────────────────────────────────────────
        annotated = r_orig.plot()                         # standard YOLO boxes
        self._draw_proximity_highlights(annotated, danger_boxes)   # danger overlays

        metrics = dict(
            sri       = sri,
            vis       = vis,
            epist     = epist,
            temp      = temp,
            prox      = prox,
            mean_conf = mean_conf,
            is_critical = sri > self.safety_threshold,
            objs      = len(conf_orig),
            peds      = len([b for b in danger_boxes]),
            fps       = fps,
            frame     = self.frame_count,
        )
        annotated = self._draw_hud(annotated, metrics)
        return annotated, metrics

    # ─── Video runner ─────────────────────────────────────────────────────
    def run(self, source) -> None:
        """Opens a video/camera source and processes it in real-time."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {source}")

        cv2.namedWindow("Safety Supervisory System v2", cv2.WINDOW_NORMAL)
        print("[*] Running — press 'q' to quit.\n")
        print(f"  {'Frame':>5}  {'SRI':>6}  {'Vis':>6}  {'Unc':>6}"
              f"  {'Tmp':>6}  {'Prox':>6}  {'Ped':>3}  {'FPS':>6}  Status")
        print("  " + "─" * 68)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\n[*] Stream ended.")
                break

            annotated, m = self.process_frame(frame)
            cv2.imshow("Safety Supervisory System v2", annotated)

            flag = "CRITICAL" if m['is_critical'] else "  safe  "
            print(f"  {m['frame']:>5}  {m['sri']:>6.3f}  {m['vis']:>6.3f}"
                  f"  {m['epist']:>6.3f}  {m['temp']:>6.3f}  {m['prox']:>6.3f}"
                  f"  {m['peds']:>3}  {m['fps']:>6.1f}  [{flag}]",
                  end="\r", flush=True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[-] Aborted by user.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[*] Done — {self.frame_count} frames processed.")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    WEIGHTS   = "yolov8n.pt"
    INPUT_SRC = "data/videos/Pedestrian near miss DashCam video, Chester, UK - Olivia Barnes (1080p, h264).mp4"

    system = OptimizedSafetySystem(model_path=WEIGHTS, safety_threshold=0.45)
    system.run(INPUT_SRC)