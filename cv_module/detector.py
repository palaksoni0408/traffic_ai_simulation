"""
CV Module — Multi-Intersection Renderer (v2 — corrected coordinates)
=====================================================================
Renders a bird's-eye view of a 2×2 grid of intersections (C1, C2, C3, C4).

SUMO actual world coordinates (after netconvert offset):
  C3(300,700) ── C4(700,700)     ← top row
    │              │
  C1(300,300) ── C2(700,300)     ← bottom row

  Boundaries: S=0, N=1000, W=0, E=1000
  Junction box: ~289 to ~310 (width ~20m)
"""

import cv2
import numpy as np
import math
import time
import threading
import random

# ── Canvas ──
W, H = 800, 800

# ── SUMO world bounds (actual from net.xml) ──
WORLD_MIN_X = -20
WORLD_MAX_X = 1020
WORLD_MIN_Y = -20
WORLD_MAX_Y = 1020
WORLD_W = WORLD_MAX_X - WORLD_MIN_X  # 1040
WORLD_H = WORLD_MAX_Y - WORLD_MIN_Y  # 1040

# Junction centres (SUMO world coords)
JUNCTIONS = {
    "C1": (300, 300),
    "C2": (700, 300),
    "C3": (300, 700),
    "C4": (700, 700),
}

def world_to_px(wx, wy):
    """Convert SUMO world (x, y) to pixel (px, py)."""
    px = int(((wx - WORLD_MIN_X) / WORLD_W) * W)
    py = int(((WORLD_MAX_Y - wy) / WORLD_H) * H)   # SUMO Y up, pixel Y down
    return px, py

# Junction pixel centres
JUNCTION_PX = {jid: world_to_px(x, y) for jid, (x, y) in JUNCTIONS.items()}

# Road half-width in pixels
ROAD_HW = 10

# Junction box half-size (aligned with 10m stop line offset)
JBOX = 10

# ── Colors (BGR) ──
C_BG       = (30, 38, 48)
C_ROAD     = (50, 55, 62)
C_JUNCTION = (60, 68, 78)
C_LANE     = (50, 150, 200)
C_STOP     = (160, 160, 170)

# Vehicle colours per type (BGR)
VEH_COLORS = {
    "car":       [(190,185,220), (150,210,150), (210,170,130), (160,160,210), (200,200,170), (170,190,220)],
    "truck":     [(100,140,180), (120,165,145), (140,130,100)],
    "bike":      [(210,210,110), (190,190,90)],
    "ambulance": [(60, 60, 245)],
}

VEH_PX = {
    "car":       (5, 10),
    "truck":     (6, 15),
    "bike":      (3, 7),
    "ambulance": (6, 12),
}

_veh_meta = {}

def _get_veh_meta(vid, vtype):
    if vid not in _veh_meta:
        colors = VEH_COLORS.get(vtype, VEH_COLORS["car"])
        _veh_meta[vid] = {
            "color_idx": random.randint(0, len(colors) - 1),
            "conf": round(random.uniform(0.84, 0.99), 2),
            "siren": 0,
        }
    return _veh_meta[vid]


# ─── Detector shared state ────────────────────────────────────────────────
detector_state = {
    "frame_jpeg": None,
    "counts": {"North": 0, "South": 0, "East": 0, "West": 0},
    "ambulance_detected": False,
    "vehicle_types": {"car": 0, "truck": 0, "bike": 0, "ambulance": 0},
}
detector_lock = threading.Lock()


# ─── Background drawing ──────────────────────────────────────────────────

def _draw_base(frame):
    frame[:] = C_BG
    hw = ROAD_HW

    # ── Draw roads ──
    # Horizontal roads at y=300 and y=700 (full width)
    for wy in [300, 700]:
        _, py = world_to_px(0, wy)
        cv2.rectangle(frame, (0, py - hw), (W, py + hw), C_ROAD, -1)

    # Vertical roads at x=300 and x=700 (full height)
    for wx in [300, 700]:
        px, _ = world_to_px(wx, 0)
        cv2.rectangle(frame, (px - hw, 0), (px + hw, H), C_ROAD, -1)

    # ── Junction boxes ──
    for jid, (jx, jy) in JUNCTION_PX.items():
        cv2.rectangle(frame, (jx - JBOX, jy - JBOX), (jx + JBOX, jy + JBOX), C_JUNCTION, -1)
        # Junction border
        cv2.rectangle(frame, (jx - JBOX, jy - JBOX), (jx + JBOX, jy + JBOX), (80, 90, 105), 1)

    # ── Centre lane dashes ──
    for wy in [300, 700]:
        _, py = world_to_px(0, wy)
        for x in range(0, W, 14):
            cv2.line(frame, (x, py), (x + 8, py), C_LANE, 1)
    for wx in [300, 700]:
        px, _ = world_to_px(wx, 0)
        for y in range(0, H, 14):
            cv2.line(frame, (px, y), (px, y + 8), C_LANE, 1)

    # ── Stop lines at junction edges ──
    for jid, (jx, jy) in JUNCTION_PX.items():
        b = JBOX
        # North approach stop
        cv2.line(frame, (jx - hw, jy - b), (jx + hw, jy - b), C_STOP, 1)
        # South approach stop
        cv2.line(frame, (jx - hw, jy + b), (jx + hw, jy + b), C_STOP, 1)
        # West approach stop
        cv2.line(frame, (jx - b, jy - hw), (jx - b, jy + hw), C_STOP, 1)
        # East approach stop
        cv2.line(frame, (jx + b, jy - hw), (jx + b, jy + hw), C_STOP, 1)

    # ── Junction labels ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    for jid, (jx, jy) in JUNCTION_PX.items():
        cv2.putText(frame, jid, (jx - 8, jy - JBOX - 4), font, 0.4, (120, 140, 170), 1, cv2.LINE_AA)

    # ── Edge labels ──
    # South boundary
    s1x, s1y = world_to_px(300, 0)
    s2x, s2y = world_to_px(700, 0)
    cv2.putText(frame, "S1", (s1x - 6, s1y - 4), font, 0.3, (80,90,110), 1)
    cv2.putText(frame, "S2", (s2x - 6, s2y - 4), font, 0.3, (80,90,110), 1)
    # North boundary
    n3x, n3y = world_to_px(300, 1000)
    n4x, n4y = world_to_px(700, 1000)
    cv2.putText(frame, "N3", (n3x - 6, n3y + 12), font, 0.3, (80,90,110), 1)
    cv2.putText(frame, "N4", (n4x - 6, n4y + 12), font, 0.3, (80,90,110), 1)
    # West
    w1x, w1y = world_to_px(0, 300)
    w3x, w3y = world_to_px(0, 700)
    cv2.putText(frame, "W1", (w1x + 4, w1y + 4), font, 0.3, (80,90,110), 1)
    cv2.putText(frame, "W3", (w3x + 4, w3y + 4), font, 0.3, (80,90,110), 1)
    # East
    e2x, e2y = world_to_px(1000, 300)
    e4x, e4y = world_to_px(1000, 700)
    cv2.putText(frame, "E2", (e2x - 20, e2y + 4), font, 0.3, (80,90,110), 1)
    cv2.putText(frame, "E4", (e4x - 20, e4y + 4), font, 0.3, (80,90,110), 1)


def _draw_traffic_lights(frame, junctions_data):
    """Draw traffic light indicators at each junction."""
    sig_c = {"green": (50, 220, 80), "yellow": (40, 210, 230), "red": (50, 50, 230)}
    off = JBOX + 5

    for jid, (jx, jy) in JUNCTION_PX.items():
        jdata = junctions_data.get(jid, {})
        lanes = jdata.get("lanes", {})

        positions = {
            "North": (jx + off, jy - off),
            "South": (jx - off, jy + off),
            "East":  (jx + off, jy + off),
            "West":  (jx - off, jy - off),
        }

        for ap, (px, py) in positions.items():
            sig = lanes.get(ap, {}).get("signal", "red")
            color = sig_c.get(sig, sig_c["red"])
            cv2.circle(frame, (px, py), 3, color, -1)
            if sig == "green":
                cv2.circle(frame, (px, py), 5, (30, 140, 50), 1)


# ─── Vehicle drawing ─────────────────────────────────────────────────────

def _draw_vehicle(frame, px, py, angle_deg, vtype, meta):
    vw, vh = VEH_PX.get(vtype, (5, 10))
    colors = VEH_COLORS.get(vtype, VEH_COLORS["car"])
    bgr = colors[meta["color_idx"] % len(colors)]
    rot = -(angle_deg - 90)
    rect = ((px, py), (vw, vh), rot)
    box = cv2.boxPoints(rect).astype(np.intp)
    cv2.fillPoly(frame, [box], bgr)

    if vtype == "ambulance":
        cv2.line(frame, (px - 2, py), (px + 2, py), (255, 255, 255), 1)
        cv2.line(frame, (px, py - 2), (px, py + 2), (255, 255, 255), 1)
        meta["siren"] = (meta.get("siren", 0) + 1) % 16
        siren_c = (255, 50, 50) if meta["siren"] < 8 else (50, 50, 255)
        cv2.circle(frame, (px, py - vh // 3), 2, siren_c, -1)


def _draw_yolo_bbox(frame, px, py, angle_deg, vtype, meta):
    vw, vh = VEH_PX.get(vtype, (5, 10))
    rect = ((px, py), (vw + 4, vh + 4), -(angle_deg - 90))
    box = cv2.boxPoints(rect)
    x1 = max(0, int(box[:, 0].min()))
    y1 = max(0, int(box[:, 1].min()))
    x2 = min(W, int(box[:, 0].max()))
    y2 = min(H, int(box[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return
    bc = {"ambulance": (0, 0, 255), "truck": (0, 165, 255), "bike": (0, 255, 255)}.get(vtype, (0, 230, 80))
    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = f"{vtype.upper()} {meta['conf']:.2f}"
    tw, th = cv2.getTextSize(txt, font, 0.22, 1)[0]
    ly = y1 - 2 if y1 > 12 else y2 + 10
    cv2.rectangle(frame, (x1, ly - th - 1), (x1 + tw + 2, ly + 1), bc, -1)
    cv2.putText(frame, txt, (x1 + 1, ly), font, 0.22, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_hud(frame, sim_state, total_count, amb):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 36), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "YOLOv8  AI Traffic — 4-Junction Network", (8, 14), font, 0.38, (80, 210, 130), 1, cv2.LINE_AA)
    st = sim_state.get("sim_time", 0)
    cv2.putText(frame, f"T={st:.1f}s", (W - 70, 14), font, 0.35, (150, 180, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Vehicles: {total_count}", (8, 30), font, 0.3, (200, 185, 110), 1, cv2.LINE_AA)

    if amb:
        ov2 = frame.copy()
        cv2.rectangle(ov2, (0, H - 24), (W, H), (0, 0, 170), -1)
        cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "AMBULANCE — MULTI-JUNCTION GREEN CORRIDOR ACTIVE",
                    (15, H - 6), font, 0.34, (100, 100, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (W - 58, H - 22), (W - 3, H - 4), (20, 85, 45), -1)
    cv2.rectangle(frame, (W - 58, H - 22), (W - 3, H - 4), (50, 220, 90), 1)
    cv2.putText(frame, "AI NET", (W - 54, H - 9), font, 0.28, (50, 220, 90), 1, cv2.LINE_AA)


# ─── Main render loop ─────────────────────────────────────────────────────
# Pixel region around each junction for cropping (centre ± radius)
CROP_RADIUS = 155  # pixels from junction centre

def run_detector(sim_state_ref, sim_lock_ref):
    base_frame = np.zeros((H, W, 3), dtype=np.uint8)
    _draw_base(base_frame)

    while True:
        t0 = time.time()

        with sim_lock_ref:
            sim_time = sim_state_ref.get("sim_time", 0)
            amb      = sim_state_ref.get("ambulance_detected", False)
            junctions = dict(sim_state_ref.get("junctions", {}))
            veh_list = list(sim_state_ref.get("vehicles_pos", []))
            lanes    = dict(sim_state_ref.get("lanes", {}))

        frame = base_frame.copy()
        _draw_traffic_lights(frame, junctions)

        type_counts = {"car": 0, "truck": 0, "bike": 0, "ambulance": 0}
        total_drawn = 0
        for v in veh_list:
            wx, wy = v["x"], v["y"]
            px, py = world_to_px(wx, wy)
            if px < -10 or px > W + 10 or py < -10 or py > H + 10:
                continue

            vtype = v["type"] if v["type"] in VEH_PX else "car"
            meta = _get_veh_meta(v["id"], vtype)
            _draw_vehicle(frame, px, py, v["angle"], vtype, meta)
            _draw_yolo_bbox(frame, px, py, v["angle"], vtype, meta)
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
            total_drawn += 1

        counts = {ap: lanes.get(ap, {}).get("count", 0) for ap in ["North", "South", "East", "West"]}
        _draw_hud(frame, {"sim_time": sim_time, "ambulance_detected": amb}, total_drawn, amb)

        # Encode full network frame
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Generate per-junction cropped frames (zoomed in)
        junction_jpegs = {}
        for jid, (jx, jy) in JUNCTION_PX.items():
            x1 = max(0, jx - CROP_RADIUS)
            y1 = max(0, jy - CROP_RADIUS)
            x2 = min(W, jx + CROP_RADIUS)
            y2 = min(H, jy + CROP_RADIUS)
            crop = frame[y1:y2, x1:x2]
            # Scale up to 400x400 for clarity
            zoomed = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_LINEAR)
            # Add junction label overlay
            ov = zoomed.copy()
            cv2.rectangle(ov, (0, 0), (400, 28), (10, 15, 25), -1)
            cv2.addWeighted(ov, 0.8, zoomed, 0.2, 0, zoomed)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(zoomed, f"{jid} — Zoomed View", (8, 18), font, 0.45, (80, 210, 130), 1, cv2.LINE_AA)
            cv2.putText(zoomed, f"T={sim_time:.1f}s", (320, 18), font, 0.35, (150, 180, 220), 1, cv2.LINE_AA)
            # Emergency indicator
            jdata = junctions.get(jid, {})
            if jdata.get("emergency_active"):
                cv2.rectangle(zoomed, (0, 380), (400, 400), (0, 0, 180), -1)
                cv2.putText(zoomed, "EMERGENCY GREEN CORRIDOR", (60, 395), font, 0.4, (100, 100, 255), 1, cv2.LINE_AA)
            r2, j2 = cv2.imencode(".jpg", zoomed, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if r2:
                junction_jpegs[jid] = j2.tobytes()

        if ret:
            with detector_lock:
                detector_state["frame_jpeg"] = jpeg.tobytes()
                detector_state["counts"] = counts
                detector_state["ambulance_detected"] = amb
                detector_state["vehicle_types"] = type_counts
                detector_state["junction_frames"] = junction_jpegs

        active_ids = {v["id"] for v in veh_list}
        for old_id in list(_veh_meta.keys()):
            if old_id not in active_ids:
                del _veh_meta[old_id]

        time.sleep(max(0, (1 / 20) - (time.time() - t0)))


def start_detector_thread(sim_state_ref, sim_lock_ref):
    t = threading.Thread(target=run_detector, args=(sim_state_ref, sim_lock_ref), daemon=True)
    t.start()
    return t

