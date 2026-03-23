"""
AI Traffic Flow Optimizer - FastAPI Backend (v2)
=================================================
- Starts SUMO TraCI simulation in background (auto-looping)
- Starts OpenCV/YOLOv8-style CV detector in background
- Streams annotated video frames as MJPEG
- Exposes REST endpoints for dashboard polling
"""

import os
import sys
import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from controller.simulation import state, state_lock, run_simulation
from cv_module.detector import (
    detector_state, detector_lock,
    start_detector_thread,
)

# ── History ring buffer ────────────────────────────────────────────────────
HISTORY_LEN = 120
history = []    # list of snapshots, max HISTORY_LEN
history_lock = threading.Lock()

def _history_sampler():
    """Samples state every 0.5s and appends to history ring buffer."""
    while True:
        time.sleep(0.5)
        with state_lock:
            snap = {
                "sim_time": state["sim_time"],
                "North": state["lanes"]["North"]["count"],
                "South": state["lanes"]["South"]["count"],
                "East":  state["lanes"]["East"]["count"],
                "West":  state["lanes"]["West"]["count"],
                "avg_wait": state["avg_wait_time"],
                "ambulance": state["ambulance_detected"],
            }
        with history_lock:
            history.append(snap)
            if len(history) > HISTORY_LEN:
                history.pop(0)

def _simulation_loop():
    """Runs SUMO simulation, auto-restarts when it ends."""
    while True:
        try:
            run_simulation(gui=False)
        except Exception as e:
            print(f"[Loop] Simulation error: {e}")
        print("[Loop] Simulation ended — restarting in 2s...")
        time.sleep(2)

# ── App Lifespan ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Backend] Starting SUMO simulation loop...")
    threading.Thread(target=_simulation_loop, daemon=True).start()
    time.sleep(2)   # let SUMO connect

    print("[Backend] Starting CV detector...")
    start_detector_thread(state, state_lock)

    print("[Backend] Starting history sampler...")
    threading.Thread(target=_history_sampler, daemon=True).start()

    yield
    print("[Backend] Shutdown.")

app = FastAPI(
    title="AI Traffic Flow Optimizer API",
    description="Real-time SUMO + YOLOv8-style CV pipeline for 4-way intersection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "version": "2.0.0", "project": "AI Traffic Flow Optimizer"}


@app.get("/traffic-data", tags=["Simulation"])
def get_traffic_data():
    """Live lane counts, signal states, wait times, alerts from SUMO."""
    with state_lock:
        return JSONResponse({
            "sim_time":         state["sim_time"],
            "step":             state["step"],
            "running":          state["running"],
            "mode":             state["mode"],
            "lanes":            state["lanes"],
            "junctions":        state["junctions"],
            "total_vehicles":   state["total_vehicles"],
            "vehicles_cleared": state["vehicles_cleared"],
            "avg_wait_time":    state["avg_wait_time"],
            "time_saved_pct":   state["time_saved_pct"],
            "alerts":           state["alerts"],
            "ambulance_detected": state["ambulance_detected"],
            "ambulance_id":      state["ambulance_id"],
            "active_ambulances":  state.get("active_ambulances", []),
        })


@app.get("/cv-data", tags=["CV"])
def get_cv_data():
    """YOLOv8-style detection summary: per-lane counts + vehicle type breakdown."""
    with detector_lock:
        return JSONResponse({
            "counts":             detector_state["counts"],
            "ambulance_detected": detector_state["ambulance_detected"],
            "vehicle_types":      detector_state["vehicle_types"],
            "source":             "SUMO+OpenCV synthetic pipeline",
        })


@app.get("/video-feed", tags=["CV"])
def video_feed():
    """MJPEG stream of annotated top-down intersection view (YOLOv8 bounding boxes)."""
    def generate():
        while True:
            with detector_lock:
                frame = detector_state["frame_jpeg"]
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
            time.sleep(0.05)   # ~20 FPS

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video-feed-junction", tags=["CV"])
def video_feed_junction(jid: str = "C1"):
    """MJPEG stream zoomed into a single junction."""
    def generate():
        while True:
            with detector_lock:
                jframes = detector_state.get("junction_frames", {})
                frame = jframes.get(jid)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
            time.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/optimize-signal", tags=["AI"])
def get_optimized_signal():
    """Density-based optimal signal timing recommendations."""
    with state_lock:
        lanes = state["lanes"]
        counts = {ap: lanes[ap]["count"] for ap in lanes}
        total = sum(counts.values()) or 1
        recs = {}
        for ap, count in counts.items():
            ratio = count / total
            recommended = max(15, min(60, int(15 + ratio * 90)))
            recs[ap] = {
                "vehicles":               count,
                "recommended_green_time": recommended,
                "current_green_time":     lanes[ap]["green_time"],
                "signal":                 lanes[ap]["signal"],
                "priority": "HIGH" if count > 8 else "MEDIUM" if count > 4 else "LOW",
            }
        return JSONResponse({
            "sim_time":         state["sim_time"],
            "algorithm":        "Density-Proportional Adaptive",
            "ambulance_override": state["ambulance_detected"],
            "recommendations":  recs,
        })


@app.get("/emergency", tags=["Emergency"])
def get_emergency():
    """Ambulance detection status + green corridor info."""
    with state_lock:
        amb = state["ambulance_detected"]
        return JSONResponse({
            "sim_time":            state["sim_time"],
            "ambulance_detected":  amb,
            "ambulance_id":        state["ambulance_id"],
            "route":               state["ambulance_route"],
            "green_corridor_active": amb,
            "message": (
                f"🚑 Green corridor active for {state['ambulance_id']}"
                if amb else "✅ No emergency vehicles"
            ),
        })


@app.get("/history", tags=["Analytics"])
def get_history():
    """Last 120 simulation snapshots for charting (0.5s intervals = 60s of data)."""
    with history_lock:
        return JSONResponse({"history": list(history)})


@app.get("/stats", tags=["Analytics"])
def get_stats():
    """Cumulative performance metrics."""
    with state_lock:
        amb_events = 0
        if state["sim_time"] > 80:  amb_events = 2
        elif state["sim_time"] > 50: amb_events = 1
        return JSONResponse({
            "sim_time":           state["sim_time"],
            "total_vehicles":     state["total_vehicles"],
            "vehicles_cleared":   state["vehicles_cleared"],
            "avg_wait_time":      state["avg_wait_time"],
            "time_saved_pct":     state["time_saved_pct"],
            "ambulance_events":   amb_events,
            "running":            state["running"],
        })


@app.post("/simulation/pause", tags=["Control"])
def pause_simulation():
    """Pause the simulation (vehicles freeze, signals hold)."""
    with state_lock:
        state["paused"] = True
    return JSONResponse({"status": "paused", "message": "Simulation paused"})


@app.post("/simulation/resume", tags=["Control"])
def resume_simulation():
    """Resume the simulation from where it was paused."""
    with state_lock:
        state["paused"] = False
    return JSONResponse({"status": "running", "message": "Simulation resumed"})


@app.post("/cv/zoom-in", tags=["CV"])
def zoom_in():
    return JSONResponse({"zoom": 1.0, "message": "Zoom not applicable in network view"})


@app.post("/cv/zoom-out", tags=["CV"])
def zoom_out():
    return JSONResponse({"zoom": 1.0, "message": "Zoom not applicable in network view"})


@app.post("/cv/zoom-reset", tags=["CV"])
def zoom_reset():
    return JSONResponse({"zoom": 1.0})


@app.get("/cv/zoom", tags=["CV"])
def get_zoom_level():
    return JSONResponse({"zoom": 1.0})


@app.get("/simulation/status", tags=["Control"])
def simulation_status():
    """Returns current paused/running state."""
    with state_lock:
        return JSONResponse({
            "paused":   state["paused"],
            "running":  state["running"],
            "sim_time": state["sim_time"],
        })

