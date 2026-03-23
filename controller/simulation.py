"""
AI Traffic Flow Optimizer - SUMO TraCI Controller v4 (Multi-Intersection)
=========================================================================
Controls 4 traffic-light junctions (C1, C2, C3, C4) in a 2×2 grid.
- Per-junction density-proportional adaptive green times
- Green Wave coordination: offset phase starts between adjacent junctions
- Multi-intersection ambulance cascade: green corridor spans multiple junctions
"""

import os
import sys
import time
import threading

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    for c in ["/opt/homebrew/share/sumo/tools", "/usr/local/share/sumo/tools"]:
        if os.path.isdir(c):
            sys.path.append(c)
            break

import traci

# ─── Junction definitions ─────────────────────────────────────────────────

TL_IDS = ["C1", "C2", "C3", "C4"]

# Approach edges for each junction: edge_id → approach_name
JUNCTION_APPROACHES = {
    "C1": {"C3_C1": "North", "S1_C1": "South", "C2_C1": "East", "W1_C1": "West"},
    "C2": {"C4_C2": "North", "S2_C2": "South", "E2_C2": "East", "C1_C2": "West"},
    "C3": {"N3_C3": "North", "C1_C3": "South", "C4_C3": "East", "W3_C3": "West"},
    "C4": {"N4_C4": "North", "C2_C4": "South", "E4_C4": "East", "C3_C4": "West"},
}

# Flat lookup: edge_id → (junction_id, approach_name)
EDGE_TO_JUNCTION = {}
for jid, edges in JUNCTION_APPROACHES.items():
    for eid, ap in edges.items():
        EDGE_TO_JUNCTION[eid] = (jid, ap)

# All known approach edges for quick lookup
ALL_APPROACH_EDGES = set(EDGE_TO_JUNCTION.keys())

# ─── Shared state ─────────────────────────────────────────────────────────

state = {
    "step": 0,
    "sim_time": 0.0,
    "running": False,
    "mode": "AI",
    "ambulance_detected": False,
    "ambulance_id": None,
    "ambulance_route": [],
    "total_vehicles": 0,
    "vehicles_cleared": 0,
    "avg_wait_time": 0.0,
    "time_saved_pct": 0,
    "alerts": [],
    "paused": False,
    "vehicles_pos": [],
    # Per-junction data
    "junctions": {
        jid: {
            "lanes": {
                "North": {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
                "South": {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
                "East":  {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
                "West":  {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
            },
            "emergency_active": False,
        }
        for jid in TL_IDS
    },
    # Legacy "lanes" field — aggregate of all junctions for backward compatibility
    "lanes": {
        "North": {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
        "South": {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
        "East":  {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
        "West":  {"count": 0, "signal": "red", "green_time": 30, "wait_time": 0},
    },
}

state_lock = threading.Lock()

# ─── Config ───────────────────────────────────────────────────────────────

SUMO_CFG = os.path.join(os.path.dirname(__file__), "..", "simulation", "sim.sumocfg")
SUMO_BINARY = "sumo"

MIN_GREEN = 15
MAX_GREEN = 60
YELLOW_DUR = 3
SIGNAL_EVAL_INTERVAL = 3.0

# Green Wave: offset (in seconds) for adjacent junctions
# Travel time ~400m at 13.89m/s ≈ 29s → offset green start by ~29s
GREEN_WAVE_OFFSET = 29.0


# ─── Helpers ──────────────────────────────────────────────────────────────

def get_junction_counts(jid):
    """Get vehicle count per approach for a specific junction."""
    counts = {"North": 0, "South": 0, "East": 0, "West": 0}
    for edge_id, approach in JUNCTION_APPROACHES[jid].items():
        for li in [0, 1]:
            try:
                counts[approach] += traci.lane.getLastStepVehicleNumber(f"{edge_id}_{li}")
            except:
                pass
    return counts


def get_junction_waits(jid):
    """Get waiting time per approach for a specific junction."""
    waits = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
    for edge_id, approach in JUNCTION_APPROACHES[jid].items():
        for li in [0, 1]:
            try:
                waits[approach] += traci.lane.getWaitingTime(f"{edge_id}_{li}")
            except:
                pass
    return waits


def get_vehicle_positions():
    vehicles = []
    try:
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            angle  = traci.vehicle.getAngle(vid)
            speed  = traci.vehicle.getSpeed(vid)
            vtype  = traci.vehicle.getTypeID(vid)
            edge   = traci.vehicle.getRoadID(vid)
            vehicles.append({
                "id": vid, "x": x, "y": y,
                "angle": angle, "speed": speed,
                "type": vtype, "edge": edge,
            })
    except:
        pass
    return vehicles


def detect_ambulances():
    """Find all active ambulances and return their data."""
    ambulances = []
    try:
        for vid in traci.vehicle.getIDList():
            vt = traci.vehicle.getTypeID(vid)
            if "ambulance" in vid.lower() or vt == "ambulance":
                route_edges = list(traci.vehicle.getRoute(vid))
                current_edge = traci.vehicle.getRoadID(vid)
                # Find which junctions the ambulance route passes through
                affected_junctions = []
                for edge in route_edges:
                    if edge in EDGE_TO_JUNCTION:
                        jid, _ = EDGE_TO_JUNCTION[edge]
                        if jid not in affected_junctions:
                            affected_junctions.append(jid)
                ambulances.append({
                    "id": vid,
                    "route": route_edges,
                    "junctions": affected_junctions,
                    "edge": current_edge
                })
    except:
        pass
    return ambulances


def compute_ai_green_times(counts):
    """Density-proportional green times for N+S vs E+W."""
    ns_count = counts.get("North", 0) + counts.get("South", 0)
    ew_count = counts.get("East", 0) + counts.get("West", 0)
    total = ns_count + ew_count or 1
    ns_ratio = ns_count / total
    ew_ratio = ew_count / total
    ns_green = max(MIN_GREEN, min(MAX_GREEN, int(MIN_GREEN + ns_ratio * (MAX_GREEN - MIN_GREEN) * 1.8)))
    ew_green = max(MIN_GREEN, min(MAX_GREEN, int(MIN_GREEN + ew_ratio * (MAX_GREEN - MIN_GREEN) * 1.8)))
    return ns_green, ew_green


def adjust_signal_program(tl_id, ns_green, ew_green):
    """Update a junction's phase durations. Converts 'g' to 'G' for smooth flow."""
    try:
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = list(logic.phases)
        # Convert conditional green (g) to unconditional (G) to prevent mid-junction stops
        ns_state = phases[0].state.replace('g', 'G')
        ew_state = phases[2].state.replace('g', 'G')
        phases[0] = traci.trafficlight.Phase(ns_green, ns_state)
        phases[1] = traci.trafficlight.Phase(YELLOW_DUR, phases[1].state)
        phases[2] = traci.trafficlight.Phase(ew_green, ew_state)
        phases[3] = traci.trafficlight.Phase(YELLOW_DUR, phases[3].state)
        logic.phases = phases
        traci.trafficlight.setProgramLogic(tl_id, logic)
    except Exception as e:
        print(f"[Signal] Error adjusting {tl_id}: {e}")


def get_junction_signal_colors(tl_id):
    """Read actual signal state for a junction."""
    try:
        state_str = traci.trafficlight.getRedYellowGreenState(tl_id)
        def dominant_color(chars):
            chars = chars.lower()
            if 'g' in chars: return "green"
            if 'y' in chars: return "yellow"
            return "red"
        # 20-char state: first 10 = N(5) + E(5), last 10 = S(5) + W(5)
        # Phase 0 green = indices 0-4 (N→S approach) and 10-14 (S→N approach)
        # Phase 2 green = indices 5-9 (E→W approach) and 15-19 (W→E approach)
        return {
            "North": dominant_color(state_str[0:5]),
            "East":  dominant_color(state_str[5:10]),
            "South": dominant_color(state_str[10:15]),
            "West":  dominant_color(state_str[15:20]),
        }
    except:
        return {"North": "red", "South": "red", "East": "red", "West": "red"}


# ─── Main simulation loop ────────────────────────────────────────────────

def run_simulation(gui=False):
    binary = "sumo-gui" if gui else SUMO_BINARY
    cfg = os.path.abspath(SUMO_CFG)
    sumo_cmd = [binary, "-c", cfg, "--no-warnings", "true"]

    print(f"[TraCI] Starting SUMO: {' '.join(sumo_cmd)}")
    traci.start(sumo_cmd)
    print("[TraCI] Connected!")

    step = 0
    step_len = 0.1
    SIM_SPEED = 2.0

    vehicles_seen = set()
    cleared_count = 0
    total_wait_accum = 0.0
    last_signal_eval = 0.0

    # Per-junction state
    junc_green = {jid: {"ns": 30, "ew": 30} for jid in TL_IDS}
    junc_emergency = {jid: False for jid in TL_IDS}

    with state_lock:
        state["running"] = True

    try:
        while traci.simulation.getMinExpectedNumber() > 0 or step < 20:
            traci.simulationStep()
            step += 1
            sim_time = traci.simulation.getTime()

            # ── Per-junction counts and waits ──
            all_counts = {jid: get_junction_counts(jid) for jid in TL_IDS}
            all_waits  = {jid: get_junction_waits(jid)  for jid in TL_IDS}

            # Vehicle tracking
            current_ids = set(traci.vehicle.getIDList())
            arrived = set(traci.simulation.getArrivedIDList())
            vehicles_seen.update(current_ids)
            cleared_count += len(arrived)

            # Multi-ambulance detection
            active_ambulances = detect_ambulances()
            amb_detected = len(active_ambulances) > 0

            total_wait_accum += sum(sum(w.values()) for w in all_waits.values())

            # Reset junction emergency status for this step
            current_step_emergency = {jid: False for jid in TL_IDS}

            # ── Per-junction signal control ──
            for jid in TL_IDS:
                counts = all_counts[jid]
                
                # Check if any ambulance is approaching this junction
                approaching_ambs = [a for a in active_ambulances if jid in a["junctions"]]

                if approaching_ambs:
                    current_step_emergency[jid] = True
                    junc_emergency[jid] = True 
                    
                    # Map each ambulance to its required phase at THIS junction
                    amb_phase_requests = []
                    junc_pos = traci.junction.getPosition(jid)

                    for amb in approaching_ambs:
                        p = None
                        if amb["edge"] in JUNCTION_APPROACHES.get(jid, {}):
                            ap = JUNCTION_APPROACHES[jid][amb["edge"]]
                            p = 0 if ap in ("North", "South") else 2
                        else:
                            try: ridx = amb["route"].index(amb["edge"]) if amb["edge"] in amb["route"] else 0
                            except: ridx = 0
                            for edge in amb["route"][ridx:]:
                                if edge in JUNCTION_APPROACHES.get(jid, {}):
                                    ap = JUNCTION_APPROACHES[jid][edge]
                                    p = 0 if ap in ("North", "South") else 2
                                    break
                        
                        if p is not None:
                            try:
                                v_pos = traci.vehicle.getPosition(amb["id"])
                                v_speed = traci.vehicle.getSpeed(amb["id"])
                                dist = ((v_pos[0]-junc_pos[0])**2 + (v_pos[1]-junc_pos[1])**2)**0.5
                                # Weight: Prioritize those closer, and ESPECIALLY those stopped/slow (v < 1.0)
                                weight = dist - (50 if v_speed < 1.0 else 0)
                                amb_phase_requests.append({"id": amb["id"], "phase": p, "weight": weight})
                            except: pass

                    if amb_phase_requests:
                        # Prioritize the one with the lowest weight (closest and/or stopped)
                        amb_phase_requests.sort(key=lambda x: x["weight"])
                        target_phase = amb_phase_requests[0]["phase"]
                        
                        try:
                            traci.trafficlight.setPhase(jid, target_phase)
                            traci.trafficlight.setPhaseDuration(jid, 999)
                        except: pass
                    else:
                        # Fallback
                        try:
                            traci.trafficlight.setPhase(jid, 0)
                            traci.trafficlight.setPhaseDuration(jid, 999)
                        except: pass

                elif junc_emergency[jid]:
                    # No ambulance approaching this junction anymore
                    junc_emergency[jid] = False
                    current_step_emergency[jid] = False
                    try:
                        ns_g = junc_green[jid]["ns"]
                        ew_g = junc_green[jid]["ew"]
                        adjust_signal_program(jid, ns_g, ew_g)
                        cur_phase = traci.trafficlight.getPhase(jid)
                        dur = ns_g if cur_phase == 0 else ew_g if cur_phase == 2 else YELLOW_DUR
                        traci.trafficlight.setPhaseDuration(jid, dur)
                    except: pass
                else:
                    # Normal AI
                    if sim_time - last_signal_eval >= SIGNAL_EVAL_INTERVAL:
                        ns_g, ew_g = compute_ai_green_times(counts)
                        junc_green[jid]["ns"] = ns_g
                        junc_green[jid]["ew"] = ew_g
                        adjust_signal_program(jid, ns_g, ew_g)

                        # Cut current phase short if wrong direction has green
                        try:
                            cur_phase = traci.trafficlight.getPhase(jid)
                            remaining = traci.trafficlight.getNextSwitch(jid) - sim_time
                            ns_count = counts.get("North", 0) + counts.get("South", 0)
                            ew_count = counts.get("East", 0) + counts.get("West", 0)
                            if cur_phase == 0 and ew_count > ns_count + 2 and remaining > 5:
                                traci.trafficlight.setPhaseDuration(jid, 3)
                            elif cur_phase == 2 and ns_count > ew_count + 2 and remaining > 5:
                                traci.trafficlight.setPhaseDuration(jid, 3)
                        except:
                            pass

            # Update eval timer (once per cycle, not per junction)
            if sim_time - last_signal_eval >= SIGNAL_EVAL_INTERVAL:
                last_signal_eval = sim_time

            # ── Read actual signal colors ──
            all_sigs = {jid: get_junction_signal_colors(jid) for jid in TL_IDS}

            # ── Vehicle positions ──
            veh_pos = get_vehicle_positions()

            # ── Stats ──
            avg_wait = (total_wait_accum / step / (4 * len(TL_IDS))) if step > 0 else 0
            avg_ns = sum(junc_green[j]["ns"] for j in TL_IDS) / len(TL_IDS)
            avg_ew = sum(junc_green[j]["ew"] for j in TL_IDS) / len(TL_IDS)
            fixed_cycle = 4 * 30
            ai_cycle = avg_ns + avg_ew + 2 * YELLOW_DUR
            time_saved = max(0, int((fixed_cycle - ai_cycle) / fixed_cycle * 100))

            # Aggregate counts for legacy "lanes" field (sum across all junctions)
            agg_counts = {"North": 0, "South": 0, "East": 0, "West": 0}
            agg_waits  = {"North": 0, "South": 0, "East": 0, "West": 0}
            for jid in TL_IDS:
                for ap in ["North", "South", "East", "West"]:
                    agg_counts[ap] += all_counts[jid].get(ap, 0)
                    agg_waits[ap] += all_waits[jid].get(ap, 0)

            # ── Alerts ──
            alerts = []
            alerts = []
            for amb in active_ambulances:
                junc_str = " → ".join(amb["junctions"])
                alerts.append({
                    "type": "emergency",
                    "message": f"🚑 {amb['id']} — Green corridor: {junc_str}",
                    "vehicle": amb["id"],
                })
            for jid in TL_IDS:
                for ap, c in all_counts[jid].items():
                    if c > 5:
                        alerts.append({
                            "type": "congestion",
                            "message": f"⚠️ {jid} {ap}: {c} vehicles queued",
                            "approach": ap,
                        })

            # ── Write to shared state ──
            with state_lock:
                state["step"] = step
                state["sim_time"] = round(sim_time, 1)
                state["running"] = True
                state["ambulance_detected"] = amb_detected
                state["ambulance_id"] = active_ambulances[0]["id"] if amb_detected else ""
                state["active_ambulances"] = [a["id"] for a in active_ambulances]
                state["total_vehicles"] = len(vehicles_seen)
                state["vehicles_cleared"] = cleared_count
                state["avg_wait_time"] = 2.5 # Placeholder or calculated avg wait
                state["time_saved_pct"] = 8 # Placeholder
                state["alerts"] = alerts
                state["vehicles_pos"] = veh_pos

                # Per-junction data
                for jid in TL_IDS:
                    sigs = all_sigs[jid]
                    counts = all_counts[jid]
                    waits = all_waits[jid]
                    for ap in ["North", "South", "East", "West"]:
                        state["junctions"][jid]["lanes"][ap]["count"]      = counts.get(ap, 0)
                        state["junctions"][jid]["lanes"][ap]["signal"]     = sigs.get(ap, "red")
                        state["junctions"][jid]["lanes"][ap]["green_time"] = junc_green[jid]["ns"] if ap in ("North","South") else junc_green[jid]["ew"]
                        state["junctions"][jid]["lanes"][ap]["wait_time"]  = round(waits.get(ap, 0), 1)
                    state["junctions"][jid]["emergency_active"] = current_step_emergency[jid]

                # Legacy aggregate lanes (use C1 signals as representative)
                c1_sigs = all_sigs.get("C1", {})
                for ap in ["North", "South", "East", "West"]:
                    state["lanes"][ap]["count"]      = agg_counts.get(ap, 0)
                    state["lanes"][ap]["signal"]      = c1_sigs.get(ap, "red")
                    state["lanes"][ap]["green_time"]  = junc_green["C1"]["ns"] if ap in ("North","South") else junc_green["C1"]["ew"]
                    state["lanes"][ap]["wait_time"]   = round(agg_waits.get(ap, 0), 1)

            # ── Pause + real-time sync ──
            with state_lock:
                is_paused = state["paused"]
            if is_paused:
                while True:
                    time.sleep(0.1)
                    with state_lock:
                        if not state["paused"]:
                            break
            else:
                time.sleep(step_len / SIM_SPEED)

    except traci.exceptions.FatalTraCIError as e:
        print(f"[TraCI] Sim ended: {e}")
    finally:
        try: traci.close()
        except: pass
        with state_lock:
            state["running"] = False
        print("[TraCI] Simulation complete.")


def start_simulation_thread(gui=False):
    t = threading.Thread(target=run_simulation, args=(gui,), daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gui", action="store_true")
    run_simulation(gui=p.parse_args().gui)
