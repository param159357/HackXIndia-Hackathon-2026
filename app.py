
import streamlit as st
import cv2
import math
import time
import os
import numpy as np
import pandas as pd
import traceback
from ultralytics import YOLO
from collections import deque
import random
import base64
from datetime import datetime
import easyocr

from tracking_engine import TrackingEngine
from trajectory_engine import TrajectoryEngine
from emergency_engine import EmergencyEngine
from obstruction_engine import ObstructionEngine
from risk_engine import RiskEngine
st.set_page_config(page_title="Smart Traffic Violation Priority Engine", layout="wide", initial_sidebar_state="expanded")
with open('static/custom.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
audio_placeholder = st.empty()

def trigger_siren():
    siren_url = "https://www.soundjay.com/misc/sounds/police-siren-1.mp3"
    audio_placeholder.markdown(f"""
        <audio autoplay key="{time.time()}">
            <source src="{siren_url}" type="audio/mpeg">
        </audio>
    """, unsafe_allow_html=True)
def inject_live_clock():
    st.components.v1.html("""
        <div id="ist-clock" style="
            text-align: right;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(8px);
            padding: 10px 15px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-family: 'Inter', -apple-system, sans-serif;
            width: fit-content;
            margin-left: auto;
        ">
            <div id="clock-time" style="font-size: 1.1rem; font-weight: 700; color: #6366f1; font-family: 'JetBrains Mono', monospace; line-height: 1.2;">00:00:00</div>
            <div id="clock-date" style="font-size: 0.75rem; color: #94a3b8; font-weight: 500; margin-top: 2px;">Loading...</div>
        </div>
        <script>
            function updateClock() {
                const now = new Date();
                const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
                const ist = new Date(utc + (3600000 * 5.5));
                
                const hours = String(ist.getHours()).padStart(2, '0');
                const minutes = String(ist.getMinutes()).padStart(2, '0');
                const seconds = String(ist.getSeconds()).padStart(2, '0');
                const timeStr = hours + ':' + minutes + ':' + seconds;
                
                const dayName = ist.toLocaleDateString('en-IN', { weekday: 'long' });
                const dateStr = ist.toLocaleDateString('en-IN', { day: '2-digit', month: '2-digit', year: 'numeric' });
                
                document.getElementById('clock-time').innerText = timeStr;
                document.getElementById('clock-date').innerText = dayName + ', ' + dateStr;
            }
            setInterval(updateClock, 1000);
            updateClock();
        </script>
    """, height=75)
def run_self_repair():
    if 'repair_logs' not in st.session_state: st.session_state.repair_logs = []
    if 'engine_restarts' not in st.session_state: st.session_state.engine_restarts = 0
   
    cutoff_time = time.time() - 5.0
    stale_ids = []
    for tid, data in track_data.items():
        if data['history']:
            last_t = data['history'][-1][2]
            if last_t < cutoff_time: stale_ids.append(tid)
        else:
            stale_ids.append(tid)
            
    if stale_ids:
        for tid in stale_ids: del track_data[tid]

def log_repair(msg):
    if 'repair_logs' not in st.session_state: st.session_state.repair_logs = []
    st.session_state.repair_logs.append(f"[{datetime.now().strftime('%H:%M')}] {msg}")
    if len(st.session_state.repair_logs) > 5: st.session_state.repair_logs.pop(0)
def get_ist_now():
    now = datetime.now()
    return now.strftime("%H:%M:%S"), now.strftime("%d/%m/%Y"), now.strftime("%A")
VEHICLE_CLASSES = {2: "CAR", 3: "BIKE", 5: "BUS", 7: "TRUCK", 9: "SIGNAL"}
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

@st.cache_resource
def load_ambulance_ref():
    ref_path = "assets/ambulance.webp"
    if not os.path.exists(ref_path): return None, None
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref_img is None: return None, None
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(ref_img, None)
    kp, des = orb.detectAndCompute(ref_img, None)
    return kp, des

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def check_ambulance_text(crop):
    if crop is None or crop.size == 0: return False
    try:
        reader = load_ocr()
        results = reader.readtext(crop, detail=0)
        keywords = ["AMBULANCE", "EMERGENCY", "108", "HOSPITAL", "RESCUE", "EMS"]
        for text in results:
            clean_text = text.upper().replace(" ", "")
            for k in keywords:
                if k in clean_text: return True
    except:
        pass
    return False

def check_emergency_colors(crop):
    if crop is None or crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pct = (cv2.countNonZero(white_mask) / crop.size) * 100
    
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    red_pct = (cv2.countNonZero(red_mask) / crop.size) * 100
    
    return white_pct > 25 and red_pct > 0.8

def is_ambulance_match(crop, tid):
    if crop is None or crop.size == 0: return False
    h, w = crop.shape[:2]
    aspect = w / h
    if not (1.0 < aspect < 4.0): return False
    if check_emergency_colors(crop):
        return True
    return False

def get_signal_state(crop):
    if crop is None or crop.size == 0: return "UNKNOWN"
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    top_zone = hsv[0:h//3, :]
    lower_red1, upper_red1 = np.array([0, 70, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 70]), np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(cv2.inRange(top_zone, lower_red1, upper_red1), cv2.inRange(top_zone, lower_red2, upper_red2))
    btm_zone = hsv[2*h//3:h, :]
    lower_green, upper_green = np.array([35, 40, 40]), np.array([95, 255, 255])
    green_mask = cv2.inRange(btm_zone, lower_green, upper_green)
    red_px = cv2.countNonZero(red_mask)
    green_px = cv2.countNonZero(green_mask)
    if red_px > green_px and red_px > 5: return "RED"
    if green_px > red_px and green_px > 5: return "GREEN"
    return "UNKNOWN"
if 'active' not in st.session_state: st.session_state.active = False
if 'last_siren_time' not in st.session_state: st.session_state.last_siren_time = 0
if 'violation_history' not in st.session_state: st.session_state.violation_history = deque(maxlen=50)
if 'risk_history' not in st.session_state: st.session_state.risk_history = deque([0]*60, maxlen=60)
if 'ambulance_sim' not in st.session_state: st.session_state.ambulance_sim = False
if 'ambulance_detected' not in st.session_state: st.session_state.ambulance_detected = set()
if 'current_video' not in st.session_state: st.session_state.current_video = "assets/traffic.mp4"
if 'signal_seen' not in st.session_state: st.session_state.signal_seen = False
if 'last_signal_state' not in st.session_state: st.session_state.last_signal_state = "UNKNOWN"
if 'tracking_engine' not in st.session_state: st.session_state.tracking_engine = TrackingEngine()
if 'trajectory_engine' not in st.session_state: st.session_state.trajectory_engine = TrajectoryEngine()
if 'emergency_engine' not in st.session_state: st.session_state.emergency_engine = EmergencyEngine()
if 'obstruction_engine' not in st.session_state: st.session_state.obstruction_engine = ObstructionEngine()
if 'risk_engine' not in st.session_state: st.session_state.risk_engine = RiskEngine()
track_data = {}
smooth_boxes = {}

def reset_engine_state():
    st.session_state.violation_history.clear()
    st.session_state.risk_history = deque([0]*60, maxlen=60)
    st.session_state.ambulance_detected.clear()
    if 'ocr_checked_ids' in st.session_state: st.session_state.ocr_checked_ids.clear()
    st.session_state.signal_seen = False
    track_data.clear()

def get_officer_advisory(v_type, v_name, speed, tid):
    advisories = {
        "TAILGATING": f"Unsafe following distance for ID #{tid}.",
        "ILLEGAL LANE CHANGE": f"Unsafe lane maneuver for ID #{tid}.",
        "COLLISION RISK": f"CRITICAL: ID #{tid} on collision course.",
        "OUT OF CONTROL": f"Erratic behavior detected for ID #{tid}.",
        "WRONG LANE": f"Wrong-way driving for ID #{tid}.",
        "OVERSPEED": f"Speed violation for ID #{tid} ({speed}km/h).",
        "SIGNAL VIOLATION": f"Red light violation for ID #{tid}."
    }
    return advisories.get(v_type, "Monitoring vehicle behavior.")

VIOL_PRIORITY = {
    "COLLISION RISK": 100,
    "OUT OF CONTROL": 90,
    "WRONG LANE": 80,
    "SIGNAL VIOLATION": 70,
    "OVERSPEED": 60,
    "TAILGATING": 50,
    "ILLEGAL LANE CHANGE": 40,
    "AMBULANCE": 10,
    None: 0
}

def set_violation(d, new_v):
    curr_v = d.get('v_type')
    if VIOL_PRIORITY.get(new_v, 0) > VIOL_PRIORITY.get(curr_v, 0):
        d['v_type'] = new_v


with st.sidebar:
    st.title("Command Center")
    st.markdown("AI Traffic Intelligence")
    
    with st.expander("Feed Selection", expanded=True):
        video_files = []
        if os.path.exists("assets"):
             video_files.extend([f"assets/{f}" for f in os.listdir("assets") if f.endswith(('.mp4', '.avi', '.mov'))])
        video_files.extend([f for f in os.listdir(".") if f.endswith(('.mp4', '.avi', '.mov'))])
        
        video_files = sorted(list(set(video_files)))
            
        if video_files:
            try: default_idx = video_files.index(st.session_state.current_video)
            except: default_idx = 0
            selected_video = st.selectbox("Select Feed", video_files, index=default_idx)
            if selected_video != st.session_state.current_video:
                st.session_state.current_video = selected_video
                reset_engine_state()
                st.rerun()
        
    with st.expander("AI Config", expanded=True):
        PEAK_ACCURACY = st.toggle("Peak Accuracy Mode", value=False, help="Disables frame skipping and maximizes resolution for 100% precision.")
        AI_CONF = st.slider("Confidence", 0.1, 0.9, 0.35)
        
        MODEL_TYPE = st.selectbox("Model Limit", ["Nano (Fast)", "Small (Balanced)", "Medium (Precise)"], index=1)
        model_map = {"Nano (Fast)": "yolov8n.pt", "Small (Balanced)": "yolov8s.pt", "Medium (Precise)": "yolov8m.pt"}
        SELECTED_MODEL = model_map[MODEL_TYPE]

        if PEAK_ACCURACY:
            IMGSZ, AI_SKIP = 640, 1
            st.info(f"Peak Accuracy: Skip=1, Res=640, Model={MODEL_TYPE}")
        else:
            IMGSZ = st.selectbox("Resolution", [320, 480, 640], index=2)
            AI_SKIP = st.slider("Skip Rate", 1, 10, 3)
            
    if 'selected_model_path' not in st.session_state: st.session_state.selected_model_path = 'yolov8s.pt'
    if st.session_state.selected_model_path != SELECTED_MODEL:
            st.session_state.selected_model_path = SELECTED_MODEL
            st.rerun()
    
    with st.expander("Emergency", expanded=True):
        st.session_state.ambulance_sim = st.toggle("Simulate Ambulance", value=st.session_state.ambulance_sim)
    
    with st.expander("Calibration", expanded=True):
        DIVIDER_TOP_X = st.slider("Divider Top", 0, 800, 400)
        DIVIDER_BTM_X = st.slider("Divider Bottom", 0, 800, 400)
        STOP_LINE_Y = st.slider("Stop Line Y", 0, 600, 300)
        FLOW_LOGIC = st.radio("Flow", ["Left=Down / Right=Up", "Left=Up / Right=Down"])
        SPEED_LIMIT = st.slider("Speed Limit", 10, 120, 60)
        SPEED_SCALE = st.slider("Scale", 0.1, 5.0, 1.0)
    
    st.markdown("---")
    if not st.session_state.active:
        if st.button("INITIALIZE SYSTEM"):
            st.session_state.active = True
            st.toast("SENTINEL COMMAND INITIALIZED: System Ready")
            st.rerun()
    else:
        if st.button("DEACTIVATE SYSTEM"):
            st.session_state.active = False
            st.rerun()

col_head, col_clock = st.columns([3, 1])
with col_head: st.title("Smart Traffic Engine")
with col_clock: inject_live_clock()

tab1, tab2 = st.tabs(["Intelligence", "Database"])

with tab1:
    col_vid, col_metrics = st.columns([3, 1])
    with col_vid:
        vid_placeholder = st.empty()
        status_placeholder = st.empty()
    with col_metrics:
        st.markdown("#### Tactical Advisory")
        recent_advisory = st.empty()
        st.markdown("---")
        risk_metric = st.empty()
        viol_metric = st.empty()
        d_col1, d_col2 = st.columns(2)
        l_density = d_col1.empty()
        r_density = d_col2.empty()
        st.markdown("---")
        risk_chart = st.empty()

with tab2:
    log_placeholder = st.empty()
def process_vehicle_logic(tid, cx, cy, t, h_frame):
    if tid not in track_data:
        buf_size = 60 if st.session_state.get('peak_mode_active', False) else 40
        track_data[tid] = {'history': deque(maxlen=buf_size), 'speed_buffer': deque(maxlen=buf_size//2), 'viol_buffer': deque(maxlen=25), 'vector': (0, 0), 'lane': None, 'last_lane_change': 0}
    
    data = track_data[tid]
    data['history'].append((cx, cy, t))
    speed_kmh, v_type, vector = 0, None, (0, 0)
    progress = cy / h_frame
    mid_x = DIVIDER_TOP_X * (1 - progress) + DIVIDER_BTM_X * progress
    current_lane = 0 if cx < mid_x else 1
    if data['lane'] is not None and data['lane'] != current_lane:
        if time.time() - data['last_lane_change'] > 2:
            v_type = "ILLEGAL LANE CHANGE"
            data['last_lane_change'] = time.time()
    data['lane'] = current_lane
    if len(data['history']) >= 2:
        (px, py, pt) = data['history'][-2]
        dt = t - pt
        if dt > 0.001:
            dx, dy = cx - px, cy - py
            vector = (dx, dy)
            data['vector'] = vector
            ppm = (11.5 * SPEED_SCALE) * (0.4 + (cy / h_frame)**1.6 * 2.2)
            raw_s = (math.sqrt(dx**2 + dy**2) / ppm / dt) * 3.6
            if 2 < raw_s < 250: data['speed_buffer'].append(raw_s)
            if data['speed_buffer']:
                speed_kmh = int(np.average(data['speed_buffer'], weights=np.linspace(0.5, 1.0, len(data['speed_buffer']))))
            is_wrong = 0
            if abs(dy) > 0.3:
                if "Left=Down" in FLOW_LOGIC:
                    if (cx < mid_x and dy < -0.2) or (cx > mid_x and dy > 0.2): is_wrong = 1
                else:
                    if (cx < mid_x and dy > 0.2) or (cx > mid_x and dy < -0.2): is_wrong = 1
            is_speeding = 1 if speed_kmh > SPEED_LIMIT else 0
            is_erratic = 1 if (abs(dx) > abs(dy) * 2.5 and speed_kmh > 35) else 0
            data['viol_buffer'].append(1 if (is_wrong or is_speeding or is_erratic or v_type) else 0)
            
            if len(data['viol_buffer']) >= 6 and (sum(data['viol_buffer']) / len(data['viol_buffer'])) > 0.5:
                if speed_kmh > 120 or is_erratic: v_type = "OUT OF CONTROL"
                elif is_speeding: v_type = "OVERSPEED"
                elif is_wrong: v_type = "WRONG LANE"
                
    return speed_kmh, v_type, vector

def analyze_safety(detections, h_frame):
    emergency_ids, safety_links = set(), []
    vehicles = [d for d in detections if d['name'] != "SIGNAL"]
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            d1, d2 = vehicles[i], vehicles[j]
            l1 = 0 if d1['pos'][0] < (DIVIDER_TOP_X + (DIVIDER_BTM_X-DIVIDER_TOP_X)*(d1['pos'][1]/h_frame)) else 1
            l2 = 0 if d2['pos'][0] < (DIVIDER_TOP_X + (DIVIDER_BTM_X-DIVIDER_TOP_X)*(d2['pos'][1]/h_frame)) else 1
            if l1 == l2:
                dist_px = math.sqrt((d1['pos'][0]-d2['pos'][0])**2 + (d1['pos'][1]-d2['pos'][1])**2)
                dist_m = dist_px / (15.0 * (0.5 + (max(d1['pos'][1], d2['pos'][1]) / h_frame) * 1.5))
                avg_v = (d1['speed'] + d2['speed']) / 2 / 3.6
                if avg_v > 5 and dist_m < (avg_v * 1.2):
                    set_violation(d1, "TAILGATING")
                    set_violation(d2, "TAILGATING")
                    safety_links.append({'p1': d1['pos'], 'p2': d2['pos'], 'dist': dist_m})
            p1_f = (d1['pos'][0] + d1['vector'][0]*12, d1['pos'][1] + d1['vector'][1]*12)
            p2_f = (d2['pos'][0] + d2['vector'][0]*12, d2['pos'][1] + d2['vector'][1]*12)
            if math.sqrt((p1_f[0]-p2_f[0])**2 + (p1_f[1]-p2_f[1])**2) < 25:
                set_violation(d1, "COLLISION RISK")
                set_violation(d2, "COLLISION RISK")
                emergency_ids.add(d1['id']); emergency_ids.add(d2['id'])
    return emergency_ids, safety_links

def run_engine():
    try:
        model = load_model(st.session_state.get('selected_model_path', 'yolov8n.pt'))
        
        video_path = st.session_state.current_video
        if not os.path.exists(video_path):
            st.error(f"Video file not found: {video_path}")
            st.info("Please add a video file to the 'assets' folder or root directory and refresh.")
            st.session_state.active = False
            st.session_state.active = False
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Failed to open video source: {video_path}")
            st.session_state.active = False
            return
        DISPLAY_W, frame_idx, cached_detections, prev_t = 640, 0, [], time.time()
        fps_history, current_skip = deque(maxlen=30), AI_SKIP
        st.session_state.peak_mode_active = PEAK_ACCURACY
        
        while cap.isOpened() and st.session_state.active:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_idx += 1
            h_orig, w_orig = frame.shape[:2]
            if w_orig > 0:
                canvas = cv2.resize(frame, (DISPLAY_W, int(h_orig * (DISPLAY_W / w_orig))))
            else:
                continue
            h_c, w_c = canvas.shape[:2]
            if frame_idx % current_skip == 0 or not cached_detections:
                results = model.track(canvas, persist=True, conf=AI_CONF, imgsz=IMGSZ, verbose=False, classes=[2,3,5,7,9], tracker="custom_tracker.yaml")
                new_detections = []
                if results and results[0].boxes.id is not None:
                    boxes, ids, clss = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.int().cpu().numpy(), results[0].boxes.cls.int().cpu().numpy()
                    for box, tid, cid in zip(boxes, ids, clss):
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        v_name = VEHICLE_CLASSES.get(cid, "OBJ")
                        if v_name == "SIGNAL":
                            state = get_signal_state(canvas[max(0,y1):min(h_c,y2), max(0,x1):min(w_c,x2)])
                            new_detections.append({'id': tid, 'box': (x1,y1,x2,y2), 'name': v_name, 'state': state, 'pos': (cx, cy)})
                        else:
                            speed, v_type, vector = process_vehicle_logic(tid, cx, cy, time.time(), h_c)
                            new_detections.append({'id': tid, 'box': (x1,y1,x2,y2), 'speed': speed, 'v_type': v_type, 'name': v_name, 'pos': (cx, cy), 'vector': vector})
                cached_detections = new_detections


            emergency_ids, safety_links = analyze_safety(cached_detections, h_c)
            signals = [d for d in cached_detections if d['name'] == "SIGNAL"]
            global_signal_state = "UNKNOWN"
            if signals:
                if not st.session_state.signal_seen:
                    status_placeholder.markdown('<div class="signal-badge">TRAFFIC SIGNAL ACQUIRED</div>', unsafe_allow_html=True)
                    st.session_state.signal_seen = True
                main_signal = max(signals, key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]))
                detected_state = main_signal['state']
                if detected_state != "UNKNOWN":
                    st.session_state.last_signal_state = detected_state
                else:
                    if st.session_state.last_signal_state != "UNKNOWN":
                        log_repair(f"Signal ID #{main_signal['id']} state persisted: {st.session_state.last_signal_state}")
                global_signal_state = st.session_state.last_signal_state
            else:
                if st.session_state.signal_seen:
                    status_placeholder.empty()
                    st.session_state.signal_seen = False
                    st.session_state.last_signal_state = "UNKNOWN"

            auto_amb_active = False
            for d in cached_detections:
                if d['name'] in ["CAR", "TRUCK", "BUS"] and d['id'] not in st.session_state.ambulance_detected:
                    if is_ambulance_match(canvas[max(0,d['box'][1]):min(h_c,d['box'][3]), max(0,d['box'][0]):min(w_c,d['box'][2])], d['id']):
                        st.session_state.ambulance_detected.add(d['id'])
                if d['id'] in st.session_state.ambulance_detected: d['v_type'] = "AMBULANCE"; auto_amb_active = True

            amb_present = st.session_state.ambulance_sim or auto_amb_active
            priority_v = None
            vehicles = [d for d in cached_detections if d['name'] != "SIGNAL"]
            if amb_present and vehicles:
                auto_amb = next((d for d in vehicles if d['id'] in st.session_state.ambulance_detected), None)
                priority_v = auto_amb if auto_amb else min(vehicles, key=lambda x: abs(x['pos'][0] - w_c//2))
                priority_v['v_type'] = "AMBULANCE"

            emergency_active = amb_present or len(emergency_ids) > 0 or any(d.get('v_type') == "OUT OF CONTROL" for d in cached_detections)
            if amb_present:
                st.markdown('<div class="evacuate-banner">EMERGENCY PRIORITY MODE ACTIVE</div>', unsafe_allow_html=True)
            elif PEAK_ACCURACY:
                status_placeholder.caption("PEAK ACCURACY MODE ACTIVE • 100% FRAME ANALYSIS")

            cv2.line(canvas, (DIVIDER_TOP_X, 0), (DIVIDER_BTM_X, h_c), (0, 255, 255), 2, cv2.LINE_AA)
            
            arrow_color = (255, 255, 0)
            mid_y = h_c // 2

            l_cx = (DIVIDER_TOP_X + DIVIDER_BTM_X) // 4
            if "Left=Down" in FLOW_LOGIC:
                cv2.arrowedLine(canvas, (l_cx, mid_y - 50), (l_cx, mid_y + 50), arrow_color, 2, tipLength=0.3)
            else:
                cv2.arrowedLine(canvas, (l_cx, mid_y + 50), (l_cx, mid_y - 50), arrow_color, 2, tipLength=0.3)
            

            r_cx = int((DIVIDER_TOP_X + DIVIDER_BTM_X) * 0.75)
            if "Right=Down" in FLOW_LOGIC: # Logic check simplifed: if Left=Up, Right=Down
                cv2.arrowedLine(canvas, (r_cx, mid_y - 50), (r_cx, mid_y + 50), arrow_color, 2, tipLength=0.3)
            else: # Left=Down, so Right=Up
                cv2.arrowedLine(canvas, (r_cx, mid_y + 50), (r_cx, mid_y - 50), arrow_color, 2, tipLength=0.3)

            if signals: cv2.line(canvas, (0, STOP_LINE_Y), (w_c, STOP_LINE_Y), (255, 255, 255), 1, cv2.LINE_AA)
            if not amb_present:
                for link in safety_links: cv2.line(canvas, link['p1'], link['p2'], (0, 165, 255), 1, cv2.LINE_AA)


            tracked = st.session_state.tracking_engine.update(cached_detections)
            trajectories = st.session_state.trajectory_engine.predict(tracked)
            emergency_routes = st.session_state.emergency_engine.detect(tracked, canvas)
            obstructions = st.session_state.obstruction_engine.compute(emergency_routes, trajectories)
            
            final_scores = st.session_state.risk_engine.compute(tracked, trajectories, obstructions, None)

            



            for tid, points in trajectories.items():
                if len(points) > 5:
                    color = (255, 200, 0) # Cyan-ish/Blue-ish in BGR
                    
                    if tid in final_scores and final_scores[tid] > 50:
                        color = (0, 0, 255) # Red for high risk
                    
                    pts = np.array(points, np.int32)
                    for i in range(0, len(pts)-1, 3):
                        cv2.line(canvas, tuple(pts[i]), tuple(pts[i+1]), color, 2, cv2.LINE_AA)
                    
                    if len(pts) > 0:
                        cv2.circle(canvas, tuple(pts[-1]), 3, color, -1)

            for eid, points in emergency_routes.items():
                if len(points) > 1:
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts], False, (255, 0, 0), 2)
            
            for obs in obstructions:
                 vid = obs['blocking_vehicle_id']
                 v_obj = next((o for o in tracked if o['id'] == vid), None)
                 if v_obj:
                     bx = v_obj['box']
                     cv2.rectangle(canvas, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                     cv2.putText(canvas, "OBSTRUCTION", (bx[0], bx[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            for tid, score in final_scores.items():
                 v_obj = next((o for o in tracked if o['id'] == tid), None)
                 if v_obj:
                     bx = v_obj['box']
                     cv2.putText(canvas, f"RISK:{int(score)}", (bx[0], bx[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

            viol_count = 0
            for d in cached_detections:
                if d['name'] == "SIGNAL":
                    color = (0, 0, 255) if d['state'] == "RED" else (0, 255, 0) if d['state'] == "GREEN" else (150, 150, 150)
                    cv2.rectangle(canvas, (d['box'][0], d['box'][1]), (d['box'][2], d['box'][3]), color, 1, cv2.LINE_AA)
                    cv2.circle(canvas, (d['box'][0]+10, d['box'][1]-10), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(canvas, f"SIGNAL: {d['state']}", (d['box'][0]+20, d['box'][1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    continue
                if amb_present and d != priority_v: continue
                
                color_bgr = (241, 102, 99)
                if d.get('v_type') == "AMBULANCE":
                    color_bgr = (255, 255, 255)
                    if int(time.time() * 5) % 2 == 0: color_bgr = (68, 68, 239)
                
                elif global_signal_state == "RED":
                    if d['pos'][1] > STOP_LINE_Y and d.get('vector', (0,0))[1] > 0.3:
                        set_violation(d, "SIGNAL VIOLATION")
                
                if d.get('v_type'): viol_count += 1; color_bgr = (68, 68, 239)
                
                tid = d['id']
                x1, y1, x2, y2 = d['box']
                if tid in smooth_boxes:
                    alpha = 0.4  # Smoothing factor (0.4 = 40% new, 60% old) - Strong smoothing
                    sx1, sy1, sx2, sy2 = smooth_boxes[tid]
                    x1 = int(alpha * x1 + (1-alpha) * sx1)
                    y1 = int(alpha * y1 + (1-alpha) * sy1)
                    x2 = int(alpha * x2 + (1-alpha) * sx2)
                    y2 = int(alpha * y2 + (1-alpha) * sy2)
                smooth_boxes[tid] = (x1, y1, x2, y2)
                
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
                label = f"ID:{d['id']} {d.get('v_type', d['name'])}"
                if d.get('speed'): label += f" {d['speed']}km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(canvas, (x1, y1-th-8), (x1+tw+8, y1), color_bgr, -1)
                cv2.putText(canvas, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

                if d.get('v_type') and d['v_type'] != "AMBULANCE":
                    existing = next((s for s in st.session_state.violation_history if s['id'] == d['id']), None)
                    if existing:
                        if d['v_type'] not in existing['type']: existing['type'] += f", {d['v_type']}"
                        existing['speed'] = max(existing['speed'], d.get('speed', 0))
                    else:
                        crop = canvas[max(0,y1-20):min(h_c,y2+20), max(0,x1-20):min(w_c,x2+20)].copy()
                        if crop.size > 0:
                            st.session_state.violation_history.appendleft({'id': d['id'], 'type': d['v_type'], 'name': d['name'], 'speed': d.get('speed', 0), 'img': cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 'advisory': get_officer_advisory(d['v_type'], d['name'], d.get('speed', 0), d['id']), 'time': time.strftime("%H:%M:%S")})

            if emergency_active and (time.time() - st.session_state.last_siren_time > (2 if amb_present else 5)):
                trigger_siren(); st.session_state.last_siren_time = time.time()

            risk_val = min(100, (len(cached_detections) * 4) + (viol_count * 20))
            st.session_state.risk_history.append(risk_val)
            if frame_idx % 10 == 0:
                risk_metric.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Index</div><div class='metric-value'>{risk_val}%</div></div>", unsafe_allow_html=True)
                viol_metric.markdown(f"<div class='metric-card'><div class='metric-label'>Violations</div><div class='metric-value'>{viol_count}</div></div>", unsafe_allow_html=True)
                l_c = sum(1 for d in cached_detections if d['pos'][0] < (DIVIDER_TOP_X + (DIVIDER_BTM_X-DIVIDER_TOP_X)*(d['pos'][1]/h_c)))
                l_density.markdown(f"<div class='metric-card' style='padding:10px;'><div class='metric-label'>Left</div><div class='metric-value'>{l_c}</div></div>", unsafe_allow_html=True)
                r_density.markdown(f"<div class='metric-card' style='padding:10px;'><div class='metric-label'>Right</div><div class='metric-value'>{len(cached_detections)-l_c}</div></div>", unsafe_allow_html=True)
                risk_chart.line_chart(list(st.session_state.risk_history), height=120)
                if st.session_state.violation_history:
                    latest = st.session_state.violation_history[0]
                    recent_advisory.markdown(f"<div class='log-entry'><strong>{latest['type']}</strong><div class='advisory-text'>{latest['advisory']}</div></div>", unsafe_allow_html=True)

            if frame_idx % 30 == 0:
                with log_placeholder.container():
                    for s in list(st.session_state.violation_history)[:5]:
                        cols = st.columns([1, 4])
                        cols[0].image(s['img'], width=80)
                        cols[1].markdown(f"**{s['name']} #{s['id']}** • {s['time']}\n\n<span style='color:#f43f5e;'>{s['type']}</span>", unsafe_allow_html=True)
                        st.markdown("<div style='height:1px; background:rgba(255,255,255,0.1); margin:10px 0;'></div>", unsafe_allow_html=True)

            vid_placeholder.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
            curr_t = time.time()
            fps = 1.0 / (curr_t - prev_t) if (curr_t - prev_t) > 0 else 0
            prev_t = curr_t
            fps_history.append(fps)
            if len(fps_history) == 30:
                avg_fps = sum(fps_history) / 30
                if avg_fps < 12 and current_skip < 10: current_skip += 1
                elif avg_fps > 25 and current_skip > AI_SKIP: current_skip -= 1
                fps_history.clear()
            status_placeholder.caption(f"Status: Operational • {fps:.1f} FPS • Skip: {current_skip}")
            time.sleep(max(0, 0.016 - (time.time() - loop_start)))
            if frame_idx % 60 == 0: run_self_repair()

        cap.release()
    except Exception as e:
        traceback.print_exc()
        st.error(f"System Error: {str(e)}")

if st.session_state.active: run_engine()
else: st.info("System Standby. Initialize to start monitoring.")
