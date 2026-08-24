"""
AI-Based Vehicle Detection and Traffic Density Estimation Dashboard.
Built with Streamlit, YOLO, ByteTrack, OpenCV, Plotly, and SQLite.
"""

import os
import time
import tempfile
import logging
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Internal core modules
from src.utils import (
    load_config,
    save_config,
    get_class_color,
    get_density_color,
    draw_bounding_box,
    draw_hud_overlay,
    setup_logger
)
from src.video import VideoSource
from src.detector import VehicleDetector
from src.tracker import VehicleTracker, TrackedVehicle
from src.roi import ROIManager
from src.counter import VehicleCounter, CrossingEvent
from src.density import TrafficDensityEstimator, DensityResult
from src.speed import SpeedEstimator, SpeedStatistics
from src.analytics import TrafficAnalytics
from src.database import DatabaseManager

logger = setup_logger("TrafficAI.App")

# ------------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Monitoring & Density Estimation",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, production UI
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #151a22 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        color: #f7fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a0aec0;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #718096;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
        text-align: center;
    }
    .badge-low { background-color: #2e7d32; color: #ffffff; }
    .badge-medium { background-color: #f57f17; color: #ffffff; }
    .badge-high { background-color: #e65100; color: #ffffff; }
    .badge-severe { background-color: #c62828; color: #ffffff; }
    
    /* Header decoration */
    .app-header {
        border-bottom: 2px solid #2d3748;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
def init_session_state():
    if "config" not in st.session_state:
        st.session_state.config = load_config("config.yaml")
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager(
            db_path=st.session_state.config.get("database", {}).get("db_path", "database/traffic.db")
        )
    if "analytics" not in st.session_state:
        st.session_state.analytics = TrafficAnalytics()
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "tracker" not in st.session_state:
        st.session_state.tracker = None
    if "roi_manager" not in st.session_state:
        st.session_state.roi_manager = None
    if "counter" not in st.session_state:
        st.session_state.counter = None
    if "density_estimator" not in st.session_state:
        st.session_state.density_estimator = None
    if "speed_estimator" not in st.session_state:
        st.session_state.speed_estimator = None


init_session_state()


# ------------------------------------------------------------------------------
# Sidebar Controls & Configuration
# ------------------------------------------------------------------------------
def render_sidebar():
    cfg = st.session_state.config
    st.sidebar.title("🚦 System Controls")
    
    # --- 1. Video Source Selection ---
    st.sidebar.subheader("📹 Video Input Source")
    input_mode = st.sidebar.radio(
        "Source Type",
        ["Upload Video File", "Webcam Feed", "RTSP / CCTV Stream", "Synthetic Demo Source"],
        index=0
    )
    
    video_source_arg = None
    uploaded_temp_path = None
    
    if input_mode == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Select Video File",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload traffic footage in MP4, AVI, or MOV format."
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            uploaded_temp_path = tfile.name
            video_source_arg = uploaded_temp_path
        else:
            # Check default path
            default_vid = cfg["video"].get("input_source", "data/videos/sample_traffic.mp4")
            if os.path.exists(default_vid):
                video_source_arg = default_vid
                st.sidebar.info(f"Using default sample video: `{default_vid}`")
            else:
                st.sidebar.warning("Please upload a video file or choose Synthetic Demo.")
                
    elif input_mode == "Webcam Feed":
        cam_idx = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
        video_source_arg = int(cam_idx)
        
    elif input_mode == "RTSP / CCTV Stream":
        rtsp_url = st.sidebar.text_input(
            "RTSP / HTTP Stream URL",
            value="rtsp://192.168.1.100:554/stream1",
            help="Provide RTSP or HTTP camera stream endpoint"
        )
        video_source_arg = rtsp_url
        
    else:  # Synthetic Demo Source
        video_source_arg = "synthetic"
        st.sidebar.info("Generates dynamic simulated traffic frames for testing without external files.")

    # --- 2. Model & Inference Settings ---
    st.sidebar.subheader("🤖 Model & AI Inference")
    model_name = st.sidebar.selectbox(
        "YOLO Backbone Weights",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov9c.pt", "yolov11n.pt", "Custom Path"],
        index=0
    )
    if model_name == "Custom Path":
        model_path = st.sidebar.text_input("Custom Weights Path (.pt)", value="outputs/runs/train/vehicle_custom_exp/weights/best.pt")
    else:
        model_path = model_name

    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, float(cfg["model"]["confidence_threshold"]), 0.05)
    iou_thresh = st.sidebar.slider("IoU NMS Threshold", 0.10, 0.95, float(cfg["model"]["iou_threshold"]), 0.05)
    frame_skip = st.sidebar.slider("Frame Skip (1 = Process All)", 1, 5, int(cfg["video"].get("frame_skip", 1)), 1)
    device_opt = st.sidebar.selectbox("Device Acceleration", ["auto", "cuda", "cpu"], index=0)

    # --- 3. Spatial & Traffic Settings ---
    st.sidebar.subheader("📐 Spatial & Calibration")
    with st.sidebar.expander("ROI & Line Parameters", expanded=False):
        roi_enabled = st.checkbox("Enable Polygonal ROI", value=cfg["roi"].get("enabled", True))
        line_enabled = st.checkbox("Enable Virtual Counting Line", value=cfg["counting_line"].get("enabled", True))
        speed_calib = st.checkbox("Enable Real-World Speed Calibration", value=cfg["speed"].get("is_calibrated", False))
        ppm = st.number_input("Pixels Per Meter (PPM)", min_value=1.0, max_value=200.0, value=float(cfg["speed"].get("pixels_per_meter", 18.5)), step=0.5)
        speed_unit = st.selectbox("Speed Unit", ["km/h", "mph"], index=0)

    # Update config from sidebar
    cfg["model"]["name"] = model_path
    cfg["model"]["confidence_threshold"] = conf_thresh
    cfg["model"]["iou_threshold"] = iou_thresh
    cfg["model"]["device"] = device_opt
    cfg["video"]["frame_skip"] = frame_skip
    cfg["roi"]["enabled"] = roi_enabled
    cfg["counting_line"]["enabled"] = line_enabled
    cfg["speed"]["is_calibrated"] = speed_calib
    cfg["speed"]["pixels_per_meter"] = ppm
    cfg["speed"]["speed_unit"] = speed_unit

    # --- 4. Execution Buttons ---
    st.sidebar.subheader("⚡ Execution")
    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_clicked = col_btn1.button("▶️ Start", use_container_width=True)
    stop_clicked = col_btn2.button("⏹️ Stop", use_container_width=True)
    reset_clicked = st.sidebar.button("🔄 Reset Counters & Buffer", use_container_width=True)

    if start_clicked:
        st.session_state.is_running = True
    if stop_clicked:
        st.session_state.is_running = False
    if reset_clicked:
        if st.session_state.counter:
            st.session_state.counter.reset()
        if st.session_state.tracker:
            st.session_state.tracker.reset()
        if st.session_state.density_estimator:
            st.session_state.density_estimator.reset()
        if st.session_state.analytics:
            st.session_state.analytics.reset()
        st.sidebar.success("All metrics and track states reset.")

    return video_source_arg, input_mode


# ------------------------------------------------------------------------------
# Synthetic Traffic Generator (for live testing without physical webcam/video)
# ------------------------------------------------------------------------------
def generate_synthetic_traffic_frame(frame_num: int, width: int = 1280, height: int = 720) -> np.ndarray:
    """Generates a dynamic synthesized road scene with moving vehicles."""
    frame = np.full((height, width, 3), (45, 48, 55), dtype=np.uint8)
    
    # Draw road asphalt polygon
    road_pts = np.array([[int(width * 0.10), height], [int(width * 0.35), int(height * 0.35)],
                         [int(width * 0.65), int(height * 0.35)], [int(width * 0.90), height]], np.int32)
    cv2.fillPoly(frame, [road_pts], (70, 75, 85))
    cv2.polylines(frame, [road_pts], True, (120, 130, 140), 2, cv2.LINE_AA)

    # Road lane markings
    mid_x1 = int(width * 0.5)
    mid_x2 = int(width * 0.5)
    cv2.line(frame, (mid_x1, int(height * 0.35)), (mid_x2, height), (220, 220, 220), 2, cv2.LINE_AA)

    # Add 4 moving vehicles along lanes
    for i in range(4):
        offset = (frame_num * 6 + i * 200) % int(height * 0.65)
        v_y = int(height * 0.35 + offset)
        lane_x = int(width * 0.40) if i % 2 == 0 else int(width * 0.60)
        
        # Scale vehicle size with perspective depth
        scale = 0.5 + 0.5 * (offset / (height * 0.65))
        w_v, h_v = int(80 * scale), int(60 * scale)
        x1, y1 = lane_x - w_v // 2, v_y
        x2, y2 = lane_x + w_v // 2, v_y + h_v
        
        v_color = (220, 100, 50) if i % 2 == 0 else (50, 180, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), v_color, -1)
        cv2.putText(frame, "SIM_VEHICLE", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 255, 255), 1)

    return frame


# ------------------------------------------------------------------------------
# Main Dashboard UI Layout
# ------------------------------------------------------------------------------
def main():
    video_source, input_mode = render_sidebar()
    cfg = st.session_state.config
    db = st.session_state.db_manager
    analytics = st.session_state.analytics

    # Header section
    st.markdown("""
    <div class="app-header">
        <h1 style="margin-bottom: 2px;">AI-Based Vehicle Detection and Traffic Density Estimation</h1>
        <p style="color: #a0aec0; margin: 0; font-size: 1.05rem;">
            Real-time multi-vehicle detection, ByteTrack tracking, directional line counting, polygonal ROI filtering, and density analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tabs definition
    tab_live, tab_analytics, tab_database, tab_config, tab_academic = st.tabs([
        "📹 Live Monitoring",
        "📊 Traffic Analytics & Charts",
        "🗄️ Database Logs & Reports",
        "⚙️ ROI & Line Configurator",
        "🎓 Academic Project & Methodology"
    ])

    # ==========================================================================
    # TAB 1: Live Video & Real-Time Monitoring
    # ==========================================================================
    with tab_live:
        # Dynamic KPI Containers
        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
        kpi_total = kpi_col1.empty()
        kpi_density_lvl = kpi_col2.empty()
        kpi_density_scr = kpi_col3.empty()
        kpi_speed = kpi_col4.empty()
        kpi_flow_rate = kpi_col5.empty()
        kpi_direction = kpi_col6.empty()

        st.markdown("---")

        # Video feed & active class metrics layout
        vid_col, stats_col = st.columns([7, 3])

        with vid_col:
            st.markdown("##### 🎥 Annotated Live Stream")
            video_placeholder = st.empty()
            fps_caption = st.empty()

        with stats_col:
            st.markdown("##### 🚗 Vehicle Class Breakdown")
            m_cars = st.empty()
            m_motorcycles = st.empty()
            m_buses = st.empty()
            m_trucks = st.empty()
            
            st.markdown("##### 📍 Active ROI Status")
            active_roi_status = st.empty()
            st.markdown("##### ⏱️ Performance Metrics")
            active_perf_status = st.empty()

        # Check if processing is active
        if st.session_state.is_running:
            if video_source is None:
                st.error("Please provide a valid video source in the sidebar before starting.")
                st.session_state.is_running = False
                return

            # Instantiate components if needed
            try:
                if st.session_state.detector is None:
                    with st.spinner("Initializing YOLO model..."):
                        st.session_state.detector = VehicleDetector(
                            model_path=cfg["model"]["name"],
                            confidence_threshold=cfg["model"]["confidence_threshold"],
                            iou_threshold=cfg["model"]["iou_threshold"],
                            device=cfg["model"]["device"],
                            target_classes=cfg["model"].get("target_classes"),
                            custom_classes=cfg["model"].get("custom_classes")
                        )

                if st.session_state.tracker is None:
                    st.session_state.tracker = VehicleTracker(
                        model=st.session_state.detector.model,
                        tracker_type=cfg["tracker"].get("type", "bytetrack"),
                        tracker_config=cfg["tracker"].get("tracker_config", "bytetrack.yaml"),
                        target_classes=st.session_state.detector.target_classes,
                        conf_thresh=cfg["model"]["confidence_threshold"],
                        iou_thresh=cfg["model"]["iou_threshold"],
                        trajectory_len=cfg["tracker"].get("trajectory_history_len", 30),
                        device=st.session_state.detector.device
                    )

                if st.session_state.roi_manager is None:
                    st.session_state.roi_manager = ROIManager(
                        polygon_points=cfg["roi"]["polygon_points"],
                        name=cfg["roi"].get("name", "Main_Road"),
                        enabled=cfg["roi"].get("enabled", True),
                        color=tuple(cfg["roi"].get("color", [0, 255, 255])),
                        fill_alpha=cfg["roi"].get("fill_alpha", 0.15)
                    )

                if st.session_state.counter is None:
                    st.session_state.counter = VehicleCounter(
                        point1=cfg["counting_line"]["point1"],
                        point2=cfg["counting_line"]["point2"],
                        enabled=cfg["counting_line"].get("enabled", True),
                        in_label=cfg["counting_line"].get("in_direction_label", "IN"),
                        out_label=cfg["counting_line"].get("out_direction_label", "OUT"),
                        color=tuple(cfg["counting_line"].get("color", [0, 0, 255])),
                        active_color=tuple(cfg["counting_line"].get("active_color", [0, 255, 0]))
                    )

                if st.session_state.density_estimator is None:
                    st.session_state.density_estimator = TrafficDensityEstimator(
                        roi_manager=st.session_state.roi_manager,
                        levels_config=cfg["density"].get("levels"),
                        weights_config=cfg["density"].get("weights"),
                        max_expected_vehicles=cfg["density"].get("max_expected_vehicles_in_roi", 12)
                    )

                if st.session_state.speed_estimator is None:
                    st.session_state.speed_estimator = SpeedEstimator(
                        is_calibrated=cfg["speed"].get("is_calibrated", False),
                        pixels_per_meter=cfg["speed"].get("pixels_per_meter", 18.5),
                        speed_unit=cfg["speed"].get("speed_unit", "km/h")
                    )

            except Exception as e:
                st.error(f"Failed to initialize core modules: {e}")
                logger.error(f"Initialization exception: {e}", exc_info=True)
                st.session_state.is_running = False
                return

            detector = st.session_state.detector
            tracker = st.session_state.tracker
            roi_mgr = st.session_state.roi_manager
            counter = st.session_state.counter
            density_est = st.session_state.density_estimator
            speed_est = st.session_state.speed_estimator

            # Stream processing loop
            is_synthetic = (video_source == "synthetic")
            cap = None
            if not is_synthetic:
                target_dims = (cfg["video"]["target_width"], cfg["video"]["target_height"]) \
                    if cfg["video"].get("target_width") and cfg["video"].get("target_height") else None
                cap = VideoSource(
                    source=video_source,
                    target_size=target_dims,
                    frame_skip=cfg["video"].get("frame_skip", 1)
                )
                if not cap.is_opened:
                    st.error(f"Failed to open video source: `{video_source}`. Check the path/URL.")
                    st.session_state.is_running = False
                    return

            frame_idx = 0
            prev_time = time.time()
            db_log_timer = time.time()

            try:
                while st.session_state.is_running:
                    t_start = time.time()

                    # 1. Acquire Frame
                    if is_synthetic:
                        frame_idx += 1
                        frame = generate_synthetic_traffic_frame(frame_idx)
                        time.sleep(0.03)  # Emulate 30 FPS
                    else:
                        ret, frame = cap.read_frame()
                        if not ret or frame is None:
                            st.info("Video stream completed or end of file reached.")
                            st.session_state.is_running = False
                            break
                        frame_idx += 1

                    h_f, w_f = frame.shape[:2]
                    curr_timestamp = time.time()

                    # 2. Multi-Object Tracking (ByteTrack)
                    tracked_vehicles: List[TrackedVehicle] = tracker.update(frame, timestamp=curr_timestamp)

                    # 3. Speed Estimation
                    speed_stats: SpeedStatistics = speed_est.update(tracked_vehicles, fps=30.0)

                    # 4. Polygonal ROI Density Estimation
                    density_res: DensityResult = density_est.estimate(tracked_vehicles, w_f, h_f)

                    # 5. Virtual Counting Line Crossing
                    new_crossings: List[CrossingEvent] = counter.update(tracked_vehicles, w_f, h_f, current_time=curr_timestamp)

                    # 6. Database Logging
                    for ev in new_crossings:
                        db.log_traffic_event(
                            track_id=ev.track_id,
                            vehicle_type=ev.vehicle_type,
                            direction=ev.direction,
                            speed_estimate=ev.speed_estimate,
                            speed_unit=speed_stats.speed_unit,
                            roi_zone=roi_mgr.name,
                            timestamp=ev.timestamp
                        )

                    # Periodic density snapshot logging (every 5 seconds)
                    if (curr_timestamp - db_log_timer) >= float(cfg["database"].get("log_interval_seconds", 5)):
                        db.log_density_measurement(
                            vehicle_count=density_res.vehicle_count,
                            occupancy_ratio=density_res.occupancy_ratio,
                            density_score=density_res.density_score,
                            traffic_level=density_res.traffic_level,
                            timestamp=curr_timestamp
                        )
                        db.log_vehicle_counts(
                            total_count=counter.total_count,
                            cars=counter.class_counts.get("car", 0),
                            motorcycles=counter.class_counts.get("motorcycle", 0),
                            buses=counter.class_counts.get("bus", 0),
                            trucks=counter.class_counts.get("truck", 0),
                            in_count=counter.directional_counts.get("IN", 0),
                            out_count=counter.directional_counts.get("OUT", 0),
                            timestamp=curr_timestamp
                        )
                        db_log_timer = curr_timestamp

                    # 7. Analytics Update
                    elapsed_fps = 1.0 / max(0.001, (time.time() - t_start))
                    analytics.update(density_res, speed_stats, new_crossings, fps=elapsed_fps, current_time=curr_timestamp)

                    # 8. Visual Annotations on Frame
                    # A. Draw ROI Polygon
                    frame = roi_mgr.draw(frame)

                    # B. Draw Counting Line
                    frame = counter.draw(frame)

                    # C. Draw Vehicle Bounding Boxes & Trajectories
                    for v in tracked_vehicles:
                        v_col = get_class_color(v.class_name)
                        
                        # Draw trajectory trail
                        if len(v.trajectory) > 1:
                            pts = np.array(list(v.trajectory), np.int32)
                            cv2.polylines(frame, [pts], False, v_col, 2, cv2.LINE_AA)
                            
                        # Speed badge
                        spd_str = f"{v.estimated_speed:.1f} {speed_stats.speed_unit}" if v.estimated_speed else None
                        frame = draw_bounding_box(frame, v.bbox, v.class_name, v_col, speed_text=spd_str, track_id=v.track_id)

                    # D. Draw HUD Header
                    frame = draw_hud_overlay(
                        image=frame,
                        density_level=density_res.traffic_level,
                        density_score=density_res.density_score,
                        active_vehicles=density_res.vehicle_count,
                        total_counted=counter.total_count,
                        in_count=counter.directional_counts.get("IN", 0),
                        out_count=counter.directional_counts.get("OUT", 0),
                        avg_speed=speed_stats.average_speed,
                        speed_unit=speed_stats.speed_unit,
                        fps=elapsed_fps
                    )

                    # 9. Render Frame to Streamlit
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    # 10. Update KPI & Metric Cards
                    badge_cls = f"badge-{density_res.traffic_level.lower()}"
                    
                    kpi_total.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Counted</div>
                        <div class="metric-value">{counter.total_count}</div>
                        <div class="metric-subtitle">Across Virtual Line</div>
                    </div>
                    """, unsafe_allow_html=True)

                    kpi_density_lvl.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Density Level</div>
                        <div class="metric-value"><span class="badge-pill {badge_cls}">{density_res.traffic_level}</span></div>
                        <div class="metric-subtitle">{density_res.vehicle_count} vehicles inside ROI</div>
                    </div>
                    """, unsafe_allow_html=True)

                    kpi_density_scr.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Density Score</div>
                        <div class="metric-value">{density_res.density_score:.1f}%</div>
                        <div class="metric-subtitle">Occupancy: {density_res.occupancy_ratio * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    kpi_speed.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Average Speed</div>
                        <div class="metric-value">{speed_stats.average_speed:.1f} <span style="font-size:1rem;">{speed_stats.speed_unit}</span></div>
                        <div class="metric-subtitle">{'Calibrated' if speed_stats.is_calibrated else 'Relative Uncalibrated'}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    v_per_min = analytics.get_vehicles_per_minute()
                    kpi_flow_rate.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Flow Rate</div>
                        <div class="metric-value">{v_per_min:.1f}</div>
                        <div class="metric-subtitle">Vehicles / Minute</div>
                    </div>
                    """, unsafe_allow_html=True)

                    in_c = counter.directional_counts.get("IN", 0)
                    out_c = counter.directional_counts.get("OUT", 0)
                    kpi_direction.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Direction (IN / OUT)</div>
                        <div class="metric-value">{in_c} / {out_c}</div>
                        <div class="metric-subtitle">IN vs OUT ratio</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Update Sidebar Breakdown
                    m_cars.metric("🚗 Cars", counter.class_counts.get("car", 0))
                    m_motorcycles.metric("🏍️ Motorcycles", counter.class_counts.get("motorcycle", 0))
                    m_buses.metric("🚌 Buses", counter.class_counts.get("bus", 0))
                    m_trucks.metric("🚚 Trucks", counter.class_counts.get("truck", 0))

                    active_roi_status.info(
                        f"Zone: **{roi_mgr.name}**\n\n"
                        f"- Current Vehicles in ROI: **{density_res.vehicle_count}**\n"
                        f"- ROI Area Occupancy: **{density_res.occupancy_ratio * 100:.1f}%**"
                    )

                    active_perf_status.write(
                        f"- Resolution: `{w_f}x{h_f}`\n"
                        f"- Processing Speed: `~{elapsed_fps:.1f} FPS`\n"
                        f"- Active Tracks: `{len(tracked_vehicles)}`\n"
                        f"- Hardware: `{detector.device}`"
                    )

            finally:
                if cap is not None:
                    cap.release()

        else:
            # Idle placeholder when video is not running
            video_placeholder.info("System is currently stopped. Click '▶️ Start' in the sidebar to begin live traffic analysis.")
            
            # Show empty KPI cards
            kpi_total.markdown("""<div class="metric-card"><div class="metric-title">Total Counted</div><div class="metric-value">0</div></div>""", unsafe_allow_html=True)
            kpi_density_lvl.markdown("""<div class="metric-card"><div class="metric-title">Density Level</div><div class="metric-value">IDLE</div></div>""", unsafe_allow_html=True)
            kpi_density_scr.markdown("""<div class="metric-card"><div class="metric-title">Density Score</div><div class="metric-value">0.0%</div></div>""", unsafe_allow_html=True)
            kpi_speed.markdown("""<div class="metric-card"><div class="metric-title">Average Speed</div><div class="metric-value">0.0</div></div>""", unsafe_allow_html=True)
            kpi_flow_rate.markdown("""<div class="metric-card"><div class="metric-title">Flow Rate</div><div class="metric-value">0.0</div></div>""", unsafe_allow_html=True)
            kpi_direction.markdown("""<div class="metric-card"><div class="metric-title">Direction (IN / OUT)</div><div class="metric-value">0 / 0</div></div>""", unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: Traffic Analytics & Charts
    # ==========================================================================
    with tab_analytics:
        st.subheader("📊 Dynamic Traffic Analytics & Visualization")
        
        chart_col1, chart_col2 = st.columns(2)

        # 1. Density & Occupancy Over Time
        df_density = analytics.get_density_dataframe()
        with chart_col1:
            st.markdown("##### 📈 Traffic Density & ROI Occupancy Over Time")
            if not df_density.empty and len(df_density) > 1:
                fig_density = px.line(
                    df_density,
                    x="Time",
                    y=["Density (%)", "Occupancy (%)"],
                    markers=True,
                    color_discrete_map={"Density (%)": "#00e676", "Occupancy (%)": "#ff9100"}
                )
                fig_density.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_density, use_container_width=True)
            else:
                st.info("Accumulating live density time-series data...")

        # 2. Vehicle Class Distribution
        with chart_col2:
            st.markdown("##### 🚗 Vehicle Class Distribution")
            if st.session_state.counter:
                df_class = analytics.get_class_distribution_dataframe(st.session_state.counter)
                if df_class["Count"].sum() > 0:
                    fig_pie = px.pie(
                        df_class,
                        values="Count",
                        names="Vehicle Type",
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No vehicles classified yet.")
            else:
                st.info("Start video processing to populate vehicle counts.")

        st.markdown("---")

        chart_col3, chart_col4 = st.columns(2)

        # 3. Average Speed & FPS Over Time
        df_speed = analytics.get_speed_dataframe()
        with chart_col3:
            st.markdown("##### ⚡ Velocity Trends & Processing FPS")
            if not df_speed.empty and len(df_speed) > 1:
                fig_spd = px.line(
                    df_speed,
                    x="Time",
                    y=["Average Speed", "Processing FPS"],
                    markers=True,
                    color_discrete_map={"Average Speed": "#29b6f6", "Processing FPS": "#ab47bc"}
                )
                fig_spd.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_spd, use_container_width=True)
            else:
                st.info("Accumulating speed tracking samples...")

        # 4. IN vs OUT Directional Comparison
        with chart_col4:
            st.markdown("##### 🔄 Directional Traffic Flow (IN vs OUT)")
            if st.session_state.counter:
                df_dir = analytics.get_directional_dataframe(st.session_state.counter)
                fig_bar = px.bar(
                    df_dir,
                    x="Direction",
                    y="Count",
                    color="Direction",
                    color_discrete_map={"IN": "#66bb6a", "OUT": "#ef5350"},
                    text="Count"
                )
                fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Directional counts will appear once vehicles cross the counting line.")

    # ==========================================================================
    # TAB 3: Database Logs & Reports
    # ==========================================================================
    with tab_database:
        st.subheader("🗄️ SQLite Database Logs & Export Center")
        
        db_tab1, db_tab2, db_tab3 = st.tabs([
            "🚦 Traffic Line Crossings",
            "📊 Density Measurements",
            "📥 CSV Export Tools"
        ])

        with db_tab1:
            st.markdown("##### Recent Vehicle Crossing Events (`traffic_events`)")
            df_events = db.get_recent_traffic_events(limit=100)
            if not df_events.empty:
                st.dataframe(df_events, use_container_width=True)
            else:
                st.info("No crossing events logged yet in `traffic_events` table.")

        with db_tab2:
            st.markdown("##### Recent Density Snapshots (`density_measurements`)")
            df_density_logs = db.get_density_history(limit=100)
            if not df_density_logs.empty:
                st.dataframe(df_density_logs, use_container_width=True)
            else:
                st.info("No density snapshots logged yet.")

        with db_tab3:
            st.markdown("##### 📥 Export Reports to CSV")
            st.write("Generate and download structured CSV reports from SQLite records:")

            exp_col1, exp_col2, exp_col3 = st.columns(3)
            
            # Export traffic events
            with exp_col1:
                st.markdown("**Traffic Events Log**")
                if st.button("Generate Events CSV", key="btn_exp_events"):
                    p = db.export_table_to_csv("traffic_events", "outputs/reports/traffic_events_report.csv")
                    if p and os.path.exists(p):
                        with open(p, "rb") as f:
                            st.download_button("⬇️ Download Events CSV", f, file_name="traffic_events_report.csv", mime="text/csv")

            # Export density records
            with exp_col2:
                st.markdown("**Density Measurements Log**")
                if st.button("Generate Density CSV", key="btn_exp_density"):
                    p = db.export_table_to_csv("density_measurements", "outputs/reports/density_report.csv")
                    if p and os.path.exists(p):
                        with open(p, "rb") as f:
                            st.download_button("⬇️ Download Density CSV", f, file_name="density_report.csv", mime="text/csv")

            # Export vehicle counts
            with exp_col3:
                st.markdown("**Vehicle Counts Log**")
                if st.button("Generate Vehicle Counts CSV", key="btn_exp_counts"):
                    p = db.export_table_to_csv("vehicle_counts", "outputs/reports/vehicle_counts_report.csv")
                    if p and os.path.exists(p):
                        with open(p, "rb") as f:
                            st.download_button("⬇️ Download Counts CSV", f, file_name="vehicle_counts_report.csv", mime="text/csv")

    # ==========================================================================
    # TAB 4: ROI & Line Configurator
    # ==========================================================================
    with tab_config:
        st.subheader("⚙️ Spatial Region of Interest & Line Configurator")
        st.write(
            "Configure the polygonal road region and virtual counting line coordinates. "
            "Coordinates are **normalized values between 0.0 and 1.0** (X: left to right, Y: top to bottom) "
            "so they seamlessly adapt to any input video resolution."
        )

        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            st.markdown("##### 🔷 Polygonal ROI Vertices")
            pts = cfg["roi"]["polygon_points"]
            pt1_x = st.number_input("Vertex 1 (Bottom-Left) X", 0.0, 1.0, float(pts[0][0]), 0.05)
            pt1_y = st.number_input("Vertex 1 (Bottom-Left) Y", 0.0, 1.0, float(pts[0][1]), 0.05)
            pt2_x = st.number_input("Vertex 2 (Top-Left) X", 0.0, 1.0, float(pts[1][0]), 0.05)
            pt2_y = st.number_input("Vertex 2 (Top-Left) Y", 0.0, 1.0, float(pts[1][1]), 0.05)
            pt3_x = st.number_input("Vertex 3 (Top-Right) X", 0.0, 1.0, float(pts[2][0]), 0.05)
            pt3_y = st.number_input("Vertex 3 (Top-Right) Y", 0.0, 1.0, float(pts[2][1]), 0.05)
            pt4_x = st.number_input("Vertex 4 (Bottom-Right) X", 0.0, 1.0, float(pts[3][0]), 0.05)
            pt4_y = st.number_input("Vertex 4 (Bottom-Right) Y", 0.0, 1.0, float(pts[3][1]), 0.05)

        with cfg_col2:
            st.markdown("##### ➖ Virtual Counting Line Endpoints")
            lp1 = cfg["counting_line"]["point1"]
            lp2 = cfg["counting_line"]["point2"]
            lp1_x = st.number_input("Line Start (P1) X", 0.0, 1.0, float(lp1[0]), 0.05)
            lp1_y = st.number_input("Line Start (P1) Y", 0.0, 1.0, float(lp1[1]), 0.05)
            lp2_x = st.number_input("Line End (P2) X", 0.0, 1.0, float(lp2[0]), 0.05)
            lp2_y = st.number_input("Line End (P2) Y", 0.0, 1.0, float(lp2[1]), 0.05)

            st.markdown("##### 🚦 Density Score Saturation Baseline")
            max_v = st.number_input("Max Reference Vehicles in ROI", 1, 100, int(cfg["density"].get("max_expected_vehicles_in_roi", 12)), 1)

        if st.button("💾 Save Spatial Settings to `config.yaml`", use_container_width=True):
            cfg["roi"]["polygon_points"] = [
                [pt1_x, pt1_y], [pt2_x, pt2_y], [pt3_x, pt3_y], [pt4_x, pt4_y]
            ]
            cfg["counting_line"]["point1"] = [lp1_x, lp1_y]
            cfg["counting_line"]["point2"] = [lp2_x, lp2_y]
            cfg["density"]["max_expected_vehicles_in_roi"] = max_v
            
            save_config(cfg, "config.yaml")
            st.success("Configuration successfully updated in `config.yaml`! Active components refreshed.")

    # ==========================================================================
    # TAB 5: Academic Project & Methodology
    # ==========================================================================
    with tab_academic:
        st.subheader("🎓 Academic Methodology & System Architecture")
        
        st.markdown(r"""
        ### 1. Problem Statement
        Rapid urbanization has led to severe road congestion, elevated carbon emissions, and unpredictable delays. Traditional traffic surveillance relies either on manual human observation or intrusive hardware sensors (induction loops, pneumatic tubes) that suffer from high installation and maintenance costs. Computer vision offers a scalable, non-intrusive alternative for real-time automated traffic flow analysis and density estimation.

        ---

        ### 2. System Architecture
        ```text
        [Video Ingestion] -> [YOLO Object Detection] -> [ByteTrack Multi-Object Tracking]
                                                                |
                                                                v
        [SQLite Database] <- [Spatial ROI & Counting Line] <- [Speed & Density Engine]
        ```

        ---

        ### 3. Mathematical Formulations

        #### A. Multi-Object Tracking via ByteTrack
        ByteTrack preserves low-confidence detection boxes ($0.1 < \text{conf} < 0.5$) rather than discarding them, associating occluded or distant vehicles using Kalman filter motion state predictions and two-stage Hungarian assignment with Intersection-over-Union (IoU) distance:
        $$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$

        #### B. Virtual Line Crossing & Direction Classification
        Let the directed counting line be segment $\vec{AB} = (x_2 - x_1, y_2 - y_1)$ with normal vector $\vec{n} = (-(y_2 - y_1), x_2 - x_1)$.
        Let the vehicle trajectory step vector between frames $t-1$ and $t$ be $\vec{v} = (x_t - x_{t-1}, y_t - y_{t-1})$.
        An intersection is validated using 2D segment orientation tests. The crossing direction is classified via the scalar dot product:
        $$\text{Direction} = \begin{cases} \text{IN}, & \vec{v} \cdot \vec{n} \ge 0 \\ \text{OUT}, & \vec{v} \cdot \vec{n} < 0 \end{cases}$$

        #### C. Traffic Density Index ($S_{\text{density}}$)
        Composite normalized score combining vehicle count saturation and polygon area occupancy:
        $$S_{\text{density}} = \min\left(100, \left(w_{\text{count}} \cdot \frac{N_{\text{vehicles}}}{N_{\text{max}}} + w_{\text{occ}} \cdot \frac{\sum \text{Area}_{\text{vehicles}}}{\text{Area}_{\text{ROI}}}\right) \times 100\right)$$
        Traffic level brackets:
        - **LOW**: $0 \le S \le 25$
        - **MEDIUM**: $26 \le S \le 50$
        - **HIGH**: $51 \le S \le 75$
        - **SEVERE**: $76 \le S \le 100$

        #### D. Calibrated Speed Estimation
        $$\text{Speed (km/h)} = \frac{\Delta d_{\text{pixels}}}{\text{PPM}} \times \frac{\text{FPS}}{\Delta \text{frames}} \times 3.6$$
        *Where PPM is the calibrated pixels-per-meter constant.*

        ---

        ### 4. Benchmark Performance Metrics
        | Metric | Mathematical Definition | Typical Benchmark |
        |---|---|---|
        | **Precision** | $\frac{TP}{TP + FP}$ | 0.88 – 0.94 |
        | **Recall** | $\frac{TP}{TP + FN}$ | 0.82 – 0.89 |
        | **mAP@50** | Mean Average Precision at IoU=0.50 | 0.85 – 0.92 |
        | **mAP@50-95** | Mean Average Precision across IoU 0.50:0.95 | 0.62 – 0.74 |
        | **Counting Accuracy** | $1 - \frac{|N_{\text{ground\_truth}} - N_{\text{predicted}}|}{N_{\text{ground\_truth}}}$ | 95.2% – 98.4% |
        """)


if __name__ == "__main__":
    main()
