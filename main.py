import os
import sys
import time
import csv
import math
import sqlite3
import urllib.request
import traceback
from datetime import datetime

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import cv2
import numpy as np
import mediapipe as mp

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QTabWidget,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView
)


try:
    cv2.setNumThreads(1)
except Exception:
    pass


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE, "models")
EVIDENCE = os.path.join(BASE, "evidence")
REPORTS = os.path.join(BASE, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVIDENCE, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


# ============================================================
# MEDIAPIPE FACE MODEL
# ============================================================

FACE_MODEL = os.path.join(
    MODEL_DIR,
    "face_landmarker.task"
)

YOLO_MODEL_NAME = "yolo11n.pt"

FACE_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)


# ============================================================
# LIGHTWEIGHT OPENCV OBJECT DETECTOR
# ============================================================

OBJECT_MODEL = os.path.join(
    MODEL_DIR,
    "ssd_mobilenet_v1_coco.pb"
)

OBJECT_CONFIG = os.path.join(
    MODEL_DIR,
    "ssd_mobilenet_v1_coco.pbtxt"
)

OBJECT_MODEL_URL = (
    "https://mirror.opencv.ai/"
    "ssd_mobilenet_v1_coco_2017_11_17.pb"
)

OBJECT_CONFIG_URL = (
    "https://raw.githubusercontent.com/"
    "opencv/opencv_extra/4.x/testdata/dnn/"
    "ssd_mobilenet_v1_coco.pbtxt"
)


# ============================================================
# DATABASE
# ============================================================

DB = os.path.join(
    BASE,
    "exam_events.db"
)


# ============================================================
# COCO CLASSES
# ============================================================

COCO_NAMES = {

    1: "person",

    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",

    10: "traffic light",

    11: "fire hydrant",

    13: "stop sign",
    14: "parking meter",
    15: "bench",

    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",

    27: "backpack",
    28: "umbrella",

    31: "handbag",
    32: "tie",
    33: "suitcase",

    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",

    38: "kite",

    39: "baseball bat",
    40: "baseball glove",

    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",

    44: "bottle",

    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",

    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",

    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",

    67: "dining table",

    70: "toilet",

    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",

    77: "cell phone",

    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",

    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(a, b):

    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


def eye_aspect_ratio(points, ids):

    a, b, c, e, f, g = [
        points[i]
        for i in ids
    ]

    return (
        distance(b, g) +
        distance(c, f)
    ) / (
        2 * distance(a, e) +
        1e-9
    )


def download_model(path, url, name):

    if (
        os.path.exists(path)
        and os.path.getsize(path) > 1024
    ):
        return True

    print()
    print("=" * 60)
    print("DOWNLOADING:", name)
    print("=" * 60)

    try:

        urllib.request.urlretrieve(
            url,
            path
        )

        return (
            os.path.exists(path)
            and os.path.getsize(path) > 1024
        )

    except Exception as error:

        print(
            "Download failed:",
            error
        )

        return False


# ============================================================
# MAIN APPLICATION
# ============================================================

class App(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "PROCTOR VISION • AI EXAM INTEGRITY SYSTEM"
        )

        self.resize(
            1520,
            940
        )


        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        self.cap = None
        self.face = None
        self.object_net = None
        self.detector = None


        # ----------------------------------------------------
        # MONITORING STATE
        # ----------------------------------------------------

        self.monitoring = False
        self.running = True


        # ----------------------------------------------------
        # DETECTION VALUES
        # ----------------------------------------------------

        self.faces = 0
        self.persons = 0
        self.phones = 0

        self.risk = 0
        self.events = 0

        self.head = "CENTER"
        self.eye = "OPEN"


        # ----------------------------------------------------
        # TIMERS
        # ----------------------------------------------------

        self.last_face = time.time()

        self.look_start = None
        self.closed_start = None

        self.last_event = {}

        self.last_ts = 0


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        self.frame = None
        self.frame_no = 0

        self.fps = 0
        self.fc = 0
        self.ft = time.time()


        # ----------------------------------------------------
        # ANALYTICS
        # ----------------------------------------------------

        self.total_samples = 0
        self.good_face_samples = 0
        self.good_attention_samples = 0


        # ----------------------------------------------------
        # DATABASE
        # ----------------------------------------------------

        self.conn = sqlite3.connect(
            DB
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events
            (
                id INTEGER PRIMARY KEY,

                timestamp TEXT,

                event TEXT,

                severity TEXT,

                risk INTEGER,

                details TEXT,

                snapshot TEXT
            )
            """
        )

        self.conn.commit()


        # ----------------------------------------------------
        # START
        # ----------------------------------------------------

        self.build_ui()

        self.load_models()

        self.start_camera()


        self.timer = QTimer()

        self.timer.timeout.connect(
            self.tick
        )

        self.timer.start(40)


    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        self.setStyleSheet(

            """
            QWidget
            {
                background:#070b10;
                color:#edf2f5;
                font-family:'Segoe UI', sans-serif;
            }

            QFrame#card
            {
                background:#101820;
                border:1px solid #202e3b;
                border-radius:14px;
            }

            QLabel#title
            {
                font-size:26px;
                font-weight:900;
                color:#ffffff;
                letter-spacing: 0.5px;
            }

            QLabel#sub
            {
                color:#82919e;
                font-size:12px;
            }

            QLabel#sec
            {
                color:#8fa0af;
                font-size:12px;
                font-weight:800;
                letter-spacing: 0.8px;
            }

            QLabel#big
            {
                font-size:34px;
                font-weight:900;
                color:#39d7a3;
            }

            QLabel#metric
            {
                font-size:17px;
                font-weight:900;
                color:#ffffff;
            }

            QLabel#small
            {
                color:#9aaab9;
                font-size:12px;
            }

            QLabel#alert
            {
                color:#ffb84d;
                font-size:12px;
                font-weight:700;
                line-height: 1.4px;
            }

            QPushButton
            {
                background:#15222d;
                border:1px solid #283949;
                border-radius:10px;
                padding:12px 18px;
                font-weight:800;
                color:#dce6ef;
                font-size:13px;
            }

            QPushButton:hover
            {
                border-color:#39d7a3;
                background:#1c2d3a;
                color:#ffffff;
            }

            QPushButton#primary
            {
                background:#123a30;
                border:1px solid #39d7a3;
                color:#39d7a3;
            }

            QPushButton#primary:hover
            {
                background:#184c3e;
                color:#ffffff;
            }

            QProgressBar
            {
                background:#16212b;
                border:0;
                border-radius:6px;
                height:10px;
                text-align: right;
            }

            QProgressBar::chunk
            {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #25c493, stop:1 #39d7a3);
                border-radius:6px;
            }

            QTabWidget::pane
            {
                border:1px solid #202e3b;
                border-radius:14px;
                background:#0a1016;
                top: -1px;
            }

            QTabBar::tab
            {
                background:#0e151d;
                padding:12px 26px;
                margin-right:6px;
                border-top-left-radius:10px;
                border-top-right-radius:10px;
                color:#8293a2;
                font-weight: 700;
                border: 1px solid #1a2530;
                border-bottom: none;
            }

            QTabBar::tab:selected
            {
                background:#13202b;
                color:#39d7a3;
                border-color:#2a3d4f;
                border-bottom: 2px solid #39d7a3;
            }

            QTableWidget
            {
                background:#0a1016;
                border:1px solid #1d2a36;
                border-radius: 10px;
                gridline-color:#1c2732;
                selection-background-color: #172a3a;
            }

            QHeaderView::section
            {
                background:#131f2a;
                color:#a6b6c3;
                padding:10px;
                border:0;
                border-bottom:1px solid #22313f;
                font-weight: 800;
            }
            """
        )


        root = QWidget()

        self.setCentralWidget(
            root
        )

        main_layout = QVBoxLayout(
            root
        )

        main_layout.setContentsMargins(
            24,
            20,
            24,
            18
        )
        main_layout.setSpacing(
            16
        )


        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = QHBoxLayout()

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)

        title = QLabel(
            "PROCTOR VISION"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "AI-powered examination integrity & attention analytics suite"
        )

        subtitle.setObjectName(
            "sub"
        )

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header.addLayout(
            title_layout
        )

        header.addStretch()


        self.status = QLabel(
            "● INITIALIZING SYSTEM"
        )

        self.status.setStyleSheet(
            "color:#d9a441;font-weight:900;font-size:13px;"
        )

        header.addWidget(
            self.status
        )

        main_layout.addLayout(
            header
        )


        # ----------------------------------------------------
        # TABS
        # ----------------------------------------------------

        self.tabs = QTabWidget()

        main_layout.addWidget(
            self.tabs
        )


        self.tabs.addTab(
            self.create_live_tab(),
            "  LIVE MONITOR  "
        )

        self.tabs.addTab(
            self.create_events_tab(),
            "  INCIDENT TIMELINE  "
        )

        self.tabs.addTab(
            self.create_gallery_tab(),
            "  EVIDENCE GALLERY  "
        )

        self.tabs.addTab(
            self.create_analytics_tab(),
            "  EXAM ANALYTICS  "
        )


        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = QHBoxLayout()

        left = QLabel(
            "Powered by OpenCV DNN • MediaPipe Face Landmarker • SQLite • PySide6"
        )

        left.setObjectName(
            "small"
        )

        footer.addWidget(
            left
        )

        footer.addStretch()


        right = QLabel(
            "Press [ESC] to Exit"
        )

        right.setObjectName(
            "small"
        )

        footer.addWidget(
            right
        )

        main_layout.addLayout(
            footer
        )


    # ========================================================
    # CARD
    # ========================================================

    def card(self):

        frame = QFrame()

        frame.setObjectName(
            "card"
        )

        return frame


    # ========================================================
    # LIVE TAB
    # ========================================================

    def create_live_tab(self):

        widget = QWidget()

        layout = QHBoxLayout(
            widget
        )

        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(
            16
        )


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        camera_card = self.card()

        camera_layout = QVBoxLayout(
            camera_card
        )
        camera_layout.setContentsMargins(16, 16, 16, 16)


        top = QHBoxLayout()

        label = QLabel(
            "LIVE EXAM FEED"
        )

        label.setObjectName(
            "sec"
        )

        top.addWidget(
            label
        )

        top.addStretch()


        self.fps_label = QLabel(
            "FPS --"
        )

        self.fps_label.setObjectName(
            "small"
        )

        top.addWidget(
            self.fps_label
        )

        camera_layout.addLayout(
            top
        )


        self.video = QLabel(
            "Initializing camera feed..."
        )

        self.video.setAlignment(
            Qt.AlignCenter
        )

        self.video.setMinimumSize(
            900,
            620
        )

        self.video.setStyleSheet(
            """
            background:#04070a;
            border:1px solid #1f2d3a;
            border-radius:10px;
            color:#7a8b99;
            font-size:13px;
            font-weight:600;
            """
        )

        camera_layout.addWidget(
            self.video
        )


        layout.addWidget(
            camera_card,
            3
        )


        # ----------------------------------------------------
        # SIDE PANEL
        # ----------------------------------------------------

        side = QVBoxLayout()
        side.setSpacing(12)


        # RISK
        risk_card = self.card()

        risk_layout = QVBoxLayout(
            risk_card
        )
        risk_layout.setContentsMargins(16, 16, 16, 16)
        risk_layout.setSpacing(8)


        label = QLabel(
            "EXAM INTEGRITY SCORE"
        )

        label.setObjectName(
            "sec"
        )

        self.risk_label = QLabel(
            "100 / 100"
        )

        self.risk_label.setObjectName(
            "big"
        )


        self.risk_bar = QProgressBar()

        self.risk_bar.setRange(
            0,
            100
        )

        self.risk_bar.setValue(
            100
        )


        self.risk_text = QLabel(
            "Monitoring not started"
        )

        self.risk_text.setObjectName(
            "small"
        )


        risk_layout.addWidget(label)
        risk_layout.addWidget(self.risk_label)
        risk_layout.addWidget(self.risk_bar)
        risk_layout.addWidget(self.risk_text)

        side.addWidget(
            risk_card
        )


        # TELEMETRY
        telemetry_card = self.card()

        telemetry_layout = QVBoxLayout(
            telemetry_card
        )
        telemetry_layout.setContentsMargins(16, 16, 16, 16)
        telemetry_layout.setSpacing(10)


        label = QLabel(
            "VISION TELEMETRY"
        )

        label.setObjectName(
            "sec"
        )

        telemetry_layout.addWidget(
            label
        )


        metrics = QHBoxLayout()


        self.face_label = QLabel(
            "0\nFACES"
        )

        self.face_label.setObjectName(
            "metric"
        )


        self.person_label = QLabel(
            "0\nPERSONS"
        )

        self.person_label.setObjectName(
            "metric"
        )


        self.phone_label = QLabel(
            "0\nPHONES"
        )

        self.phone_label.setObjectName(
            "metric"
        )


        metrics.addWidget(
            self.face_label
        )

        metrics.addWidget(
            self.person_label
        )

        metrics.addWidget(
            self.phone_label
        )


        telemetry_layout.addLayout(
            metrics
        )


        self.attention_label = QLabel(
            "ATTENTION • CENTER"
        )

        self.attention_label.setObjectName(
            "small"
        )


        self.eye_label = QLabel(
            "EYES • OPEN"
        )

        self.eye_label.setObjectName(
            "small"
        )


        self.event_label = QLabel(
            "EVENTS • 0"
        )

        self.event_label.setObjectName(
            "small"
        )


        telemetry_layout.addWidget(
            self.attention_label
        )

        telemetry_layout.addWidget(
            self.eye_label
        )

        telemetry_layout.addWidget(
            self.event_label
        )


        side.addWidget(
            telemetry_card
        )


        # SYSTEM
        system_card = self.card()

        system_layout = QVBoxLayout(
            system_card
        )
        system_layout.setContentsMargins(16, 16, 16, 16)
        system_layout.setSpacing(6)


        label = QLabel(
            "AI SYSTEM STATUS"
        )

        label.setObjectName(
            "sec"
        )


        self.model_label = QLabel(
            "Models: starting..."
        )

        self.model_label.setObjectName(
            "small"
        )


        self.message_label = QLabel(
            "Waiting for camera analysis."
        )

        self.message_label.setWordWrap(
            True
        )

        self.message_label.setObjectName(
            "small"
        )


        system_layout.addWidget(
            label
        )

        system_layout.addWidget(
            self.model_label
        )

        system_layout.addWidget(
            self.message_label
        )


        side.addWidget(
            system_card
        )


        # CONTROL
        control_card = self.card()

        control_layout = QVBoxLayout(
            control_card
        )
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_layout.setSpacing(10)


        label = QLabel(
            "EXAM CONTROL"
        )

        label.setObjectName(
            "sec"
        )

        control_layout.addWidget(
            label
        )


        self.monitor_button = QPushButton(
            "▶  START MONITORING"
        )

        self.monitor_button.setObjectName(
            "primary"
        )

        self.monitor_button.clicked.connect(
            self.toggle_monitoring
        )


        control_layout.addWidget(
            self.monitor_button
        )


        report_button = QPushButton(
            "↓  DOWNLOAD EXAM REPORT"
        )

        report_button.clicked.connect(
            self.export_report
        )


        control_layout.addWidget(
            report_button
        )


        side.addWidget(
            control_card
        )


        # ALERT
        alert_card = self.card()

        alert_layout = QVBoxLayout(
            alert_card
        )
        alert_layout.setContentsMargins(16, 16, 16, 16)
        alert_layout.setSpacing(6)


        label = QLabel(
            "LIVE ALERT"
        )

        label.setObjectName(
            "sec"
        )


        self.alert_label = QLabel(
            "No active alerts."
        )

        self.alert_label.setObjectName(
            "alert"
        )

        self.alert_label.setWordWrap(
            True
        )


        alert_layout.addWidget(
            label
        )

        alert_layout.addWidget(
            self.alert_label
        )


        side.addWidget(
            alert_card
        )


        layout.addLayout(
            side,
            1
        )


        return widget


    # ========================================================
    # EVENTS TAB
    # ========================================================

    def create_events_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )
        layout.setContentsMargins(16, 16, 16, 16)


        self.events_table = QTableWidget(
            0,
            6
        )


        self.events_table.setHorizontalHeaderLabels(
            [
                "Time",
                "Event",
                "Severity",
                "Risk",
                "Details",
                "Evidence"
            ]
        )


        self.events_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )


        self.events_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )


        layout.addWidget(
            self.events_table
        )


        return widget


    # ========================================================
    # GALLERY TAB
    # ========================================================

    def create_gallery_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )
        layout.setContentsMargins(16, 16, 16, 16)


        self.gallery_scroll = QScrollArea()

        self.gallery_scroll.setWidgetResizable(
            True
        )
        self.gallery_scroll.setStyleSheet("border: none; background: transparent;")


        self.gallery_widget = QWidget()
        self.gallery_widget.setStyleSheet("background: transparent;")

        self.gallery_layout = QVBoxLayout(
            self.gallery_widget
        )
        self.gallery_layout.setSpacing(14)


        self.gallery_layout.addStretch()


        self.gallery_scroll.setWidget(
            self.gallery_widget
        )


        layout.addWidget(
            self.gallery_scroll
        )


        return widget


    # ========================================================
    # ANALYTICS TAB
    # ========================================================

    def create_analytics_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )
        layout.setContentsMargins(24, 24, 24, 24)


        self.analytics_label = QLabel(
            "Start monitoring to generate exam analytics."
        )

        self.analytics_label.setObjectName(
            "small"
        )

        self.analytics_label.setWordWrap(
            True
        )


        layout.addWidget(
            self.analytics_label
        )

        layout.addStretch()


        return widget


    # ========================================================
    # LOAD AI MODELS
    # ========================================================

    def load_models(self):

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        try:

            if not download_model(
                FACE_MODEL,
                FACE_URL,
                "MediaPipe face model"
            ):

                raise RuntimeError(
                    "Could not download face_landmarker.task"
                )


            BaseOptions = mp.tasks.BaseOptions

            Vision = mp.tasks.vision


            options = Vision.FaceLandmarkerOptions(

                base_options=BaseOptions(
                    model_asset_path=FACE_MODEL
                ),

                running_mode=Vision.RunningMode.VIDEO,

                num_faces=5,

                min_face_detection_confidence=0.35,

                min_face_presence_confidence=0.35,

                min_tracking_confidence=0.35
            )


            self.face = (
                Vision.FaceLandmarker
                .create_from_options(options)
            )


            self.model_label.setText(
                "MediaPipe Face: READY"
            )


        except Exception as error:

            self.model_label.setText(
                "MediaPipe Face: ERROR"
            )

            self.message_label.setText(
                "Face analysis unavailable."
            )

            print(
                "\nMEDIAPIPE ERROR"
            )

            traceback.print_exc()


        # ----------------------------------------------------
        # WORKING OBJECT DETECTOR
        # ----------------------------------------------------
        try:
            if YOLO is None:
                raise RuntimeError("Ultralytics is not installed.")

            self.model_label.setText(
                self.model_label.text() + "\nLoading object detector..."
            )

            self.detector = YOLO(YOLO_MODEL_NAME)

            self.model_label.setText(
                self.model_label.text() + "\nObject Detector: READY"
            )

        except Exception as error:
            self.detector = None
            self.model_label.setText(
                self.model_label.text() + "\nObject Detector: ERROR"
            )
            self.message_label.setText(
                "Object detector unavailable."
            )
            print("\nOBJECT DETECTOR ERROR")
            traceback.print_exc()


    # ========================================================
    # CAMERA
    # ========================================================

    def start_camera(self):

        self.cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )


        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            960
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            540
        )


        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        if not self.cap.isOpened():

            self.message_label.setText(
                "CAMERA ERROR: webcam could not be opened."
            )

        else:

            self.message_label.setText(
                "Camera ready. Press START MONITORING."
            )


    # ========================================================
    # EVENT LOGGER
    # ========================================================

    def log_event(
        self,
        name,
        severity="MEDIUM",
        weight=8,
        details=""
    ):

        if not self.monitoring:

            return


        now = time.time()


        # Prevent duplicate events every frame.

        if (
            now -
            self.last_event.get(name, 0)
            < 4
        ):

            return


        self.last_event[name] = now


        self.risk = min(
            100,
            self.risk + weight
        )


        self.events += 1


        snapshot = ""


        # Serious incidents automatically receive evidence screenshots.

        if (
            severity in
            ("HIGH", "CRITICAL")
            and self.frame is not None
        ):

            safe_name = "".join(
                character
                if character.isalnum()
                else "_"
                for character in name
            )


            snapshot = os.path.join(

                EVIDENCE,

                f"{datetime.now():%Y%m%d_%H%M%S}_"
                f"{safe_name}.jpg"
            )


            cv2.imwrite(
                snapshot,
                self.frame
            )


        self.conn.execute(

            """
            INSERT INTO events
            (
                timestamp,
                event,
                severity,
                risk,
                details,
                snapshot
            )

            VALUES
            (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,

            (
                datetime.now().strftime(
                    "%H:%M:%S"
                ),

                name,

                severity,

                self.risk,

                details,

                snapshot
            )
        )


        self.conn.commit()


        self.alert_label.setText(
            f"⚠ {severity}: {name}\n{details}"
        )


        self.refresh_events()

        self.refresh_gallery()

        self.refresh_analytics()


    # ========================================================
    # FACE ANALYSIS
    # ========================================================

    def analyze_face(
        self,
        result
    ):

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if not result.face_landmarks:

            self.faces = 0


            if (
                self.monitoring
                and
                time.time() -
                self.last_face
                > 2.5
            ):

                self.log_event(

                    "Face missing",

                    "HIGH",

                    12,

                    "No face detected for 2.5+ seconds."
                )


            return


        self.faces = len(
            result.face_landmarks
        )


        self.last_face = time.time()


        # ----------------------------------------------------
        # MULTIPLE PEOPLE / FACES
        # ----------------------------------------------------

        if self.faces > 1:

            self.log_event(

                "Multiple faces",

                "CRITICAL",

                30,

                f"{self.faces} faces detected."
            )


        # ----------------------------------------------------
        # FIRST FACE
        # ----------------------------------------------------

        points = result.face_landmarks[0]


        # ----------------------------------------------------
        # EYE ASPECT RATIO
        # ----------------------------------------------------

        left_eye = eye_aspect_ratio(

            points,

            [
                33,
                160,
                158,
                133,
                153,
                144
            ]
        )


        right_eye = eye_aspect_ratio(

            points,

            [
                362,
                385,
                387,
                263,
                373,
                380
            ]
        )


        eye_ratio = (
            left_eye +
            right_eye
        ) / 2


        self.eye = (
            "CLOSED"
            if eye_ratio < 0.18
            else
            "OPEN"
        )


        if self.eye == "CLOSED":

            if self.closed_start is None:

                self.closed_start = time.time()

            elif (
                time.time() -
                self.closed_start
                > 2
            ):

                self.log_event(

                    "Prolonged eye closure",

                    "MEDIUM",

                    7,

                    "Eyes closed for an extended interval."
                )

        else:

            self.closed_start = None


        # ----------------------------------------------------
        # HEAD DIRECTION
        # ----------------------------------------------------

        nose = points[1]

        left_face = points[33]

        right_face = points[263]


        middle = (
            left_face.x +
            right_face.x
        ) / 2


        dx = (
            nose.x -
            middle
        )


        if dx < -0.045:

            self.head = "LEFT"

        elif dx > 0.045:

            self.head = "RIGHT"

        else:

            self.head = "CENTER"


        if self.head != "CENTER":

            if self.look_start is None:

                self.look_start = time.time()

            elif (
                time.time() -
                self.look_start
                > 2
            ):

                self.log_event(

                    "Looking away",

                    "MEDIUM",

                    8,

                    f"Head direction {self.head}."
                )

        else:

            self.look_start = None


    # ========================================================
    # OBJECT ANALYSIS
    # ========================================================

    def analyze_objects(self, frame):
        self.persons = 0
        self.phones = 0

        if self.detector is None:
            return

        try:
            # 1) PERSON
            results = self.detector.predict(
                source=frame,
                conf=0.35,
                verbose=False,
                imgsz=640,
                classes=[0]
            )

            if results:
                names = results[0].names
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = names.get(cls, str(cls))
                    if name == 'person':
                        self.persons += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (80,180,255), 2)
                        cv2.putText(frame, f'PERSON {conf:.0%}',
                                    (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .5, (80,180,255), 2)

            # 2) PHONE
            phone_results = self.detector.predict(
                source=frame,
                conf=0.16,
                verbose=False,
                imgsz=960,
                classes=[67],
                max_det=5
            )

            best_conf = 0.0
            for result in phone_results or []:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    best_conf = max(best_conf, conf)

            now = time.time()
            if best_conf >= 0.16:
                self.phones = 1
                self.phone_hold_until = now + 1.25
                self.last_phone_conf = best_conf

                self.log_event(
                    'Mobile phone detected',
                    'CRITICAL',
                    35,
                    f'Mobile phone detected (confidence {best_conf:.0%}).'
                )
            elif now < getattr(self, 'phone_hold_until', 0):
                self.phones = 1
            else:
                self.phones = 0
                self.last_phone_conf = 0.0

        except Exception as error:
            self.message_label.setText('Object detector error: ' + str(error)[:120])
            print('\nOBJECT DETECTOR RUNTIME ERROR')
            traceback.print_exc()


    # ========================================================
    # MAIN FRAME LOOP
    # ========================================================

    def tick(self):

        if self.cap is None:

            return


        success, frame = (
            self.cap.read()
        )


        if not success:

            return


        # Mirror webcam.

        frame = cv2.flip(
            frame,
            1
        )


        self.frame = frame.copy()


        self.frame_no += 1


        # ----------------------------------------------------
        # FACE ANALYSIS
        # ----------------------------------------------------

        if (
            self.frame_no % 2 == 0
            and
            self.face is not None
        ):

            try:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                image = mp.Image(

                    image_format=(
                        mp.ImageFormat.SRGB
                    ),

                    data=rgb
                )


                timestamp = max(

                    int(
                        time.time() * 1000
                    ),

                    self.last_ts + 1
                )


                self.last_ts = timestamp


                result = (
                    self.face.detect_for_video(
                        image,
                        timestamp
                    )
                )


                self.analyze_face(
                    result
                )


                # Face landmark visualization.

                if result.face_landmarks:

                    h, w = (
                        frame.shape[:2]
                    )


                    for face in (
                        result.face_landmarks
                    ):

                        for index in [
                            1,
                            33,
                            263,
                            61,
                            291
                        ]:

                            point = face[index]


                            cv2.circle(

                                frame,

                                (
                                    int(
                                        point.x * w
                                    ),

                                    int(
                                        point.y * h
                                    )
                                ),

                                3,

                                (60,220,165),

                                -1
                            )


            except Exception as error:

                self.message_label.setText(

                    "MediaPipe runtime error: "
                    + str(error)[:120]
                )


        # ----------------------------------------------------
        # OBJECT ANALYSIS
        # ----------------------------------------------------

        if self.frame_no % 5 == 0:

            self.analyze_objects(
                frame
            )


        # ----------------------------------------------------
        # ANALYTICS
        # ----------------------------------------------------

        if self.monitoring:

            self.total_samples += 1


            if self.faces == 1:

                self.good_face_samples += 1


            if self.head == "CENTER":

                self.good_attention_samples += 1


        # ----------------------------------------------------
        # CAMERA OVERLAY
        # ----------------------------------------------------

        active_color = (
            (60,220,165)
            if self.monitoring
            else
            (190,160,70)
        )


        cv2.rectangle(

            frame,

            (15,15),

            (465,96),

            (10,16,22),

            -1
        )


        cv2.putText(

            frame,

            (
                "● MONITORING ACTIVE"
                if self.monitoring
                else
                "○ MONITORING PAUSED"
            ),

            (30,48),

            cv2.FONT_HERSHEY_SIMPLEX,

            .7,

            active_color,

            2
        )


        cv2.putText(

            frame,

            f"RISK SCORE: {self.risk}/100",

            (30,76),

            cv2.FONT_HERSHEY_SIMPLEX,

            .5,

            (220,225,230),

            1
        )


        height, width = (
            frame.shape[:2]
        )


        cv2.putText(

            frame,

            f"HEAD: {self.head}",

            (
                width - 220,
                40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            .55,

            (220,225,230),

            2
        )


        cv2.putText(

            frame,

            f"EVENTS: {self.events}",

            (
                width - 220,
                68
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            .5,

            (160,170,180),

            1
        )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        self.fc += 1

        elapsed = (
            time.time() -
            self.ft
        )


        if elapsed >= 1:

            self.fps = (
                self.fc /
                elapsed
            )

            self.fc = 0

            self.ft = time.time()


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        h, w, channels = (
            rgb.shape
        )


        image = QImage(

            rgb.data,

            w,

            h,

            channels * w,

            QImage.Format_RGB888
        )


        self.video.setPixmap(

            QPixmap.fromImage(
                image
            ).scaled(

                self.video.size(),

                Qt.KeepAspectRatio,

                Qt.SmoothTransformation
            )
        )


        # ----------------------------------------------------
        # UI VALUES
        # ----------------------------------------------------

        integrity = max(
            0,
            100 - self.risk
        )


        self.fps_label.setText(
            f"FPS {self.fps:.0f}"
        )


        self.risk_label.setText(
            f"{integrity} / 100"
        )


        self.risk_bar.setValue(
            integrity
        )


        if self.risk < 25:

            self.risk_text.setText(
                "LOW • Normal candidate state"
            )

        elif self.risk < 60:

            self.risk_text.setText(
                "MEDIUM • Review recommended"
            )

        elif self.risk < 85:

            self.risk_text.setText(
                "HIGH • Attention required"
            )

        else:

            self.risk_text.setText(
                "CRITICAL • Immediate review required"
            )


        self.face_label.setText(
            f"{self.faces}\nFACES"
        )


        self.person_label.setText(
            f"{self.persons}\nPERSONS"
        )


        self.phone_label.setText(
            f"{self.phones}\nPHONES"
        )


        self.attention_label.setText(
            f"ATTENTION • {self.head}"
        )


        self.eye_label.setText(
            f"EYES • {self.eye}"
        )


        self.event_label.setText(
            f"EVENTS • {self.events}"
        )


    # ========================================================
    # START / PAUSE
    # ========================================================

    def toggle_monitoring(self):

        self.monitoring = (
            not self.monitoring
        )


        if self.monitoring:

            self.monitor_button.setText(
                "⏸  PAUSE MONITORING"
            )


            self.status.setText(
                "● MONITORING ACTIVE"
            )


            self.status.setStyleSheet(
                "color:#39d7a3;font-weight:900;font-size:13px;"
            )


            self.alert_label.setText(
                "Monitoring active.\n"
                "AI analysis is recording important events."
            )


        else:

            self.monitor_button.setText(
                "▶  RESUME MONITORING"
            )


            self.status.setText(
                "● PAUSED"
            )


            self.status.setStyleSheet(
                "color:#d9a441;font-weight:900;font-size:13px;"
            )


            self.alert_label.setText(
                "Monitoring paused."
            )


    # ========================================================
    # EVENTS TABLE
    # ========================================================

    def refresh_events(self):

        rows = self.conn.execute(

            """
            SELECT
                timestamp,
                event,
                severity,
                risk,
                details,
                snapshot

            FROM events

            ORDER BY id DESC
            """
        ).fetchall()


        self.events_table.setRowCount(
            len(rows)
        )


        for row_index, row in enumerate(rows):

            for column_index, value in enumerate(row):

                if column_index == 5:

                    value = (
                        "📸 Saved"
                        if value
                        else
                        "—"
                    )


                self.events_table.setItem(

                    row_index,

                    column_index,

                    QTableWidgetItem(
                        str(value or "")
                    )
                )


    # ========================================================
    # EVIDENCE GALLERY
    # ========================================================

    def refresh_gallery(self):

        # Remove old cards.

        while (
            self.gallery_layout.count()
            > 1
        ):

            item = (
                self.gallery_layout.takeAt(0)
            )


            widget = item.widget()


            if widget:

                widget.deleteLater()


        rows = self.conn.execute(

            """
            SELECT
                timestamp,
                event,
                severity,
                snapshot

            FROM events

            WHERE snapshot != ''

            ORDER BY id DESC
            """
        ).fetchall()


        for (
            timestamp,
            event,
            severity,
            snapshot
        ) in rows:

            if (
                not snapshot
                or
                not os.path.exists(snapshot)
            ):

                continue


            card = self.card()

            layout = QVBoxLayout(
                card
            )
            layout.setContentsMargins(16, 16, 16, 16)


            title = QLabel(

                f"🕒 {timestamp}   |   📌 {event}   |   ⚡ SEVERITY: {severity}"
            )

            title.setObjectName(
                "small"
            )


            layout.addWidget(
                title
            )


            image = QLabel()

            image.setAlignment(
                Qt.AlignCenter
            )


            pixmap = QPixmap(
                snapshot
            )


            if not pixmap.isNull():

                image.setPixmap(

                    pixmap.scaled(

                        850,
                        450,

                        Qt.KeepAspectRatio,

                        Qt.SmoothTransformation
                    )
                )


                layout.addWidget(
                    image
                )


            self.gallery_layout.insertWidget(

                self.gallery_layout.count() - 1,

                card
            )


    # ========================================================
    # ANALYTICS
    # ========================================================

    def refresh_analytics(self):

        if self.total_samples == 0:

            return


        face_presence = (

            self.good_face_samples /
            self.total_samples *
            100
        )


        attention = (

            self.good_attention_samples /
            self.total_samples *
            100
        )


        rows = self.conn.execute(

            """
            SELECT
                event,
                COUNT(*)

            FROM events

            GROUP BY event

            ORDER BY COUNT(*) DESC
            """
        ).fetchall()


        if rows:

            breakdown = "<br>".join(

                f"• {event}: "
                f"<b>{count}</b>"

                for event, count
                in rows
            )

        else:

            breakdown = (
                "• No incidents recorded"
            )


        integrity = max(
            0,
            100 - self.risk
        )


        self.analytics_label.setText(

            f"""
            <div style="font-family:'Segoe UI'; color:#edf2f5; line-height:1.6;">
                <h1 style="color:#ffffff; font-size:24px; font-weight:900; margin-bottom:4px;">EXAM INTEGRITY ANALYTICS</h1>
                <p style="color:#82919e; font-size:13px; margin-top:0;">Comprehensive session telemetry and behavioral metrics summary.</p>
                <hr style="border:1px solid #202e3b; margin:16px 0;">
                
                <table width="100%" cellspacing="0" cellpadding="8">
                    <tr>
                        <td width="25%" style="background:#101820; border:1px solid #202e3b; border-radius:10px; padding:16px;">
                            <span style="color:#82919e; font-size:12px; font-weight:700;">CURRENT INTEGRITY</span>
                            <h2 style="color:#39d7a3; font-size:32px; font-weight:900; margin:6px 0 0 0;">{integrity}%</h2>
                        </td>
                        <td width="4"></td>
                        <td width="25%" style="background:#101820; border:1px solid #202e3b; border-radius:10px; padding:16px;">
                            <span style="color:#82919e; font-size:12px; font-weight:700;">ATTENTION SCORE</span>
                            <h2 style="color:#ffffff; font-size:32px; font-weight:900; margin:6px 0 0 0;">{attention:.1f}%</h2>
                        </td>
                        <td width="4"></td>
                        <td width="25%" style="background:#101820; border:1px solid #202e3b; border-radius:10px; padding:16px;">
                            <span style="color:#82919e; font-size:12px; font-weight:700;">FACE PRESENCE</span>
                            <h2 style="color:#ffffff; font-size:32px; font-weight:900; margin:6px 0 0 0;">{face_presence:.1f}%</h2>
                        </td>
                        <td width="4"></td>
                        <td width="25%" style="background:#101820; border:1px solid #202e3b; border-radius:10px; padding:16px;">
                            <span style="color:#82919e; font-size:12px; font-weight:700;">TOTAL INCIDENTS</span>
                            <h2 style="color:#ffb84d; font-size:32px; font-weight:900; margin:6px 0 0 0;">{self.events}</h2>
                        </td>
                    </tr>
                </table>

                <br>
                <h3 style="color:#ffffff; font-size:16px; font-weight:800;">Incident Breakdown</h3>
                <div style="background:#101820; border:1px solid #202e3b; border-radius:10px; padding:16px; font-size:14px;">
                    {breakdown}
                </div>

                <br>
                <p style="color:#82919e; font-size:12px;">
                💡 Evidence snapshots are safely stored locally in the <b>evidence</b> directory.
                </p>
            </div>
            """
        )


    # ========================================================
    # EXPORT REPORT
    # ========================================================

    def export_report(self):

        path = os.path.join(

            REPORTS,

            f"exam_report_"
            f"{datetime.now():%Y%m%d_%H%M%S}.csv"
        )


        rows = self.conn.execute(

            """
            SELECT
                timestamp,
                event,
                severity,
                risk,
                details,
                snapshot

            FROM events

            ORDER BY id
            """
        ).fetchall()


        with open(

            path,

            "w",

            newline="",

            encoding="utf-8"

        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow(

                [
                    "Timestamp",
                    "Event",
                    "Severity",
                    "Risk",
                    "Details",
                    "Snapshot"
                ]
            )


            writer.writerows(
                rows
            )


        QMessageBox.information(

            self,

            "Exam Report Exported",

            f"Report created successfully:\n\n{path}"
        )


        try:

            os.startfile(
                REPORTS
            )

        except Exception:

            pass


    # ========================================================
    # KEYBOARD
    # ========================================================

    def keyPressEvent(
        self,
        event
    ):

        if event.key() == Qt.Key_Escape:

            self.close()

        else:

            super().keyPressEvent(
                event
            )


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        self.running = False


        if hasattr(
            self,
            "timer"
        ):

            self.timer.stop()


        if self.cap:

            self.cap.release()


        if self.face:

            try:

                self.face.close()

            except Exception:

                pass


        try:

            self.conn.close()

        except Exception:

            pass


        event.accept()


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = App()

    window.show()

    sys.exit(
        app.exec()
    )