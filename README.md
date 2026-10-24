# AI Exam Proctor Vision — FIXED

This version fixes the earlier silent-analysis behavior.

It shows MediaPipe/YOLO readiness and runtime errors on the dashboard instead of silently ignoring them.

Features:
- Face and multiple-face detection
- Head direction / looking-away detection
- Eye-closure detection
- Person detection
- Cell-phone detection
- Risk score and event history
- Evidence snapshots
- SQLite event log
- CSV report export
- Professional PySide6 UI
- Local processing
- No Ollama required

Setup:
1. Install Python 3.13.
2. Run setup.bat.
3. Run run.bat.
4. The MediaPipe face model downloads automatically if missing.
5. YOLO downloads yolo11n.pt on first use.

Use this as decision-support/demo software. Detection events are signals for human review, not proof of misconduct.
