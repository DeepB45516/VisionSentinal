================================================================
  SENTINEL AI — Complete Project
================================================================

STRUCTURE:
  sentinel-ai-final/
  ├── main.py                  ← RUN THIS (Python desktop app)
  ├── requirements.txt
  ├── models/                  ← AI model files (auto-included)
  ├── modules/                 ← 7 detection modules
  │   ├── face_intelligence.py
  │   ├── object_detection.py
  │   ├── pose_tracking.py
  │   ├── wrist_velocity.py
  │   ├── coordination_detector.py
  │   ├── risk_scoring.py
  │   └── temporal_filter.py
  └── web-app/                 ← React browser version (optional)
      ├── src/App.jsx
      ├── package.json
      ├── index.html
      └── vite.config.js

================================================================
HOW TO RUN (Python — main project):
================================================================
  1. Install Python 3.10+ from python.org
  2. Open terminal in this folder
  3. pip install mediapipe opencv-python numpy
  4. python main.py
  5. Press 'q' to quit, 'r' for report, 's' for screenshot

================================================================
HOW TO RUN (Web app — optional browser version):
================================================================
  1. Install Node.js from nodejs.org
  2. cd web-app
  3. npm install
  4. npm run dev
  5. Open http://localhost:5173
  6. Login: admin / admin123

================================================================
