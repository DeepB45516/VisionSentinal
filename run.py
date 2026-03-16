import subprocess
import webbrowser
import time

print("Starting Sentinel AI...")

# start backend
backend = subprocess.Popen(
    ["python", "backend/server.py"]
)

# start frontend
frontend = subprocess.Popen(
    ["cmd", "/c", "npm run dev"],
    cwd="web-app"
)

# wait for servers to start
time.sleep(5)

# open browser
webbrowser.open("http://localhost:5173")

backend.wait()
frontend.wait()