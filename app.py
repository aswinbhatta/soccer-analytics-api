import os
import cv2
import uuid
import math
import numpy as np
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans

# Render / Production Fixes
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

app = FastAPI(title="AI Soccer Match Analyzer",
              description="Player tracking and performance analysis using YOLOv8")

MODEL_PATH = "yolov8x.pt"

# Load model
model = YOLO(MODEL_PATH)

class TeamAssigner:
    def __init__(self):
        self.kmeans = None

    def get_dominant_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return (0,0,0)
        h = crop.shape[0]
        crop = crop[:h//2, :, :]
        pixels = crop.reshape(-1, 3)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = ~((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85))
        filtered = pixels[mask.flatten()]
        if len(filtered) < 25: return (0,0,0)
        kmeans = KMeans(n_clusters=1, n_init=5).fit(filtered)
        return kmeans.cluster_centers_[0]

    def assign_teams(self, player_colors):
        if len(player_colors) < 15: return
        colors = np.array(list(player_colors.values()))
        self.kmeans = KMeans(n_clusters=2, n_init=10).fit(colors)

    def get_player_team(self, color):
        if self.kmeans is None or color is None: return 0
        return self.kmeans.predict([color])[0]


# [Add your best MatchAnalytics class and process_video_pipeline here]
# (Use the improved version we had earlier)

@app.post("/analyze")
async def analyze_video(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Only MP4 files supported")
    
    run_id = str(uuid.uuid4())[:8]
    temp_path = f"temp_{run_id}.mp4"
    
    try:
        with open(temp_path, "wb") as f:
            f.write(await video.read())
        
        report = process_video_pipeline(temp_path, run_id)
        return {"status": "success", "run_id": run_id, "data": report}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
