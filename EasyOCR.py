import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import easyocr
import cv2
import csv
from datetime import datetime

def simple_zoom_crop(frame, crop_x, crop_y, crop_width, crop_height, scale_factor=4):
    cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    new_width = crop_width * scale_factor
    new_height = crop_height * scale_factor
    zoomed = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return zoomed

def visualize_regions(frame, regions, output_path):
    vis_frame = frame.copy()
    for label, coords in regions.items():
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        color = (0, 255, 0) if label == 'home' else (0, 0, 255)
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis_frame, label.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(output_path), vis_frame)

# Coordinates
HOME_X1, HOME_Y1, HOME_X2, HOME_Y2 = 930, 225, 970, 265
AWAY_X1, AWAY_Y1, AWAY_X2, AWAY_Y2 = 1050, 210, 1090, 250

SCORE_REGIONS = {
    'home': {'x': HOME_X1, 'y': HOME_Y1, 'width': HOME_X2 - HOME_X1, 'height': HOME_Y2 - HOME_Y1},
    'away': {'x': AWAY_X1, 'y': AWAY_Y1, 'width': AWAY_X2 - AWAY_X1, 'height': AWAY_Y2 - AWAY_Y1}
}

SCALE_FACTOR = 4
INTERVAL_SECONDS = 1

# Get video
videos_dir = Path('videos')
video_files = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.avi')) + \
              list(videos_dir.glob('*.mov')) + list(videos_dir.glob('*.mkv'))

if not video_files:
    print("No video found")
    exit()

video_path = video_files[0]

# Setup
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)
debug_dir = results_dir / 'debug'
debug_dir.mkdir(exist_ok=True)

csv_filename = results_dir / f"scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Open video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = int(fps * INTERVAL_SECONDS)

print(f"Video: {video_path.name}")
print(f"FPS: {fps:.1f}, Total frames: {total_frames}")
print(f"Extracting 1 frame every {INTERVAL_SECONDS}s ({frame_interval} frames)")
print(f"Estimated frames to process: {total_frames // frame_interval}\n")

# STEP 1: Extract all frames first (FAST)
print("Step 1: Extracting frames...")
frames_data = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        frame_num = frame_count // frame_interval
        timestamp = frame_count / fps
        frames_data.append((frame_num, timestamp, frame.copy()))
        
        if frame_num == 0:
            cv2.imwrite(str(debug_dir / 'full_frame.png'), frame)
            visualize_regions(frame, SCORE_REGIONS, debug_dir / 'regions_visualization.png')
    
    frame_count += 1

cap.release()
print(f"✓ Extracted {len(frames_data)} frames\n")

# STEP 2: Initialize OCR (only once)
print("Step 2: Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=True)
print("✓ OCR ready\n")

# STEP 3: Process all frames with OCR
print("Step 3: Running OCR on all frames...")

with open(csv_filename, 'w', newline='') as f:
    csv.writer(f).writerow(['Frame', 'Time', 'Home', 'Home Conf', 'Away', 'Away Conf'])

last_home = None
last_away = None

for frame_num, timestamp, frame in frames_data:
    frame_scores = {}
    
    for label, coords in SCORE_REGIONS.items():
        cropped = simple_zoom_crop(frame, coords['x'], coords['y'], coords['width'], coords['height'], SCALE_FACTOR)
        
        if frame_num == 0:
            cv2.imwrite(str(debug_dir / f'{label}_crop.png'), cropped)
        
        frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        result = reader.readtext(frame_rgb, allowlist='0123456789')
        result = [r for r in result if r[2] >= 0.75 and len(r[1]) <= 2 and r[1].strip()]
        
        if result:
            best = max(result, key=lambda x: x[2])
            frame_scores[label] = {'detected': best[1], 'confidence': best[2]}
            if label == 'home':
                last_home = best[1]
            else:
                last_away = best[1]
        else:
            frame_scores[label] = {'detected': 'NO DETECTION', 'confidence': 0.0}
    
    print(f"Frame {frame_num} @ {timestamp:.1f}s: HOME={last_home or '-'} AWAY={last_away or '-'}")
    
    with open(csv_filename, 'a', newline='') as f:
        csv.writer(f).writerow([
            frame_num, f"{timestamp:.1f}",
            frame_scores['home']['detected'],
            f"{frame_scores['home']['confidence']:.3f}" if frame_scores['home']['confidence'] > 0 else "",
            frame_scores['away']['detected'],
            f"{frame_scores['away']['confidence']:.3f}" if frame_scores['away']['confidence'] > 0 else ""
        ])

print(f"\n✓ Done: {csv_filename}")
print(f"✓ Debug: results/debug/regions_visualization.png")