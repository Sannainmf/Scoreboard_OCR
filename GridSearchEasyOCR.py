import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import easyocr
import cv2
import csv
from datetime import datetime
import itertools

def simple_zoom_crop(frame, crop_x, crop_y, crop_width, crop_height, scale_factor=4):
    cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    new_width = crop_width * scale_factor
    new_height = crop_height * scale_factor
    zoomed = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return zoomed

# Coordinates (same as before)
HOME_X1, HOME_Y1, HOME_X2, HOME_Y2 = 925, 221, 975, 268
AWAY_X1, AWAY_Y1, AWAY_X2, AWAY_Y2 = 1050, 210, 1090, 250

SCORE_REGIONS = {
    'home': {'x': HOME_X1, 'y': HOME_Y1, 'width': HOME_X2 - HOME_X1, 'height': HOME_Y2 - HOME_Y1},
    'away': {'x': AWAY_X1, 'y': AWAY_Y1, 'width': AWAY_X2 - AWAY_X1, 'height': AWAY_Y2 - AWAY_Y1}
}

SCALE_FACTOR = 4
INTERVAL_SECONDS = 1

# Parameter Grid
MAG_RATIOS = [1, 2, 3, 4]
CANVAS_SIZES = [1024, 2048, 4096]
CONTRAST_THRESHOLDS = [0.1, 0.2, 0.3]
ADJUST_CONTRASTS = [0.5, 0.7, 0.9]

# Get video
videos_dir = Path('videos')
video_files = list(videos_dir.glob('*.mp4')) 

if not video_files:
    print("No video found")
    exit()

video_path = video_files[0]

# Setup results directory
results_dir = Path('results/parameter_grid')
results_dir.mkdir(exist_ok=True, parents=True)
debug_dir = results_dir / 'debug'
debug_dir.mkdir(exist_ok=True)

# Prepare parameter grid
parameter_combinations = list(itertools.product(
    MAG_RATIOS, 
    CANVAS_SIZES, 
    CONTRAST_THRESHOLDS, 
    ADJUST_CONTRASTS
))

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Prepare CSV for parameter grid results
grid_results_filename = results_dir / f"parameter_grid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(grid_results_filename, 'w', newline='') as f:
    csv.writer(f).writerow([
        'Mag Ratio', 'Canvas Size', 'Contrast Threshold', 'Adjust Contrast', 
        'Home Detections', 'Home Confidence', 'Away Detections', 'Away Confidence'
    ])

# Open video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = int(fps * INTERVAL_SECONDS)

# Extract frames (same as before)
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
    
    frame_count += 1
cap.release()

# Parameter Grid Search
for mag_ratio, canvas_size, contrast_ths, adjust_contrast in parameter_combinations:
    print(f"\nTesting Parameters:")
    print(f"Mag Ratio: {mag_ratio}, Canvas Size: {canvas_size}, " +
          f"Contrast Threshold: {contrast_ths}, Adjust Contrast: {adjust_contrast}")
    
    # Tracking results for this parameter set
    home_detections = []
    away_detections = []
    home_confidences = []
    away_confidences = []
    
    for frame_num, timestamp, frame in frames_data:
        frame_scores = {}
        
        for label, coords in SCORE_REGIONS.items():
            cropped = simple_zoom_crop(frame, coords['x'], coords['y'], coords['width'], coords['height'], SCALE_FACTOR)
            frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
            try:
                result = reader.readtext(
                    frame_rgb, 
                    allowlist='0123456789', 
                    decoder='greedy', 
                    mag_ratio=mag_ratio,
                    canvas_size=canvas_size,
                    contrast_ths=contrast_ths, 
                    adjust_contrast=adjust_contrast
                )
                
                # Filter results
                result = [r for r in result if r[2] >= 0.75 and len(r[1]) <= 2 and r[1].strip()]
                
                if result:
                    best = max(result, key=lambda x: x[2])
                    if label == 'home':
                        home_detections.append(best[1])
                        home_confidences.append(best[2])
                    else:
                        away_detections.append(best[1])
                        away_confidences.append(best[2])
            
            except Exception as e:
                print(f"Error processing {label} region: {e}")
    
    # Calculate statistics
    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    def safe_mode(lst):
        if not lst:
            return 'NO_DETECTION'
        return max(set(lst), key=lst.count)
    
    # Write results to CSV
    with open(grid_results_filename, 'a', newline='') as f:
        csv.writer(f).writerow([
            mag_ratio, canvas_size, contrast_ths, adjust_contrast,
            safe_mode(home_detections), 
            safe_avg(home_confidences),
            safe_mode(away_detections), 
            safe_avg(away_confidences)
        ])

print(f"\n✓ Done: {grid_results_filename}")
print("Analyze the results to find the best parameter combination!")