import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import cv2
import csv
from datetime import datetime
import torch
from PIL import Image

# IDEFICS2 imports
from transformers import AutoProcessor, AutoModelForVision2Seq


# -------------------------
#   SIMPLE ZOOM CROP
# -------------------------
def simple_zoom_crop(frame, crop_x, crop_y, crop_width, crop_height, scale_factor=4):
    cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    new_width = crop_width * scale_factor
    new_height = crop_height * scale_factor
    zoomed = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return zoomed


# -------------------------
#   VISUALIZATION
# -------------------------
def visualize_regions(frame, regions, output_path):
    vis_frame = frame.copy()
    for label, coords in regions.items():
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        color = (0, 255, 0) if label == 'home' else (0, 0, 255)
        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis_frame, label.upper(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(output_path), vis_frame)


# -------------------------
#   OCR FUNCTION (IDEFICS2)
# ------------------------- 
def ocr_with_idefics2(cropped_frame, model, processor):
    """Use Idefics2 to read the score from cropped frame"""

    # Convert BGR → RGB → PIL
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

    # Chat-style prompt for Idefics2
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What number is shown in this LED display? Return only digit(s)."}
            ]
        }
    ]

    # Build prompt
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=[img],
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)

    # Decode
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extract digits only
    import re
    numbers = re.findall(r"\d+", output_text)
    return numbers[-1] if numbers else None



# -------------------------
#   SCORE REGIONS
# -------------------------
HOME_X1, HOME_Y1, HOME_X2, HOME_Y2 = 925, 221, 975, 268
AWAY_X1, AWAY_Y1, AWAY_X2, AWAY_Y2 = 1050, 210, 1090, 250

SCORE_REGIONS = {
    'home': {'x': HOME_X1, 'y': HOME_Y1, 'width': HOME_X2 - HOME_X1, 'height': HOME_Y2 - HOME_Y1},
    'away': {'x': AWAY_X1, 'y': AWAY_Y1, 'width': AWAY_X2 - AWAY_X1, 'height': AWAY_Y2 - AWAY_Y1}
}

SCALE_FACTOR = 4
INTERVAL_SECONDS = 1


# -------------------------
#   VIDEO DISCOVERY
# -------------------------
videos_dir = Path('videos')
video_files = (
    list(videos_dir.glob('*.mp4')) +
    list(videos_dir.glob('*.avi')) +
    list(videos_dir.glob('*.mov')) +
    list(videos_dir.glob('*.mkv'))
)

if not video_files:
    print("No video found")
    exit()

video_path = video_files[0]

# Directories
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)
debug_dir = results_dir / 'debug'
debug_dir.mkdir(exist_ok=True)

csv_filename = results_dir / f"scores_idefics2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# -------------------------
#   STEP 1 — EXTRACT FRAMES
# -------------------------
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = int(fps * INTERVAL_SECONDS)

print(f"Video: {video_path.name}")
print(f"FPS: {fps:.1f}, Total frames: {total_frames}")
print(f"Extracting 1 frame every {INTERVAL_SECONDS}s ({frame_interval} frames)")
print(f"Estimated frames to process: {total_frames // frame_interval}\n")

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

        # Save debug info for frame 0
        if frame_num == 0:
            cv2.imwrite(str(debug_dir / 'full_frame.png'), frame)
            visualize_regions(frame, SCORE_REGIONS, debug_dir / 'regions_visualization.png')

    frame_count += 1

cap.release()
print(f"Extracted {len(frames_data)} frames\n")


# -------------------------
#   STEP 2 — LOAD IDEFICS2
# -------------------------
print("Step 2: Loading Idefics2 8B model...")

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Model ready\n")


# -------------------------
#   STEP 3 — RUN OCR
# -------------------------
print("Step 3: Running OCR on all frames...")

# CSV header
with open(csv_filename, 'w', newline='') as f:
    csv.writer(f).writerow(['Frame', 'Time', 'Home', 'Away'])

last_home = None
last_away = None

for frame_num, timestamp, frame in frames_data:
    frame_scores = {}

    for label, coords in SCORE_REGIONS.items():
        cropped = simple_zoom_crop(frame, coords['x'], coords['y'], coords['width'], coords['height'], SCALE_FACTOR)

        if frame_num == 0:
            cv2.imwrite(str(debug_dir / f'{label}_crop.png'), cropped)

        # OCR using Idefics2
        detected = ocr_with_idefics2(cropped, model, processor)

        if detected:
            frame_scores[label] = detected
            if label == 'home':
                last_home = detected
            else:
                last_away = detected
        else:
            frame_scores[label] = 'NO DETECTION'

    print(f"Frame {frame_num} @ {timestamp:.1f}s: HOME={last_home or '-'} AWAY={last_away or '-'}")

    with open(csv_filename, 'a', newline='') as f:
        csv.writer(f).writerow([
            frame_num,
            f"{timestamp:.1f}",
            frame_scores.get('home', 'NO DETECTION'),
            frame_scores.get('away', 'NO DETECTION')
        ])

print(f"\nDone: {csv_filename}")
print("Debug images saved in: results/debug/")
