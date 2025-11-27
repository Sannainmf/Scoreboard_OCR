import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import cv2
import csv
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

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

def ocr_with_qwen(cropped_frame, model, processor):
    """Use Qwen2-VL to read the score from cropped frame"""
    # Convert BGR to RGB and create PIL Image
    img_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    
    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What number is shown in this LED display? Return only the digit(s), nothing else."}
            ]
        }
    ]
    
    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=10)
    
    # Decode
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Extract just the number from response
    # The model might return something like "The number shown is 10" so we extract digits
    import re
    numbers = re.findall(r'\d+', output_text)
    if numbers:
        return numbers[-1]  # Return last number found
    return None

# Coordinates
HOME_X1, HOME_Y1, HOME_X2, HOME_Y2 = 925, 221, 975, 268
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

csv_filename = results_dir / f"scores_qwen2vl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

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
print(f"Extracted {len(frames_data)} frames\n")

# STEP 2: Load Qwen2-VL model (only once)
print("Step 2: Loading Qwen2-VL-2B model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,  # Use half precision for faster inference
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("Model ready\n")

# STEP 3: Process all frames with Qwen2-VL
print("Step 3: Running OCR on all frames...")

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
        
        # Use Qwen2-VL for OCR
        detected = ocr_with_qwen(cropped, model, processor)
        
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
print(f"Debug: results/debug/regions_visualization.png")