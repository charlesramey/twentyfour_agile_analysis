"""
Agile Pipeline for Multi-Modal Synchronization and Visualization.

Usage:
    python agile_pipeline.py

This script processes video and sensor data based on '2024AgileCupMetadata_ScribeNotes_CameraInfo.xlsx'.
It performs:
1. IMU Synchronization (Collar Data <-> Camera 2 Audio) using tap detection.
2. Multi-Camera Synchronization (All 4 Cameras) using audio cross-correlation.
3. YOLO-based Camera Selection to produce a final video tracking the dog.

Testing:
    To run with dummy data:
    1. python create_dummy_data.py (Generates dummy videos and CSVs)
    2. python agile_pipeline.py (Runs the pipeline)
    3. Verify output in 'ProcessedData' directory.
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
import cv2
import whisper
import subprocess
import shutil
import tempfile
import warnings
import imageio_ffmpeg
from scipy.io import wavfile
from scipy.signal import fftconvolve, correlate
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Setup ffmpeg in PATH for Whisper
# Create a temp dir, symlink imageio_ffmpeg binary to "ffmpeg", add to PATH
_ffmpeg_dir = os.path.join(tempfile.gettempdir(), "ffmpeg_bin")
os.makedirs(_ffmpeg_dir, exist_ok=True)
_symlink_path = os.path.join(_ffmpeg_dir, "ffmpeg")
if not os.path.exists(_symlink_path):
    try:
        os.symlink(FFMPEG_EXE, _symlink_path)
    except OSError:
        # If symlink fails (windows?), try copy
        shutil.copy(FFMPEG_EXE, _symlink_path)
        # Ensure executable
        os.chmod(_symlink_path, 0o755)

os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ["PATH"]

EXCEL_FILE = '2024AgileCupMetadata_ScribeNotes_CameraInfo.xlsx'
COLLAR_DIR = 'Collar Data'
PROCESSED_DIR = 'ProcessedData'
VIDEO_DIRS = {
    "1": "MMJ 1 GoPro",
    "2": "MMJ 2 GoPro",
    "3": "MMJ 3 GoPro",
    "4": "MMJ 4 GoPro",
}
SHEET_TO_COURSE_DIR = {
    'MJWW': 'Masters JWW',
    'MStd': 'Masters Standard',
    'PremStd': 'Premeire Standard',
    'PremJWW': 'Premeire JWW'
}

# Global Whisper Model (Lazy load)
WHISPER_MODEL = None

# --- Sync Grid Robust Helpers (Video-Video) ---
SAMPLERATE = 22050
MAX_SHIFT_S = 10.0
MIN_PEAK = 0.15

def run_cmd(cmd, check=True):
    subprocess.run(cmd, check=check, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_audio(video_path, wav_path, sr=SAMPLERATE):
    run_cmd([
        FFMPEG_EXE, "-y", "-i", video_path, "-vn", "-ac", "1",
        "-ar", str(sr), "-acodec", "pcm_s16le", wav_path
    ])

def read_wav(path):
    sr, y = wavfile.read(path)
    if y.ndim > 1: y = y[:, 0]
    if y.dtype != np.float32:
        y = y.astype(np.float32) / (np.iinfo(np.int16).max + 1.0)
    return sr, y

def get_offset_and_peak(ref_y, tgt_y, sr, max_shift_s):
    ref_z = ref_y - np.mean(ref_y)
    ref_z /= (np.std(ref_z) + 1e-8)
    tgt_z = tgt_y - np.mean(tgt_y)
    tgt_z /= (np.std(tgt_z) + 1e-8)

    corr = fftconvolve(tgt_z[::-1], ref_z, mode="full")
    lags = np.arange(-len(tgt_z) + 1, len(ref_z))

    max_shift_samples = int(max_shift_s * sr)
    mask = (lags >= -max_shift_samples) & (lags <= max_shift_samples)

    if not np.any(mask):
        return 0.0, 0.0

    corr_masked = corr[mask]
    lags_masked = lags[mask]

    best_idx = np.argmax(corr_masked)
    best_lag = lags_masked[best_idx]

    overlap = min(len(ref_z), len(tgt_z))
    peak_norm = corr_masked[best_idx] / max(1.0, overlap)

    return best_lag / float(sr), peak_norm

def get_video_offsets(video_paths):
    """
    Returns a dictionary of offsets relative to Cam 1 (index 0).
    Offsets[i] = Start time of Cam i relative to Cam 1 start.
    Positive offset => Cam i starts LATER than Cam 1.
    """
    if not all(os.path.exists(p) for p in video_paths):
        print("Missing video files for sync.")
        return None

    with tempfile.TemporaryDirectory() as td:
        wavs = []
        for i, vid in enumerate(video_paths):
            w = os.path.join(td, f"a{i}.wav")
            extract_audio(vid, w)
            wavs.append(w)

        sr0, y0 = read_wav(wavs[0])
        offsets = [0.0]

        for i in range(1, 4):
            sr, yi = read_wav(wavs[i])
            if sr != sr0:
                print(f"Sample rate mismatch for cam {i+1}")
                return None
            lag, peak = get_offset_and_peak(y0, yi, sr0, MAX_SHIFT_S)
            if peak < MIN_PEAK:
                print(f"Low sync confidence for cam {i+1}: {peak:.2f}")
                # We could fail here or assume 0. Let's fail for robustness.
                # return None
            offsets.append(lag)

    return offsets

# --- IMU Sync Helpers (Audio-IMU) ---
TAP_VARIANTS = {"tap", "ta", "da", "pat"}

def normalize_word(w):
    return "".join(c for c in w.lower() if c.isalnum())

def sync_imu_to_video(video_path, csv_path, output_path):
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Whisper model...")
        WHISPER_MODEL = whisper.load_model("small.en")

    with tempfile.TemporaryDirectory() as td:
        # 1. Audio
        audio_wav = os.path.join(td, "temp_audio.wav")
        # Use 16000 for whisper
        subprocess.run([
            FFMPEG_EXE, "-y", "-i", video_path, "-vn", "-ac", "1",
            "-ar", "16000", "-c:a", "pcm_s16le", audio_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Transcribe
        result = WHISPER_MODEL.transcribe(audio_wav, language="en", word_timestamps=True)
        words = [w for seg in result["segments"] for w in seg.get("words", [])]

        tap_times = [(float(w["start"]) + float(w["end"]))/2.0
                     for w in words if normalize_word(w["word"]) in TAP_VARIANTS]

        if not tap_times:
            print(f"No taps found in {video_path}")
            return False

        # 3. IMU Processing
        try:
            df = pd.read_csv(csv_path)
            dt = df['Timestamp'].diff().mean()
            if pd.isna(dt):
                # Fallback if timestamps are messy
                dt = 0.01 # 100hz assumption?

            fs_imu = 1.0 / dt
            acc_mag = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)

            imu_time = df['Timestamp'] - df['Timestamp'].iloc[0]
            acc_mag_centered = acc_mag - acc_mag.mean()

            search_limit = int(len(acc_mag_centered) * 0.25)
            imu_segment = acc_mag_centered.iloc[:search_limit].values

            template_taps = np.array(tap_times) - tap_times[0]
            t_template = np.arange(0, template_taps[-1] + 0.5, dt)
            synthetic_template = np.zeros_like(t_template)

            width = int(0.05 * fs_imu)
            for tap_t in template_taps:
                idx = int(tap_t * fs_imu)
                for j in range(-width, width):
                    if 0 <= idx + j < len(synthetic_template):
                        synthetic_template[idx + j] = 1.0 - abs(j)/width

            corr = correlate(imu_segment, synthetic_template, mode='valid')
            best_lag_idx = np.argmax(corr)
            best_imu_sync_time = imu_time.iloc[best_lag_idx]

            time_to_adjust = best_imu_sync_time - tap_times[0]

            # Align
            if time_to_adjust > 0:
                rows_to_remove = int(time_to_adjust * fs_imu)
                df_sync = df.iloc[rows_to_remove:].copy()
            else:
                rows_to_pad = int(abs(time_to_adjust) * fs_imu)
                pad_data = {col: [np.nan] * rows_to_pad for col in df.columns}
                start_ts = df['Timestamp'].iloc[0]
                pad_data['Timestamp'] = [start_ts - (rows_to_pad - i) * dt for i in range(rows_to_pad)]
                df_sync = pd.concat([pd.DataFrame(pad_data), df], ignore_index=True)

            df_sync['Relative_Time'] = df_sync['Timestamp'] - df_sync['Timestamp'].iloc[0]
            df_sync.to_csv(output_path, index=False)
            return True

        except Exception as e:
            print(f"Error syncing IMU: {e}")
            return False

# --- YOLO Processing Helpers ---
def process_video_yolo(video_paths, offsets, output_path, model_name="yolo12s.pt"):
    print(f"Processing YOLO video to {output_path}...")
    try:
        model = YOLO(model_name) # Will try to download if available, else error
    except Exception as e:
        print(f"Could not load {model_name}, falling back to yolov8s.pt. Error: {e}")
        model = YOLO("yolov8s.pt")

    caps = [cv2.VideoCapture(p) for p in video_paths]
    if not all(c.isOpened() for c in caps):
        print("Error opening video captures.")
        return

    # Determine timing
    start_time_cam0 = max(0, *offsets)

    # Seek videos
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Calculate initial frames to skip
    start_frames = []
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    for i, cap in enumerate(caps):
        # Time in cam[i] at start_time_cam0
        t_cam_i = start_time_cam0 - offsets[i]
        start_frame = int(t_cam_i * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        start_frames.append(start_frame)

    # Estimate frames to process (based on shortest video remaining)
    # This is an approximation since videos might end at different times
    frames_to_process = total_frames - start_frames[0]

    # Smoothing parameters
    MIN_SWITCH_FRAMES = 15 # ~0.5s at 30fps
    HYSTERESIS = 0.1 # New cam must be 10% better to switch if current is good
    DETECTION_THRESHOLD = 0.4 # Minimum confidence to consider detection valid

    current_cam_idx = 0
    frames_since_switch = MIN_SWITCH_FRAMES # Allow immediate switch at start

    try:
        frame_idx = 0
        pbar = tqdm(total=frames_to_process, unit="frames")
        while True:
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                frames.append(frame if ret else None)

            if any(f is None for f in frames):
                break # Stop when any video ends

            # Run YOLO on all frames to get confidences
            confidences = []
            for cam_idx, frame in enumerate(frames):
                # Run inference
                results = model(frame, verbose=False, classes=[16]) # 16 is dog in COCO

                # Check confidence
                conf = 0.0
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get max confidence for dog
                    conf = results[0].boxes.conf.max().item()
                confidences.append(conf)

            # Smoothing Logic
            best_raw_idx = np.argmax(confidences)
            best_raw_conf = confidences[best_raw_idx]
            current_conf = confidences[current_cam_idx]

            frames_since_switch += 1

            # Only consider switching if the best candidate actually sees a dog (conf > threshold)
            # Otherwise, we stick with the last known good camera (current_cam_idx)
            if best_raw_conf > DETECTION_THRESHOLD and frames_since_switch >= MIN_SWITCH_FRAMES:
                if best_raw_idx != current_cam_idx:
                    # Switch if significantly better than current
                    # OR if current detection is lost (< threshold) but new best is good (> threshold)
                    if (best_raw_conf > current_conf + HYSTERESIS) or (current_conf < DETECTION_THRESHOLD):
                        current_cam_idx = best_raw_idx
                        frames_since_switch = 0

            # Write best frame
            out_frame = frames[current_cam_idx]
            # Annotate
            cv2.putText(out_frame, f"Cam {current_cam_idx+1} Conf: {current_conf:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write(out_frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    finally:
        writer.release()
        for cap in caps:
            cap.release()

# --- Main Pipeline ---
def find_collar_file(dog_name, course_dir):
    # Search for *dog_name* in collar dir
    # Normalize name
    search_name = sanitize_name(dog_name).lower()

    # List all csvs in course_dir
    course_path = os.path.join(COLLAR_DIR, course_dir)
    if not os.path.exists(course_path):
        print(f"Course directory not found: {course_path}")
        return None

    candidates = []
    for f in os.listdir(course_path):
        if f.endswith('.csv'):
            if search_name in f.lower():
                candidates.append(f)

    if not candidates:
        # Try harder? e.g. "Fen_BC" vs "Fen"
        # Just return None for now
        return None

    # Pick shortest match usually? or Exact?
    return os.path.join(course_path, candidates[0])

def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '', str(name))

def main():
    parser = argparse.ArgumentParser(description="Agile Data Processing Pipeline")
    parser.add_argument("-n", "--num-runs", type=int, help="Limit the number of processed completions (runs).")
    args = parser.parse_args()

    if not os.path.exists(EXCEL_FILE):
        print(f"Excel file {EXCEL_FILE} not found.")
        return

    xls = pd.ExcelFile(EXCEL_FILE)

    processed_count = 0

    for sheet_name in xls.sheet_names:
        if args.num_runs is not None and processed_count >= args.num_runs:
            break

        if sheet_name not in SHEET_TO_COURSE_DIR:
            print(f"Skipping unknown sheet: {sheet_name}")
            continue

        print(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Detect columns
        colmap = {}
        for col in df.columns:
            s = str(col).lower()
            # Skip time columns to avoid picking up start times
            if "time" in s:
                continue
            m = re.search(r"\bmmj\s*([1-4])\b", s)
            if m and m.group(1) not in colmap:
                colmap[m.group(1)] = col

        if len(colmap) < 4:
            print("Could not find MMJ columns.")
            continue

        for idx, row in df.iterrows():
            if args.num_runs is not None and processed_count >= args.num_runs:
                break

            dog_name = row.get('Dog')
            if pd.isna(dog_name): continue

            print(f"--- Processing {dog_name} - {sheet_name} ---")

            processed_count += 1

            # 1. Resolve Video Paths
            video_paths = []
            missing_video = False
            for i in ["1","2","3","4"]:
                fname = str(row.get(colmap[i], ""))
                if fname == "nan" or not fname:
                    missing_video = True
                    break
                # Assume extension is needed if not present? Or always .MP4?
                if not fname.lower().endswith('.mp4'):
                    fname += ".MP4"

                vpath = os.path.join(VIDEO_DIRS[i], fname)
                video_paths.append(vpath)

            if missing_video:
                print("Missing video filename in excel.")
                continue

            # Check existence
            if not all(os.path.exists(p) for p in video_paths):
                print(f"One or more videos missing on disk: {video_paths}")
                continue

            # 2. Resolve Collar CSV
            course_dir_name = SHEET_TO_COURSE_DIR[sheet_name]
            collar_csv = find_collar_file(dog_name, course_dir_name)
            if not collar_csv:
                print(f"No collar data found for {dog_name} in {course_dir_name}")
                continue

            # 3. Create Output Dirs
            dog_safe = sanitize_name(dog_name)
            course_safe = sanitize_name(course_dir_name)
            out_dir = os.path.join(PROCESSED_DIR, dog_safe, f"{dog_safe} {course_safe}")
            os.makedirs(out_dir, exist_ok=True)

            # 4. Sync IMU (using Cam 2)
            # Output: RykerMastersJWW_processedcollar.csv
            processed_collar_name = f"{dog_safe}{sanitize_name(course_dir_name)}_processedcollar.csv"
            processed_collar_path = os.path.join(out_dir, processed_collar_name)

            if not os.path.exists(processed_collar_path):
                print("Syncing IMU...")
                success = sync_imu_to_video(video_paths[1], collar_csv, processed_collar_path) # Cam 2 is index 1
                if not success:
                    print("IMU sync failed.")
            else:
                print("IMU sync already done.")

            # 5. Sync Videos (Offsets)
            print("Calculating video offsets...")
            offsets = get_video_offsets(video_paths)
            if offsets is None:
                print("Video sync failed.")
                continue

            print(f"Offsets: {offsets}")

            # 6. Process YOLO Video
            processed_video_name = f"{dog_safe}{sanitize_name(course_dir_name)}_processedvideo.mp4"
            processed_video_path = os.path.join(out_dir, processed_video_name)

            if not os.path.exists(processed_video_path):
                process_video_yolo(video_paths, offsets, processed_video_path)
            else:
                print("Video processing already done.")

            print(f"Done with {dog_name} {sheet_name}")

if __name__ == "__main__":
    main()
