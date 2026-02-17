import os
import whisper
import warnings
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, windows, correlate

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# --- Configuration ---
BASE_DIR = "/Users/charles/Desktop/PerDogSynchronization" #
MAX_SEARCH_PERCENT = 0.25

# Word variants for synchronization
OK_VARIANTS = {"ok", "okay", "oke", "oak", "oh", "o"}
GO_VARIANTS = {"go", "goh", "toe", "two", "though"}
TAP_VARIANTS = {"tap", "ta", "da", "pat"}

print("Loading Whisper model...")
model = whisper.load_model("small.en") #

def normalize_word(w):
    return "".join(c for c in w.lower() if c.isalnum())

def process_session(repo_path):
    repo_path = Path(repo_path)
    # Finding the specific cleaned CSV and MP4 within the repetition folder
    csv_matches = list(repo_path.glob("*_cleaned.csv"))
    mp4_matches = list(repo_path.glob("*.MP4"))
    if not mp4_matches:
        mp4_matches = list(repo_path.glob("*.mp4"))

    if not csv_matches or not mp4_matches:
        return f"Skipped: Missing files in {repo_path.name}"

    csv_path, mp4_path = csv_matches[0], mp4_matches[0]
    log_dir = repo_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # 1. Audio Extraction via FFmpeg
    audio_wav = log_dir / "temp_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4_path), "-vn", "-ac", "1", 
        "-ar", "16000", "-c:a", "pcm_s16le", str(audio_wav)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Transcription with Whisper
    result = model.transcribe(str(audio_wav), language="en", word_timestamps=True)
    words = [w for seg in result["segments"] for w in seg.get("words", [])]
    
    # Extract tap midpoints for the template
    tap_times = [(float(w["start"]) + float(w["end"]))/2.0 
                 for w in words if normalize_word(w["word"]) in TAP_VARIANTS]
    
    if not tap_times:
        return f"Error: No taps detected in {repo_path.name}"

    # 3. IMU Processing
    df = pd.read_csv(csv_path)
    
    # Calculate sampling rate dynamically from 'Timestamp'
    dt = df['Timestamp'].diff().mean() 
    fs_imu = 1.0 / dt
    
    # Manually calculate Magnitude from Ax, Ay, Az
    acc_mag = np.sqrt(df["Ax"]**2 + df["Ay"]**2 + df["Az"]**2)
    
    # Zero-center for correlation
    imu_time = df['Timestamp'] - df['Timestamp'].iloc[0]
    acc_mag_centered = acc_mag - acc_mag.mean()
    
    # 4. Correlation Sync
    search_limit = int(len(acc_mag_centered) * MAX_SEARCH_PERCENT)
    imu_segment = acc_mag_centered.iloc[:search_limit].values
    
    # Create Synthetic Template from audio taps
    template_taps = np.array(tap_times) - tap_times[0]
    t_template = np.arange(0, template_taps[-1] + 0.5, dt)
    synthetic_template = np.zeros_like(t_template)
    
    width = int(0.05 * fs_imu) # 50ms pulse
    for tap_t in template_taps:
        idx = int(tap_t * fs_imu)
        for j in range(-width, width):
            if 0 <= idx + j < len(synthetic_template):
                synthetic_template[idx + j] = 1.0 - abs(j)/width

    # Cross-correlation to find the best fit
    corr = correlate(imu_segment, synthetic_template, mode='valid')
    best_lag_idx = np.argmax(corr)
    best_imu_sync_time = imu_time.iloc[best_lag_idx]
    
    # Calculate final offset
    time_to_adjust = best_imu_sync_time - tap_times[0]

    # 5. Data Alignment (Trim or Pad)
    if time_to_adjust > 0:
        rows_to_remove = int(time_to_adjust * fs_imu)
        df_sync = df.iloc[rows_to_remove:].copy()
    else:
        rows_to_pad = int(abs(time_to_adjust) * fs_imu)
        pad_data = {col: [np.nan] * rows_to_pad for col in df.columns}
        start_ts = df['Timestamp'].iloc[0]
        pad_data['Timestamp'] = [start_ts - (rows_to_pad - i) * dt for i in range(rows_to_pad)]
        df_sync = pd.concat([pd.DataFrame(pad_data), df], ignore_index=True)

    # Re-zero the relative time for the final output
    df_sync['Relative_Time'] = df_sync['Timestamp'] - df_sync['Timestamp'].iloc[0]
    
    out_name = f"{csv_path.stem}_synchronized.csv"
    df_sync.to_csv(repo_path / out_name, index=False)
    
    if audio_wav.exists(): audio_wav.unlink()
    return f"Success: {repo_path.name} | Offset: {time_to_adjust:.3f}s"

def main():
    # Crawl the directory structure: /data/PerDogSynchronization/Dog/Repetition
    clean = True
    base = Path(BASE_DIR)
    if clean:
        for dog_folder in base.iterdir():
            if dog_folder.is_dir():
                for rep_folder in dog_folder.iterdir():
                    if rep_folder.is_dir():
                        csv_matches = list(rep_folder.glob("*_synchronized.csv"))
                        if not csv_matches:
                            continue
                        csv_path = csv_matches[0]
                        os.remove(csv_path)
    for dog_folder in base.iterdir():
        if dog_folder.is_dir():
            print(f"\nProcessing Dog: {dog_folder.name}")
            for rep_folder in dog_folder.iterdir():
                if rep_folder.is_dir():
                    try:
                        print(process_session(rep_folder))
                    except Exception as e:
                        print(f"Failed {rep_folder.name}: {str(e)}")
        return

if __name__ == "__main__":
    main()