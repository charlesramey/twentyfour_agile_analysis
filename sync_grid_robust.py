#!/usr/bin/env python3

"""
Robust 4-way video sync by audio correlation with sanity checks.

Usage:
  python sync_grid_robust.py /path/to/rootdir

Root dir must contain:
  - 2024AgileCupMetadata_ScribeNotes_CameraInfo.xlsx   (first sheet used)
  - "MMJ 1 GoPro", "MMJ 2 GoPro", "MMJ 3 GoPro", "MMJ 4 GoPro" subfolders

The sheet should have columns like "MMJ2 Video Name" (flexible headers).
The script:
  - auto-detects the 4 MMJ file-name columns
  - validates each row's 4 videos:
      * file exists
      * has an audio track
      * frame rate not timelapse (avg fps >= 10)
      * audio correlation with ref (MMJ1) has a clear peak and reasonable offset
  - if any check fails → skip the row
  - trims/pads each video to align by the detected offsets
  - writes a 2x2 grid mosaic to an output directory named "<individual name>_<sheet name>"

Hardcoded parameters:
  SAMPLERATE = 22050
  GRID_SIZE  = 1080
  MAX_SHIFT_S = 10.0
  MIN_FPS = 10.0
  MIN_PEAK = 0.15  # normalized peak corr threshold (heuristic)
"""

import os
import re
import sys
import math
import json
import shutil
import tempfile
import subprocess
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ------------------ Config ------------------
SPREADSHEET_NAME = "2024AgileCupMetadata_ScribeNotes_CameraInfo.xlsx"
SAMPLERATE = 22050
GRID_SIZE = 4096
MAX_SHIFT_S = 10.0
MIN_FPS = 10.0
MIN_PEAK = 0.15

DIR_MAP = {
    "1": "MMJ 1 GoPro",
    "2": "MMJ 2 GoPro",
    "3": "MMJ 3 GoPro",
    "4": "MMJ 4 GoPro",
}

# ------------------ Helpers ------------------
def run(cmd: List[str], check=True, capture_output=False, text=False):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)

def ensure_ffmpeg():
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            print(f"ERROR: '{exe}' not found on PATH.", file=sys.stderr); sys.exit(1)

def ffprobe_streams(path: str) -> dict:
    """Return ffprobe JSON for streams."""
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path],
              capture_output=True, text=True)
    return json.loads(out.stdout)

def has_audio_stream(path: str) -> bool:
    try:
        info = ffprobe_streams(path)
        return any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    except Exception:
        return False

def get_avg_fps(path: str) -> float:
    try:
        info = ffprobe_streams(path)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                r = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
                if r and r != "0/0":
                    num, den = r.split("/")
                    den = float(den) if float(den) != 0 else 1.0
                    return float(num) / den
    except Exception:
        pass
    return 0.0

def get_duration_seconds(video_path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path],
              capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        return 0.0

def extract_audio_mono_wav(video_path: str, wav_path: str, sr: int = SAMPLERATE):
    run([
        "ffmpeg","-y","-i",video_path,"-vn","-ac","1","-ar",str(sr),"-acodec","pcm_s16le",wav_path
    ])

def read_wav(path: str) -> Tuple[int, np.ndarray]:
    sr, y = wavfile.read(path)
    if y.ndim > 1: y = y[:,0]
    if y.dtype != np.float32:
        y = y.astype(np.float32) / (np.iinfo(np.int16).max + 1.0)
    return sr, y

def normalized_xcorr_offset_and_peak(ref: np.ndarray, tgt: np.ndarray, sr: int, max_shift_s: float) -> Tuple[float, float]:
    """Return (lag_seconds, peak_norm) where positive lag means tgt starts later than ref."""
    ref_z = ref - np.mean(ref); ref_z = ref_z / (np.std(ref_z)+1e-8)
    tgt_z = tgt - np.mean(tgt); tgt_z = tgt_z / (np.std(tgt_z)+1e-8)
    corr = fftconvolve(tgt_z[::-1], ref_z, mode="full")
    lags = np.arange(-len(tgt_z)+1, len(ref_z))
    max_shift = int(max_shift_s * sr)
    mask = (lags >= -max_shift) & (lags <= max_shift)
    corr_masked = corr[mask]; lags_masked = lags[mask]
    best_idx = int(np.argmax(corr_masked))
    best_lag = int(lags_masked[best_idx])
    # Normalize the peak by overlap length to roughly get a correlation coefficient-like value
    overlap = min(len(ref_z), len(tgt_z))
    peak_norm = float(corr_masked[best_idx]) / max(1.0, overlap)
    return best_lag / float(sr), peak_norm

def make_sync_cut(video_in: str, video_out: str, offset_s: float, cut_to_s: float, scale_w: int, scale_h: int):
    vf_parts = []; af_parts = []
    if offset_s > 1e-4:
        vf_parts.append(f"tpad=start_duration={offset_s}")
        af_parts.append(f"adelay={int(offset_s*1000)}|{int(offset_s*1000)}"); af_parts.append("apad")
    elif offset_s < -1e-4:
        start = abs(offset_s)
        vf_parts.append(f"trim=start={start},setpts=PTS-STARTPTS")
        af_parts.append(f"atrim=start={start},asetpts=PTS-STARTPTS")
    vf_parts.append(f"scale=w={scale_w}:h={scale_h}:force_original_aspect_ratio=decrease")
    vf_parts.append(f"pad=w={scale_w}:h={scale_h}:x=(ow-iw)/2:y=(oh-ih)/2:color=black")
    vf_parts.append(f"trim=end={cut_to_s},setpts=PTS-STARTPTS")
    af_parts.append(f"atrim=end={cut_to_s},asetpts=PTS-STARTPTS")
    vf = ",".join(vf_parts); af = ",".join(af_parts)
    run([
        "ffmpeg","-y","-i",video_in,
        "-filter_complex", f"[0:v]{vf}[v];[0:a]{af}[a]",
        "-map","[v]","-map","[a]",
        "-c:v","h264","-crf","18","-preset","veryfast",
        "-c:a","aac","-b:a","128k",
        video_out
    ])

def make_grid(videos: List[str], out_path: str, grid_size: int, take_audio_from: int = 0, mute: bool = False):
    tile = grid_size // 2
    inputs = []; scale_parts = []; maps = []
    for i, p in enumerate(videos):
        inputs += ["-i", p]
        scale_parts.append(f"[{i}:v]scale={tile}:{tile}:force_original_aspect_ratio=decrease,pad={tile}:{tile}:(ow-iw)/2:(oh-ih)/2:black[v{i}]")
        maps.append(f"[v{i}]")
    xstack = f"{''.join(maps)}xstack=inputs=4:layout=0_0|{tile}_0|0_{tile}|{tile}_{tile}[vout]"
    if mute:
        audio = "anullsrc=r=44100:cl=mono,atrim=0,setpts=PTS-STARTPTS[aout]"
    else:
        audio = f"[0:a]anull[aout]"
    fc = ";".join(scale_parts + [xstack, audio])
    run(["ffmpeg","-y"] + inputs + [
        "-filter_complex", fc,
        "-map","[vout]","-map","[aout]",
        "-c:v","h264","-crf","18","-preset","veryfast",
        "-c:a","aac","-b:a","128k",
        out_path
    ])

def detect_mmj_columns(df: pd.DataFrame):
    colmap = {}
    for col in df.columns:
        s = str(col).lower()
        m = re.search(r"\bmmj\s*([1-4])\b", s)
        if m and m.group(1) not in colmap:
            colmap[m.group(1)] = col
    missing = [k for k in ["1","2","3","4"] if k not in colmap]
    if missing:
        raise ValueError(f"Missing MMJ columns for: {missing}. Found: {colmap}")
    return colmap

def detect_name_column(df: pd.DataFrame):
    for col in df.columns:
        s = str(col).lower()
        if re.search(r"\b(individual|handler|name)\b", s):
            return col
    # fallback if exists
    return "Name" if "Name" in df.columns else None

def sanitize_for_path(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

# ------------------ Main ------------------
def main(root_dir: str):
    ensure_ffmpeg()

    # Load workbook (first sheet name for suffix)
    xls = pd.ExcelFile(os.path.join(root_dir, SPREADSHEET_NAME), engine="openpyxl")
    sheet_name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)

    colmap = detect_mmj_columns(df)  # {"1": "...", ...}
    name_col = detect_name_column(df)
    count = 0
    for idx, row in df.iterrows():
        print(row)
        if count is 2:
            quit()
        count += 1
        # Build absolute video paths
        files = []
        for cam in ["1","2","3","4"]:
            fname = str(row.get(colmap[cam], "")).strip()
            print(fname)
            if not fname or fname.lower() == "nan":
                print("in the failed if")
                print(f"[Row {idx}] Missing filename for MMJ{cam}; skipping.")
                files = []
                break
            print(f'appending {os.path.join(root_dir, DIR_MAP[cam], fname, ".MP4")}')
            files.append(os.path.join(root_dir, DIR_MAP[cam], fname + ".MP4"))
        if not files:
            continue
        # Quick file existence check
        if any(not os.path.exists(p) for p in files):
            print(f"[Row {idx}] One or more files missing; skipping.")
            continue

        # Validate audio & fps
        problems = []
        for i, p in enumerate(files):
            if not has_audio_stream(p):
                problems.append(f"cam{i+1} no-audio")
            fps = get_avg_fps(p)
            if fps < MIN_FPS:
                problems.append(f"cam{i+1} low-fps {fps:.2f} (timelapse?)")
        if problems:
            print(f"[Row {idx}] Validation fail: {problems}; skipping.")
            continue

        # Extract audio and compute offsets/peaks
        with tempfile.TemporaryDirectory() as td:
            wavs = []
            print("=========================================================================\n\n\n\n\n")
            print("Extracting audio")
            print("=========================================================================\n\n\n\n\n")
            for i, vid in enumerate(files):
                w = os.path.join(td, f"a{i}.wav")
                extract_audio_mono_wav(vid, w, sr=SAMPLERATE)
                wavs.append(w)
            sr0, y0 = read_wav(wavs[0])
            offsets = [0.0]; peaks = [1.0]
            print("=========================================================================\n\n\n\n\n")
            print("Attempting alignment")
            print("=========================================================================\n\n\n\n\n")
            for i in range(1, 4):
                sri, yi = read_wav(wavs[i])
                if sri != sr0:
                    print(f"[Row {idx}] WAV SR mismatch; skipping.")
                    offsets = []; break
                lag, peak = normalized_xcorr_offset_and_peak(y0, yi, sr0, MAX_SHIFT_S)
                offsets.append(lag); peaks.append(peak)
                print("=========================================================================\n")
                print(f"[Row {idx}] Offset: {lag}.")
                print("=========================================================================\n")
            if not offsets:
                continue
            # Check peak quality and offset reasonableness
            if any(abs(o) > MAX_SHIFT_S for o in offsets):
                print("=========================================================================\n\n\n\n\n")
                print(f"[Row {idx}] Offset exceeds {MAX_SHIFT_S}s: {offsets}; skipping.")
                print("=========================================================================\n\n\n\n\n")
                continue
            if any(p < MIN_PEAK for p in peaks[1:]):  # ignore ref peak
                print("=========================================================================\n\n\n\n\n")
                print(f"[Row {idx}] Weak audio match (peaks {peaks}); skipping.")
                print("=========================================================================\n\n\n\n\n")
                continue

            # Compute aligned common duration
            durs = [get_duration_seconds(v) for v in files] # Computing how long each video is
            starts = [max(0.0, -o) for o in offsets] # how far before (if the lag is negative) or after (if the lag is positive) was the video started compared to the reference (camera 1)
            aligned_durs = [dur - starts[i] for i, dur in enumerate(durs)] # Subtract the
            common_dur = max(0.1, min(aligned_durs))

            # Prepare per-video trimmed/padded
            tile = GRID_SIZE // 2
            prepped = []
            for i, vid in enumerate(files):
                outv = os.path.join(td, f"prep{i}.mp4")
                make_sync_cut(vid, outv, offset_s=offsets[i], cut_to_s=common_dur, scale_w=tile, scale_h=tile)
                prepped.append(outv)

            # Output dir: "<individual name>_<sheet name>"
            indiv_val = str(row.get(name_col, "unknown")) if name_col else "unknown"
            out_dir_name = f"{sanitize_for_path(indiv_val)}_{sanitize_for_path(sheet_name)}"
            out_dir = os.path.join(root_dir, out_dir_name)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"row{idx:03d}_synced_grid.mp4")
            make_grid(prepped, out_path, grid_size=GRID_SIZE, take_audio_from=0, mute=False)
            print(f"[Row {idx}] OK → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sync_grid_robust.py <rootdir>")
        sys.exit(1)
    main(sys.argv[1])
