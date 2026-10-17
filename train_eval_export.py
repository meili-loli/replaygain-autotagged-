# train_eval_export.py
# -*- coding: utf-8 -*-
"""
单模型 LGB 回归训练（v2 —— 2026-10-16 重写）

为什么从「双专家 + 二分类门控」换成单模型
----------------------------------------
干净评估协议（GroupKFold，杜绝增强样本 OOF 泄漏）下实测：

    双专家+门控 硬路由    MAE 0.940   >2dB 95 首   max 5.94
    单 LGB 回归           MAE 0.788   >2dB 60 首   max 4.88

旧架构在 MAE 和尾部指标上全面落后，连 Ridge 线性回归（0.820）都不如。
根因是 LOW_THR=-10 / A_THR=-9 之间夹着 -10 <= y < -9 这段
「两个专家都没训过」的空洞，推理时总得有一个专家来预测它。

v2 的关键修正
-------------
1. 单 LGB 回归 —— 无专家、无门控、无 tau
2. 评估用 GroupKFold —— 同一首歌的增强样本与原始样本锁在同一折。
   v1 用朴素 KFold，增强样本会漏进训练集，把 MAE 虚报低了 0.42 dB
   （0.519 vs 0.940）
3. 默认不再使用增强样本 —— 实测无收益（0.7844 无 vs 0.7877 有），
   而且它正是泄漏的来源
4. 集成是真的 —— 必须开 bagging_fraction / feature_fraction，
   否则 LightGBM 是确定性的，random_state 完全不生效，
   --n-ensemble 会训出 N 个一模一样的模型（v1 就是这样）

用法：
  python train_eval_export.py --csv delta_gain.csv --audio-root "/path/to/your/music" --n-jobs 16 --shift -5
"""

import os
import sys

# Windows 控制台默认 GBK，文件名里的 ♯ / ‡ 之类字符会让 print 抛异常
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

import json
import shutil
import subprocess
import hashlib
import argparse
import warnings
import time
import unicodedata
import platform
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import librosa
import pyloudnorm as pyln
from sklearn.model_selection import KFold, GroupKFold

warnings.filterwarnings('ignore')


# ============================================================
# 特征提取（内嵌，ffmpeg 优先）—— 未改动
# ============================================================
SR_TARGET = 44100
MAX_DURATION_SEC = 600
HPSS_KERNEL_SIZE = 15
FFMPEG_TIMEOUT = 1800

FEATURE_ORDER = [
    'loudness', 'true_peak', 'plr', 'lra',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'spectral_flatness', 'spectral_contrast', 'zcr',
    'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio',
    'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
    'loudness_std', 'loudness_skew', 'loudness_kurt',
    'loudness_deriv_std', 'loudness_range_short',
    'transient_strength', 'onset_density',
    'dynamic_range_db', 'rms_energy', 'plr_std',
    'lufs_distance', 'loudness_per_sec',
    'loudness_plr_interaction', 'lra_std_interaction', 'tp_duration_ratio',
    'harmonic_ratio', 'percussive_ratio',
    'spectral_entropy', 'low_mid_ratio', 'mid_high_ratio',
    'loudness_p10', 'loudness_p90', 'loudness_iqr',
    'crest_factor',
]
assert len(FEATURE_ORDER) == 42
N_FEATURES = 42

_FFMPEG_OK = None


def ffmpeg_available():
    global _FFMPEG_OK
    if _FFMPEG_OK is None:
        _FFMPEG_OK = shutil.which('ffmpeg') is not None
    return _FFMPEG_OK


def ffprobe_duration(path):
    if shutil.which('ffprobe') is None:
        return None
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=30)
        s = r.stdout.strip()
        if s:
            return float(s)
    except Exception:
        pass
    return None


def load_audio(path, sr_target=SR_TARGET, start_sec=0.0, duration_sec=None,
               mono=True):
    """ffmpeg 优先解码，fallback soundfile / librosa。"""
    if ffmpeg_available():
        try:
            cmd = ['ffmpeg', '-v', 'error', '-nostdin']
            if start_sec > 0:
                cmd += ['-ss', f'{start_sec:.3f}']
            cmd += ['-i', path]
            if duration_sec is not None:
                cmd += ['-t', f'{duration_sec:.3f}']
            cmd += ['-f', 'f32le', '-acodec', 'pcm_f32le',
                    '-ac', '1' if mono else '2',
                    '-ar', str(sr_target), '-']
            proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
            if proc.returncode == 0 and proc.stdout:
                data = np.frombuffer(proc.stdout, dtype=np.float32).copy()
                if len(data) > 0:
                    if not mono:
                        data = data.reshape(-1, 2)
                    return data, sr_target
        except Exception:
            pass

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        if mono and data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != sr_target:
            if data.ndim == 1:
                data = librosa.resample(data, orig_sr=sr, target_sr=sr_target,
                                        res_type='kaiser_fast')
            else:
                data = np.stack([
                    librosa.resample(data[:, c], orig_sr=sr, target_sr=sr_target,
                                     res_type='kaiser_fast')
                    for c in range(data.shape[1])], axis=1)
        if start_sec > 0 or duration_sec is not None:
            a = int(start_sec * sr_target)
            b = a + int((duration_sec or 1e9) * sr_target)
            data = data[a:b]
        return data, sr_target
    except Exception:
        y, sr = librosa.load(path, sr=sr_target, mono=mono,
                             res_type='kaiser_fast')
        if start_sec > 0 or duration_sec is not None:
            a = int(start_sec * sr_target)
            b = a + int((duration_sec or 1e9) * sr_target)
            y = y[a:b]
        return y, sr


def compute_true_peak(y, sr):
    try:
        return pyln.TruePeakMeter(sr).true_peak(y)
    except Exception:
        y_up = librosa.resample(y, orig_sr=sr, target_sr=sr * 4,
                                res_type='kaiser_fast')
        return 20 * np.log10(np.max(np.abs(y_up)) + 1e-12)


def extract_handcraft_features(y, sr):
    if len(y) / sr > MAX_DURATION_SEC:
        y = y[:MAX_DURATION_SEC * sr]
    if len(y) / sr < 0.5 or np.max(np.abs(y)) < 1e-6:
        return None

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    if not np.isfinite(loudness):
        loudness = -100.0
    true_peak = compute_true_peak(y, sr)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    S_db = librosa.power_to_db(S)

    centroid = librosa.feature.spectral_centroid(S=S, freq=freqs)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, freq=freqs)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(
        S=S, freq=freqs, roll_percent=0.85)[0].mean()
    flatness = librosa.feature.spectral_flatness(S=S)[0].mean()
    contrast = librosa.feature.spectral_contrast(S=S, freq=freqs)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()

    total_e = np.sum(S ** 2) + 1e-12
    low = np.sum(S[(freqs >= 20) & (freqs <= 250)] ** 2) / total_e
    mid = np.sum(S[(freqs > 250) & (freqs <= 4000)] ** 2) / total_e
    high = np.sum(S[(freqs > 4000) & (freqs <= 20000)] ** 2) / total_e

    mfcc = librosa.feature.mfcc(S=S_db, sr=sr, n_mfcc=5).mean(axis=1)

    hop = int(0.1 * sr)
    frame = int(0.4 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    lseq = 20 * np.log10(rms + 1e-12)

    if len(lseq) >= 2:
        lra = np.percentile(lseq, 95) - np.percentile(lseq, 10)
        std = np.std(lseq)
        skew = pd.Series(lseq).skew()
        kurt = pd.Series(lseq).kurt()
        skew = 0.0 if np.isnan(skew) else skew
        kurt = 0.0 if np.isnan(kurt) else kurt
        d_std = np.std(np.diff(lseq))
        r_short = np.max(lseq) - np.min(lseq)
    else:
        lra = std = skew = kurt = d_std = r_short = 0.0

    onset_env = librosa.onset.onset_strength(S=S_db, sr=sr)
    trans = onset_env.mean()
    onset_d = len(librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr)) / max(len(y) / sr, 0.01)

    db_y = 20 * np.log10(np.abs(y) + 1e-12)
    dyn_range = true_peak - np.min(db_y)
    plr = true_peak - loudness
    rms_e = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)

    fp = np.max(librosa.util.frame(y, frame_length=2048, hop_length=512), axis=0)
    fp_db = 20 * np.log10(np.maximum(fp, 1e-12))
    plr_std = np.std(fp_db - loudness)

    dur = len(y) / sr

    try:
        y_harm, y_perc = librosa.effects.hpss(y, kernel_size=HPSS_KERNEL_SIZE)
        harm_e = float(np.sum(y_harm ** 2))
        perc_e = float(np.sum(y_perc ** 2))
        del y_harm, y_perc
        total_t = harm_e + perc_e + 1e-12
        harmonic_ratio = harm_e / total_t
        percussive_ratio = perc_e / total_t
    except Exception:
        harmonic_ratio = percussive_ratio = 0.5

    p = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)
    spectral_entropy = float(-np.mean(np.sum(p * np.log2(p + 1e-12), axis=0)))
    low_mid_ratio = float(low / (mid + 1e-12))
    mid_high_ratio = float(mid / (high + 1e-12))

    if len(lseq) >= 2:
        loudness_p10 = float(np.percentile(lseq, 10))
        loudness_p90 = float(np.percentile(lseq, 90))
        loudness_iqr = float(np.percentile(lseq, 75) - np.percentile(lseq, 25))
    else:
        loudness_p10 = loudness_p90 = loudness_iqr = 0.0

    peak_abs = float(np.max(np.abs(y)))
    rms_val = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    crest_factor = 20 * np.log10(peak_abs / rms_val + 1e-12)

    return {
        'loudness': loudness, 'true_peak': true_peak, 'plr': plr, 'lra': lra,
        'spectral_centroid': centroid, 'spectral_bandwidth': bandwidth,
        'spectral_rolloff': rolloff, 'spectral_flatness': flatness,
        'spectral_contrast': contrast, 'zcr': zcr,
        'low_energy_ratio': low, 'mid_energy_ratio': mid,
        'high_energy_ratio': high,
        'mfcc_0': mfcc[0], 'mfcc_1': mfcc[1], 'mfcc_2': mfcc[2],
        'mfcc_3': mfcc[3], 'mfcc_4': mfcc[4],
        'loudness_std': std, 'loudness_skew': skew, 'loudness_kurt': kurt,
        'loudness_deriv_std': d_std, 'loudness_range_short': r_short,
        'transient_strength': trans, 'onset_density': onset_d,
        'dynamic_range_db': dyn_range, 'rms_energy': rms_e, 'plr_std': plr_std,
        'lufs_distance': -14.0 - loudness,
        'loudness_per_sec': loudness / max(dur, 0.1),
        'loudness_plr_interaction': loudness * plr,
        'lra_std_interaction': lra * std,
        'tp_duration_ratio': true_peak / max(dur, 0.1),
        'harmonic_ratio': harmonic_ratio,
        'percussive_ratio': percussive_ratio,
        'spectral_entropy': spectral_entropy,
        'low_mid_ratio': low_mid_ratio,
        'mid_high_ratio': mid_high_ratio,
        'loudness_p10': loudness_p10,
        'loudness_p90': loudness_p90,
        'loudness_iqr': loudness_iqr,
        'crest_factor': crest_factor,
    }


def extract_features(audio_path, sr_target=SR_TARGET, chunk_sec=MAX_DURATION_SEC):
    dur = ffprobe_duration(audio_path)
    if dur is None or dur <= chunk_sec:
        try:
            y, sr = load_audio(audio_path, sr_target)
        except Exception:
            return None
        feats = extract_handcraft_features(y, sr)
        if feats is None:
            return None
        return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)

    n_chunks = int(np.ceil(dur / chunk_sec))
    feat_accum = np.zeros(N_FEATURES, dtype=np.float64)
    total_weight = 0.0
    n_valid = 0
    for i in range(n_chunks):
        start = i * chunk_sec
        try:
            seg, sr = load_audio(audio_path, sr_target,
                                 start_sec=start, duration_sec=chunk_sec)
        except Exception:
            continue
        if len(seg) / sr < 0.5:
            continue
        f = extract_handcraft_features(seg, sr)
        if f is None:
            continue
        arr = np.array([f[k] for k in FEATURE_ORDER], dtype=np.float64)
        w = float(len(seg))
        feat_accum += arr * w
        total_weight += w
        n_valid += 1
    if total_weight == 0 or n_valid == 0:
        return None
    feat_accum /= total_weight
    return feat_accum.astype(np.float32)


# ============================================================
# 训练配置
# ============================================================
CACHE_PATH     = 'handcraft_cache_42d_v2.pkl'   # v1 是 _aug05，含增强样本
CACHE_VERSION  = 2
FEAT_CACHE_DIR = 'feat_cache'
N_JOBS_MAX     = 16

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
EVAL_DIR = 'eval_out'
MODEL_DIR = 'model_dir'

# 评估报表用的标签分组边界（不参与训练）。
# shift=-5 时对应 y<-11 / -11~-9 / -9~-7 / y>=-7
BIN_EDGES_BASE = (-6.0, -4.0, -2.0)

# 让 random_state 真正生效的子采样参数。
# 不设这两项时 LightGBM 是确定性的，random_state 完全不生效，
# 集成会退化成 N 个一模一样的模型。
BAG_PARAMS = dict(bagging_fraction=0.9, bagging_freq=1, feature_fraction=0.9)

DEFAULT_N_ENSEMBLE = 5

# v1 遗留的模型文件名，训练时清掉，免得 load_models 读错
LEGACY_MODEL_FILES = [
    'low_model.pkl', 'a_model.pkl', 'gate_model.pkl',
    'model.pkl',
] + [f'{p}_{i}.pkl' for p in ('low_model', 'a_model', 'gate_model')
     for i in range(20)]

AUDIO_EXTS = {'.flac', '.wav', '.mp3', '.m4a', '.mp4', '.m4b', '.m4r',
              '.ogg', '.opus', '.aiff', '.aif', '.ape', '.wv'}


# ============================================================
# CSV / 文件匹配 —— 未改动
# ============================================================
def read_csv_any_encoding(path):
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'无法读取 CSV: {path}')


def validate_csv(df):
    if 'delta_gain' not in df.columns:
        raise ValueError("CSV 缺少 'delta_gain' 列")
    col = None
    for c in ('file', 'filename', 'path'):
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("CSV 缺少 'file' / 'filename' / 'path' 列")
    files = df[col].astype(str).tolist()
    gains = df['delta_gain'].values.astype(float)
    mask = ~np.isnan(gains)
    files = [f for i, f in enumerate(files) if mask[i]]
    gains = gains[mask]
    gains = np.clip(gains, -30, 20)
    print(f'[CSV] 有效样本: {len(files)}, 标签范围 [{gains.min():.2f}, {gains.max():.2f}]')
    return files, gains


def normalize_filename(s):
    s = unicodedata.normalize('NFC', str(s))
    for old, new in {'Ø': 'O', 'ø': 'o', 'Æ': 'AE', 'æ': 'ae', 'ß': 'ss',
                     'Ł': 'L', 'ł': 'l', 'Œ': 'OE', 'œ': 'oe'}.items():
        s = s.replace(old, new)
    return s.lower().strip()


def build_file_index(audio_root):
    idx = {}
    dup = 0
    for dp, _, fns in os.walk(audio_root):
        for f in fns:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                key = normalize_filename(f)
                if key in idx:
                    dup += 1
                else:
                    idx[key] = os.path.join(dp, f)
    if dup > 0:
        print(f'[WARN] 发现 {dup} 个同名文件，只保留第一个')
    return idx


def match_csv_to_files(files, gains, audio_root):
    idx = build_file_index(audio_root)
    paths, y_out, miss = [], [], []
    for f, g in zip(files, gains):
        key = normalize_filename(os.path.basename(f))
        p = idx.get(key)
        if p is None:
            miss.append(f)
            continue
        paths.append(p)
        y_out.append(g)
    print(f'[MATCH] 匹配成功 {len(paths)} / {len(files)}')
    if miss:
        print(f'[MATCH] 未匹配 {len(miss)} 个，示例:')
        for m in miss[:10]:
            print(f'   - {m}')
    return paths, np.array(y_out, dtype=np.float32)


# ============================================================
# 特征缓存（增量）—— 去掉了增强样本
# ============================================================
def compute_csv_hash(csv_path):
    with open(csv_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def file_key(path):
    """只用绝对路径做 key。写标签不影响缓存。"""
    raw = os.path.abspath(path)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def extract_and_cache_one(path, y_label, cache_dir):
    key = file_key(path)
    out_path = os.path.join(cache_dir, key + '.npz')

    if os.path.exists(out_path):
        try:
            d = np.load(out_path, allow_pickle=False)
            if 'x' in d and d['x'].shape == (N_FEATURES,):
                return {'key': key, 'path': path, 'y': float(y_label),
                        'ok': True, 'cached': True, 'err': None}
        except Exception:
            pass

    try:
        y, sr = load_audio(path, SR_TARGET)
        feats = extract_handcraft_features(y, sr)
        if feats is None:
            return {'key': key, 'path': path, 'y': float(y_label),
                    'ok': False, 'cached': False, 'err': 'extract returned None'}
        vec = np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)
        np.savez(out_path, x=vec, y=np.float32(y_label))
        return {'key': key, 'path': path, 'y': float(y_label),
                'ok': True, 'cached': False, 'err': None}
    except Exception as e:
        return {'key': key, 'path': path, 'y': float(y_label),
                'ok': False, 'cached': False,
                'err': f'{type(e).__name__}: {e}'}


def load_cached_features(results, cache_dir):
    X, y, valid_paths = [], [], []
    for r in results:
        if not r.get('ok'):
            continue
        npz = os.path.join(cache_dir, r['key'] + '.npz')
        try:
            d = np.load(npz, allow_pickle=False)
            x = d['x']
            if x.shape != (N_FEATURES,):
                continue
        except Exception:
            continue
        X.append(x)
        y.append(float(r['y']))
        valid_paths.append(r['path'])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            valid_paths)


def build_cache_from_csv(csv_path, audio_root, cache_path, force=False, n_jobs=8):
    n_jobs = max(1, min(int(n_jobs), N_JOBS_MAX))
    os.makedirs(FEAT_CACHE_DIR, exist_ok=True)
    csv_hash = compute_csv_hash(csv_path)

    print('[CSV] 读取 ...', flush=True)
    df = read_csv_any_encoding(csv_path)
    files, gains = validate_csv(df)
    print(f'[MATCH] 扫描音频目录: {audio_root}', flush=True)
    paths, y_raw = match_csv_to_files(files, gains, audio_root)
    if not paths:
        sys.exit('无匹配音频文件')

    if not force and os.path.exists(cache_path):
        try:
            cache = joblib.load(cache_path)
            if (cache.get('csv_hash') == csv_hash
                    and cache.get('cache_version') == CACHE_VERSION):
                X = np.asarray(cache['X'])
                y = np.asarray(cache['y'])
                orig_paths = cache['orig_paths']
                print(f'[CACHE] 命中: {cache_path}  X={X.shape}')
                return X, y, orig_paths
            print('[CACHE] 版本或 CSV 已变化，重建')
        except Exception as e:
            print(f'[CACHE] 读取失败: {e}，重建')

    print(f'[SCAN] 预扫描 {len(paths)} 个文件的 feat_cache ...', flush=True)
    t0 = time.time()
    tasks_to_extract = []
    cached_results = []
    for p, y in zip(paths, y_raw):
        key = file_key(p)
        npz_path = os.path.join(FEAT_CACHE_DIR, key + '.npz')
        if not force and os.path.exists(npz_path):
            try:
                d = np.load(npz_path, allow_pickle=False)
                if 'x' in d and d['x'].shape == (N_FEATURES,):
                    cached_results.append({
                        'key': key, 'path': p, 'y': float(y),
                        'ok': True, 'cached': True, 'err': None})
                    continue
            except Exception:
                pass
        tasks_to_extract.append((p, y))

    n_cached = len(cached_results)
    n_need = len(tasks_to_extract)
    print(f'[SCAN] 命中缓存={n_cached}  需提取={n_need}  '
          f'耗时={time.time()-t0:.1f}s', flush=True)

    if n_need == 0:
        results = cached_results
    else:
        print(f'[FEAT] 提取 {n_need} 个文件（n_jobs={n_jobs}，ffmpeg）...', flush=True)
        t1 = time.time()
        new_results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(extract_and_cache_one, p, y, FEAT_CACHE_DIR): (p, y)
                       for p, y in tasks_to_extract}
            n_done = 0
            for fut in as_completed(futures):
                n_done += 1
                try:
                    r = fut.result(timeout=600)
                except Exception as e:
                    p, y = futures[fut]
                    r = {'key': file_key(p), 'path': p, 'y': float(y),
                         'ok': False, 'cached': False,
                         'err': f'{type(e).__name__}: {e}'}
                new_results.append(r)
                if n_done % 20 == 0 or n_done == n_need:
                    n_ok_ = sum(1 for x in new_results if x.get('ok'))
                    print(f'  [{n_done}/{n_need}]  ok={n_ok_}  '
                          f'耗时={time.time()-t1:.1f}s', flush=True)
        results = cached_results + new_results

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r.get('ok'))
    n_cached = sum(1 for r in results if r.get('cached'))
    n_err = sum(1 for r in results if not r.get('ok'))
    print(f'[FEAT] 完成，耗时 {elapsed:.1f}s  成功={n_ok}  '
          f'命中缓存={n_cached}  失败={n_err}', flush=True)
    if n_err > 0:
        for r in results:
            if not r.get('ok') and r.get('err'):
                print(f'   ERR: {os.path.basename(r["path"])} -> {r["err"]}')

    X, y, valid_paths = load_cached_features(results, FEAT_CACHE_DIR)
    print(f'[FEAT] 有效样本: {len(valid_paths)}')

    joblib.dump({'cache_version': CACHE_VERSION, 'csv_hash': csv_hash,
                 'orig_paths': valid_paths, 'X': X, 'y': y},
                cache_path, compress=3)
    print(f'[CACHE] 已保存: {cache_path}  X={X.shape}', flush=True)
    return X, y, valid_paths


# ============================================================
# 模型 —— 单 LGB 回归
# ============================================================
def train_reg(X, y, seed=0, bag=True):
    m = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=10, random_state=seed, verbose=-1, n_jobs=-1,
        **(BAG_PARAMS if bag else {}))
    m.fit(X, y)
    return m


# ============================================================
# OOF 评估
# ============================================================
def run_oof(X, y, seeds, groups=None):
    """干净 OOF。groups 非空时用 GroupKFold（防增强样本泄漏）。"""
    rows = []
    t0 = time.time()
    for seed in seeds:
        if groups is None:
            it = KFold(N_FOLDS, shuffle=True, random_state=seed).split(X)
        else:
            it = GroupKFold(N_FOLDS).split(X, y, groups)
        for fold, (tr_i, va_i) in enumerate(it):
            m = train_reg(X[tr_i], y[tr_i], seed)
            p = m.predict(X[va_i])
            for k, vi in enumerate(va_i):
                rows.append({
                    'seed': seed, 'fold': fold, 'idx': int(vi),
                    'y': float(y[vi]), 'pred': float(p[k])})
        print(f'[seed {seed}] done, elapsed {time.time()-t0:.1f}s')
    return pd.DataFrame(rows)


# ============================================================
# 两阶段（组信号）—— 阶段1: 42维; 阶段2: 42维 + 组信号
# ============================================================
def run_oof_two_stage(X, y, seeds, paths, use_keys=('dir', 'album', 'artist')):
    """两阶段 OOF。

    每折内：
      1. 一阶段模型在训练段做折内 OOF -> 训练段的 base_pred（无泄漏）
         一阶段模型（本折）预测验证段 -> 验证段的 base_pred
      2. 用完整 base_pred 算组信号（组内**其他成员**的均值，排除自己）
      3. 二阶段模型用 [X + 组信号] 拟合训练段，预测验证段
    """
    import group_signal as GS

    meta = GS.load_meta(GS.META_PATH)
    if not meta:
        print('[WARN] 找不到 library_meta.csv，组信号将退化为按目录分组。')
        print('       先跑 build_library_meta.py 才能用上专辑/艺人 tag。')
    n = len(y)
    rows = []
    t0 = time.time()
    for seed in seeds:
        for fold, (tr_i, va_i) in enumerate(
                KFold(N_FOLDS, shuffle=True, random_state=seed).split(X)):
            bp = np.zeros(n)
            # 训练段：折内 OOF，避免组信号偷看自己
            inner = KFold(N_FOLDS, shuffle=True, random_state=seed + 100)
            for a, b in inner.split(X[tr_i]):
                mi = train_reg(X[tr_i][a], y[tr_i][a], seed)
                bp[tr_i[b]] = mi.predict(X[tr_i][b])
            # 验证段：本折一阶段模型
            m1 = train_reg(X[tr_i], y[tr_i], seed)
            bp[va_i] = m1.predict(X[va_i])
            # 组信号
            G, _ = GS.build_group_features(paths, bp, meta, use_keys)
            m2 = train_reg(np.hstack([X[tr_i], G[tr_i]]), y[tr_i], seed)
            p = m2.predict(np.hstack([X[va_i], G[va_i]]))
            for k, vi in enumerate(va_i):
                rows.append({
                    'seed': seed, 'fold': fold, 'idx': int(vi),
                    'y': float(y[vi]), 'pred': float(p[k])})
        print(f'[seed {seed}] 两阶段 done, elapsed {time.time()-t0:.1f}s')
    return pd.DataFrame(rows)


def final_train_two_stage(X, y, paths, model_dir, seeds, label_shift=0.0,
                          oof_stats=None, use_keys=('dir', 'album', 'artist')):
    """导出两阶段模型。

    阶段1 用全量数据训（那么全库 base_pred 里就含了训练集自己的预测）。
    为了让二阶段训练时的组信号与推理时同分布，训练段仍用**折外** base_pred。

    产物
      stage1/model_{i}.pkl     一阶段（42 维）
      stage2/model_{i}.pkl     二阶段（42 + 组信号）
      meta.json                model_type = two_stage_lgb
    """
    import group_signal as GS

    os.makedirs(model_dir, exist_ok=True)
    for fn in LEGACY_MODEL_FILES:
        p = os.path.join(model_dir, fn)
        if os.path.isfile(p):
            os.remove(p)
            print(f'  [CLEAN] 删除遗留 {fn}')

    meta_gs = GS.load_meta(GS.META_PATH)

    # ---- 一阶段 ----
    s1_dir = os.path.join(model_dir, 'stage1')
    s2_dir = os.path.join(model_dir, 'stage2')
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)

    n = len(y)
    # 训练段的折外 base_pred（取多 seed 平均，与推理行为一致）
    bp_oof = np.zeros(n)
    cnt = np.zeros(n)
    for s in seeds:
        for a, b in KFold(N_FOLDS, shuffle=True,
                          random_state=s + 100).split(X):
            mi = train_reg(X[a], y[a], s)
            bp_oof[b] += mi.predict(X[b])
            cnt[b] += 1
    bp_oof /= np.maximum(cnt, 1)

    G_tr, used_keys = GS.build_group_features(paths, bp_oof, meta_gs, use_keys)

    n_ens = len(seeds)
    for i, s in enumerate(seeds):
        print(f'[ensemble {i+1}/{n_ens}] seed={s} ...')
        m1 = train_reg(X, y, s)
        name = 'model.pkl' if n_ens == 1 else f'model_{i}.pkl'
        joblib.dump(m1, os.path.join(s1_dir, name))
        m2 = train_reg(np.hstack([X, G_tr]), y, s)
        joblib.dump(m2, os.path.join(s2_dir, name))

    meta = {
        'model_type': 'two_stage_lgb',
        'n_ensemble': int(n_ens),
        'seeds': [int(s) for s in seeds],
        'bag_params': BAG_PARAMS,
        'n_features_raw': int(X.shape[1]),
        'group_signal_keys': list(used_keys),
        'n_group_features': int(G_tr.shape[1]),
        'n_features_total': int(X.shape[1] + G_tr.shape[1]),
        'meta_file': GS.META_PATH,
        'label_shift': float(label_shift),
        'bin_edges_base': list(BIN_EDGES_BASE),
        'oof_stats': oof_stats or {},
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'n_train_samples': int(X.shape[0]),
        'variant': 'two-stage LGB: 42d base + group-signal (dir/album/artist)',
        'decode_backend': 'ffmpeg',
        'python': platform.python_version(),
        'lightgbm': lgb.__version__,
    }
    with open(os.path.join(model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'两阶段模型已保存到 {model_dir}/stage1|stage2  ({n_ens} 个 seed)')
    return meta


def summarize_oof(df):
    grp = df.groupby('idx').agg({'y': 'mean',
                                 'pred': ['mean', 'std']}).reset_index()
    grp.columns = ['idx', 'y', 'pred', 'pred_std']
    grp['pred_std'] = grp['pred_std'].fillna(0.0)
    grp['ae'] = np.abs(grp['y'] - grp['pred'])
    return grp


def group_mae(grp, shift=0.0):
    def lb(v):
        return v + shift
    e0, e1, e2 = (lb(v) for v in BIN_EDGES_BASE)
    bins = [
        ('ALL', -np.inf, np.inf),
        (f'y < {e0:.1f}', -np.inf, e0),
        (f'{e0:.1f} <= y < {e1:.1f}', e0, e1),
        (f'{e1:.1f} <= y < {e2:.1f}', e1, e2),
        (f'y >= {e2:.1f}', e2, np.inf)]
    rows = []
    for name, lo, hi in bins:
        if np.isinf(lo) and np.isinf(hi):
            m = np.ones(len(grp), bool)
        elif np.isinf(lo):
            m = grp['y'] < hi
        elif np.isinf(hi):
            m = grp['y'] >= lo
        else:
            m = (grp['y'] >= lo) & (grp['y'] < hi)
        sub = grp[m]
        if len(sub) == 0:
            continue
        a = sub['ae']
        rows.append({
            'group': name, 'n': len(sub),
            'mae': a.mean(),
            'p95': a.quantile(0.95),
            'max': a.max(),
            'gt1': int((a > 1).sum()),
            'gt2': int((a > 2).sum()),
            'gt3': int((a > 3).sum()),
            'pred_std_mean': sub['pred_std'].mean()})
    return pd.DataFrame(rows)


def topk_errors(grp, k=50):
    return (grp.sort_values('ae', ascending=False).head(k)
            .reset_index(drop=True)[['idx', 'y', 'pred', 'pred_std', 'ae']])


# ============================================================
# 全量训练并导出
# ============================================================
def final_train_and_export(X, y, model_dir, seeds, label_shift=0.0,
                           oof_stats=None):
    os.makedirs(model_dir, exist_ok=True)
    # 清掉 v1 遗留文件
    for fn in LEGACY_MODEL_FILES:
        p = os.path.join(model_dir, fn)
        if os.path.isfile(p):
            os.remove(p)
            print(f'  [CLEAN] 删除遗留 {fn}')

    n_ens = len(seeds)
    for i, s in enumerate(seeds):
        print(f'[ensemble {i+1}/{n_ens}] seed={s} ...')
        m = train_reg(X, y, s)
        name = 'model.pkl' if n_ens == 1 else f'model_{i}.pkl'
        joblib.dump(m, os.path.join(model_dir, name))

    meta = {
        'model_type': 'single_lgb',
        'n_ensemble': int(n_ens),
        'seeds': [int(s) for s in seeds],
        'bag_params': BAG_PARAMS,
        'n_features_raw': int(X.shape[1]),
        'label_shift': float(label_shift),
        'bin_edges_base': list(BIN_EDGES_BASE),
        'oof_stats': oof_stats or {},
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'n_train_samples': int(X.shape[0]),
        'variant': 'single LGB regressor + bagging-enabled N-seed ensemble',
        'decode_backend': 'ffmpeg',
        'python': platform.python_version(),
        'lightgbm': lgb.__version__,
    }
    with open(os.path.join(model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'模型已保存到 {model_dir}/  ({n_ens} 个 seed)')
    return meta


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--audio-root', required=True)
    ap.add_argument('--cache', default=CACHE_PATH)
    ap.add_argument('--eval-dir', default=EVAL_DIR)
    ap.add_argument('--model-dir', default=MODEL_DIR)
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--force-recache', action='store_true')
    ap.add_argument('--force-reextract', action='store_true')
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--n-ensemble', type=int, default=DEFAULT_N_ENSEMBLE)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--shift', type=float, default=0.0)
    ap.add_argument('--no-bag', action='store_true')
    ap.add_argument('--group-signal', action='store_true',
                    help='启用两阶段（42维 + 目录/专辑/艺人组信号）。'
                         '默认关闭，关闭时行为与旧版完全一致。')
    args = ap.parse_args()

    if args.no_bag:
        BAG_PARAMS.clear()
        print('[WARN] --no-bag：random_state 将完全失效，集成会退化')

    os.makedirs(args.eval_dir, exist_ok=True)

    if args.force_reextract:
        if os.path.exists(FEAT_CACHE_DIR):
            shutil.rmtree(FEAT_CACHE_DIR)
            print(f'[CLEAN] 已清空 {FEAT_CACHE_DIR}/')
        if os.path.exists(args.cache):
            os.remove(args.cache)
            print(f'[CLEAN] 已删除 {args.cache}')

    print('=' * 80)
    print(f'CSV        = {args.csv}')
    print(f'audio-root = {args.audio_root}')
    print(f'cache      = {args.cache}')
    print(f'decoder    = {"ffmpeg" if ffmpeg_available() else "soundfile/librosa (fallback)"}')
    print('模型       = 单 LGB 回归（无专家 / 无门控 / 无增强样本）')
    print(f'bagging    = {BAG_PARAMS or "关闭"}')
    print(f'n_seeds    = {args.n_seeds}')
    print(f'n_ensemble = {args.n_ensemble}')
    print(f'n_jobs     = {args.n_jobs}')
    print(f'shift      = {args.shift:+.2f}')

    X, y, orig_paths = build_cache_from_csv(
        csv_path=args.csv, audio_root=args.audio_root,
        cache_path=args.cache, force=args.force_recache, n_jobs=args.n_jobs)

    print(f'\nX={X.shape}  y={y.shape}  n_orig={len(orig_paths)}')
    print(f'y range: {y.min():.3f} ~ {y.max():.3f}')

    seeds = SEEDS[:max(1, args.n_seeds)]
    print('\n' + '=' * 80)
    if args.group_signal:
        print(f'Running {len(seeds)}x{N_FOLDS} 两阶段 OOF（42维 + 组信号）...')
        df_oof = run_oof_two_stage(X, y, seeds, orig_paths)
    else:
        print(f'Running {len(seeds)}x{N_FOLDS} OOF ...')
        df_oof = run_oof(X, y, seeds)
    df_oof.to_csv(os.path.join(args.eval_dir, 'per_seed_sample.csv'), index=False)
    grp = summarize_oof(df_oof)
    grp.to_csv(os.path.join(args.eval_dir, 'per_idx.csv'), index=False)
    summary = group_mae(grp, shift=args.shift)
    summary.to_csv(os.path.join(args.eval_dir, 'group_summary.csv'), index=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 240)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print('\n分组指标:')
    print(summary.to_string(index=False))

    a = grp['ae']
    oof_stats = {
        'mae': float(a.mean()),
        'rmse': float(np.sqrt((a ** 2).mean())),
        'p90': float(a.quantile(0.90)),
        'p95': float(a.quantile(0.95)),
        'p99': float(a.quantile(0.99)),
        'max': float(a.max()),
        'gt1': int((a > 1).sum()),
        'gt2': int((a > 2).sum()),
        'gt3': int((a > 3).sum()),
        'n': int(len(grp)),
    }
    print('\n整体指标（干净 OOF）:')
    print(f"  MAE={oof_stats['mae']:.4f}  RMSE={oof_stats['rmse']:.4f}  "
          f"p95={oof_stats['p95']:.3f}  max={oof_stats['max']:.2f}")
    print(f"  >1dB={oof_stats['gt1']} 首  >2dB={oof_stats['gt2']} 首  "
          f">3dB={oof_stats['gt3']} 首   (共 {oof_stats['n']} 首)")

    top = topk_errors(grp, k=args.topk)
    top.to_csv(os.path.join(args.eval_dir, f'top{args.topk}_errors.csv'), index=False)
    print(f'\nTop {args.topk} errors  ->  {args.eval_dir}/top{args.topk}_errors.csv')

    print('\n' + '=' * 80)
    print('Final training & export ...')
    if args.group_signal:
        final_train_two_stage(X, y, orig_paths, args.model_dir, seeds,
                             label_shift=args.shift, oof_stats=oof_stats)
    else:
        final_train_and_export(X, y, args.model_dir, seeds,
                              label_shift=args.shift, oof_stats=oof_stats)
    print('\nAll done.')


if __name__ == '__main__':
    main()
