# train_eval_export.py
# -*- coding: utf-8 -*-
"""
双专家 + 二分类门控训练（方案 B），特征提取内嵌，ffmpeg 解码。

特性：
  - 特征提取内嵌，无独立 feature_extractor.py
  - ffmpeg 优先解码（和 annotate_gain.py 一致）
  - feat_cache 增量缓存，key 只用路径
  - --shift 同步调整所有阈值和分组边界

用法：
  python train_eval_export.py --csv delta_gain.csv --audio-root "D:\\Music" --n-jobs 16 --shift -5
"""

import os
import sys
import json
import shutil
import subprocess
import hashlib
import argparse
import warnings
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import librosa
import pyloudnorm as pyln
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


# ============================================================
# 特征提取（内嵌，ffmpeg 优先）
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
    feat_accum = np.zeros(42, dtype=np.float64)
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
CACHE_PATH     = 'handcraft_cache_42d_aug05.pkl'
FEAT_CACHE_DIR = 'feat_cache'
N_JOBS_MAX     = 16

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
N_INNER = 5
WEIGHT_HIGH = 5.0
EVAL_DIR = 'eval_out'
MODEL_DIR = 'model_dir'

AUG_RANGE = (-0.5, 0.5)

GATE_LABEL_THR = 0.0
GATE_SPW = 16.0
TAU_GRID_PROB = np.linspace(0.0, 1.0, 201)

LOW_THR_BASE = -5.0
A_THR_BASE = -4.0
EXTREME_THR_BASE = -2.0

LOW_THR = LOW_THR_BASE
A_THR = A_THR_BASE
EXTREME_THR = EXTREME_THR_BASE

WATCH_IDX = [1432, 708, 1411, 687]

AUDIO_EXTS = {'.flac', '.wav', '.mp3', '.m4a', '.mp4', '.m4b', '.m4r',
              '.ogg', '.opus', '.aiff', '.aif', '.ape', '.wv'}


# ============================================================
# CSV / 文件匹配
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
    for old, new in {'Ø':'O','ø':'o','Æ':'AE','æ':'ae','ß':'ss',
                     'Ł':'L','ł':'l','Œ':'OE','œ':'oe'}.items():
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
# 特征缓存（增量）
# ============================================================
def compute_csv_hash(csv_path):
    with open(csv_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def file_key(path):
    """只用绝对路径做 key。写标签不影响缓存。"""
    raw = os.path.abspath(path)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def extract_and_cache_one(path, y_label, cache_dir, aug_enable, aug_range):
    key = file_key(path)
    out_path = os.path.join(cache_dir, key + '.npz')

    if os.path.exists(out_path):
        try:
            d = np.load(out_path, allow_pickle=False)
            if 'x' in d and 'y' in d:
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

        if aug_enable:
            gain_db = float(np.random.uniform(aug_range[0], aug_range[1]))
            y_aug = y * (10.0 ** (gain_db / 20.0))
            feats_aug = extract_handcraft_features(y_aug, sr)
            if feats_aug is not None:
                vec_aug = np.array([feats_aug[k] for k in FEATURE_ORDER],
                                   dtype=np.float32)
            else:
                vec_aug = np.zeros(42, dtype=np.float32)
                gain_db = 0.0
        else:
            vec_aug = np.zeros(42, dtype=np.float32)
            gain_db = 0.0

        np.savez(out_path, x=vec, x_aug=vec_aug,
                 gain_db=np.float32(gain_db), y=np.float32(y_label))
        return {'key': key, 'path': path, 'y': float(y_label),
                'ok': True, 'cached': False, 'err': None}
    except Exception as e:
        return {'key': key, 'path': path, 'y': float(y_label),
                'ok': False, 'cached': False,
                'err': f'{type(e).__name__}: {e}'}


def load_cached_features(results, cache_dir):
    X_orig, y_orig, valid_paths = [], [], []
    X_aug, y_aug = [], []
    for r in results:
        if not r.get('ok'):
            continue
        npz = os.path.join(cache_dir, r['key'] + '.npz')
        try:
            d = np.load(npz, allow_pickle=False)
            x = d['x']; x_aug = d['x_aug']
            gain_db = float(d['gain_db'])
            y = float(r['y'])
        except Exception:
            continue
        X_orig.append(x); y_orig.append(y); valid_paths.append(r['path'])
        if np.any(x_aug) and gain_db != 0.0:
            X_aug.append(x_aug); y_aug.append(y - gain_db)

    X_orig = np.array(X_orig, dtype=np.float32)
    y_orig = np.array(y_orig, dtype=np.float32)
    if X_aug:
        X_aug = np.array(X_aug, dtype=np.float32)
        y_aug = np.array(y_aug, dtype=np.float32)
        X = np.vstack([X_orig, X_aug])
        y = np.concatenate([y_orig, y_aug])
    else:
        X, y = X_orig, y_orig
    return X, y, valid_paths


def build_cache_from_csv(csv_path, audio_root, cache_path,
                         force=False, aug_enable=True, n_jobs=8):
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
            if cache.get('csv_hash') == csv_hash:
                X = np.asarray(cache['X']); y = np.asarray(cache['y'])
                orig_paths = cache['orig_paths']
                print(f'[CACHE] 命中: {cache_path}  X={X.shape}')
                return X, y, orig_paths
            else:
                print('[CACHE] CSV 已变化，重建')
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
                if 'x' in d and 'x_aug' in d and 'gain_db' in d:
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
            futures = {ex.submit(extract_and_cache_one, p, y, FEAT_CACHE_DIR,
                                 aug_enable, AUG_RANGE): (p, y)
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
    print(f'[FEAT] 原始有效: {len(valid_paths)}')
    n_aug = len(X) - len(valid_paths)
    if n_aug > 0:
        print(f'[FEAT] 增强样本: {n_aug}')

    joblib.dump({'csv_hash': csv_hash, 'orig_paths': valid_paths,
                 'X': X, 'y': y}, cache_path, compress=3)
    print(f'[CACHE] 已保存: {cache_path}  X={X.shape}', flush=True)
    return X, y, valid_paths


# ============================================================
# 模型
# ============================================================
def train_reg(X, y, seed=0):
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                          num_leaves=31, min_child_samples=10,
                          random_state=seed, verbose=-1, n_jobs=-1)
    m.fit(X, y)
    return m


def train_gate_clf(G, target_bin, seed=0):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           num_leaves=15, min_child_samples=10,
                           random_state=seed, verbose=-1, n_jobs=-1,
                           scale_pos_weight=GATE_SPW)
    m.fit(G, target_bin)
    return m


def gate_proba(model, G):
    return model.predict_proba(G)[:, 1].astype(np.float64)


def _f6(low_p, a_p):
    d = a_p - low_p
    return np.column_stack([low_p, a_p, d, np.abs(d),
                            np.minimum(low_p, a_p),
                            np.maximum(low_p, a_p)]).astype(np.float32)


def gate_feat_new(X, low_p, a_p):
    return np.hstack([X, _f6(low_p, a_p)]).astype(np.float32)


def pick_tau(prob_pred, y, low_p, a_p, weight_high=WEIGHT_HIGH,
             tau_grid=TAU_GRID_PROB):
    w = np.where(y > EXTREME_THR, weight_high, 1.0)
    best_tau, best_cost = 0.5, np.inf
    for tau in tau_grid:
        route_a = prob_pred > tau
        pred = np.where(route_a, a_p, low_p)
        cost = float(np.sum(np.abs(y - pred) * w))
        if cost < best_cost:
            best_cost = cost
            best_tau = float(tau)
    return best_tau


# ============================================================
# OOF
# ============================================================
def run_oof(X, y, idx, seeds):
    rows = []
    t0 = time.time()
    for seed in seeds:
        kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_i, va_i) in enumerate(kf.split(X)):
            X_tr, X_va = X[tr_i], X[va_i]
            y_tr, y_va = y[tr_i], y[va_i]

            low_oof = np.zeros(len(tr_i))
            a_oof = np.zeros(len(tr_i))
            inner_kf = KFold(N_INNER, shuffle=True,
                             random_state=seed * 100 + fold)
            for itr, iva in inner_kf.split(X_tr):
                Xit, Xiv = X_tr[itr], X_tr[iva]
                yit = y_tr[itr]
                lm_ = yit < LOW_THR; am_ = yit >= A_THR
                if lm_.sum() < 30: lm_ = np.ones(len(yit), bool)
                if am_.sum() < 10: am_ = np.ones(len(yit), bool)
                lm = train_reg(Xit[lm_], yit[lm_], seed)
                am = train_reg(Xit[am_], yit[am_], seed)
                low_oof[iva] = lm.predict(Xiv)
                a_oof[iva] = am.predict(Xiv)

            lm_ = y_tr < LOW_THR; am_ = y_tr >= A_THR
            if lm_.sum() < 30: lm_ = np.ones(len(y_tr), bool)
            if am_.sum() < 10: am_ = np.ones(len(y_tr), bool)
            lm = train_reg(X_tr[lm_], y_tr[lm_], seed)
            am = train_reg(X_tr[am_], y_tr[am_], seed)
            low_val = lm.predict(X_va)
            a_val = am.predict(X_va)

            G_tr = gate_feat_new(X_tr, low_oof, a_oof)
            G_va = gate_feat_new(X_va, low_val, a_val)

            diff_target = np.abs(y_tr - low_oof) - np.abs(y_tr - a_oof)
            y_bin_tr = (diff_target > GATE_LABEL_THR).astype(int)

            prob_oof = np.zeros(len(tr_i))
            gkf = KFold(N_INNER, shuffle=True,
                        random_state=seed * 1000 + fold)
            for gtr, gva in gkf.split(G_tr):
                gn = train_gate_clf(G_tr[gtr], y_bin_tr[gtr], seed=seed)
                prob_oof[gva] = gate_proba(gn, G_tr[gva])
            tau = pick_tau(prob_oof, y_tr, low_oof, a_oof)

            gate = train_gate_clf(G_tr, y_bin_tr, seed=seed)
            prob_val = gate_proba(gate, G_va)
            route = (prob_val > tau).astype(int)

            for k, vi in enumerate(va_i):
                rows.append({
                    'seed': seed, 'fold': fold, 'idx': int(idx[vi]),
                    'y': float(y_va[k]),
                    'low_pred': float(low_val[k]),
                    'a_pred': float(a_val[k]),
                    'diff_pred': float(prob_val[k]),
                    'route_new': int(route[k]),
                    'tau_new': float(tau)})
        print(f'[seed {seed}] done, elapsed {time.time()-t0:.1f}s')
    return pd.DataFrame(rows)


# ============================================================
# 汇总
# ============================================================
def summarize_oof(df):
    df = df.copy()
    df['pred_gate_new'] = np.where(df['route_new'] == 1,
                                    df['a_pred'], df['low_pred'])
    df['pred_oracle'] = np.where(
        np.abs(df['y'] - df['a_pred']) < np.abs(df['y'] - df['low_pred']),
        df['a_pred'], df['low_pred'])
    grp = df.groupby('idx').agg({
        'y': 'mean', 'low_pred': 'mean', 'a_pred': 'mean',
        'diff_pred': 'mean', 'pred_gate_new': 'mean',
        'pred_oracle': 'mean', 'route_new': 'mean'}).reset_index()
    for name, col in [('low', 'low_pred'), ('a', 'a_pred'),
                      ('gate_new', 'pred_gate_new'), ('oracle', 'pred_oracle')]:
        grp[f'ae_{name}'] = np.abs(grp['y'] - grp[col])
    return grp


def group_mae(grp, shift=0.0):
    def lb(v): return v + shift
    bins = [
        ('ALL', -np.inf, np.inf, False),
        (f'y < {lb(-6):.1f}', -np.inf, lb(-6), True),
        (f'{lb(-6):.1f} <= y < {lb(-4):.1f}', lb(-6), lb(-4), True),
        (f'{lb(-4):.1f} <= y < {lb(-2):.1f}', lb(-4), lb(-2), True),
        (f'y >= {lb(-2):.1f}', lb(-2), np.inf, False)]
    rows = []
    for name, lo, hi, strict_hi in bins:
        if np.isinf(lo) and np.isinf(hi):
            m = np.ones(len(grp), bool)
        elif np.isinf(lo):
            m = grp['y'] < hi
        elif np.isinf(hi):
            m = grp['y'] >= lo
        elif strict_hi:
            m = (grp['y'] >= lo) & (grp['y'] < hi)
        else:
            m = (grp['y'] >= lo) & (grp['y'] <= hi)
        sub = grp[m]
        if len(sub) == 0:
            continue
        rows.append({
            'group': name, 'n': len(sub),
            'mae_low': sub['ae_low'].mean(),
            'mae_a': sub['ae_a'].mean(),
            'mae_gate_new': sub['ae_gate_new'].mean(),
            'mae_oracle': sub['ae_oracle'].mean(),
            'route_new_rate': sub['route_new'].mean()})
    return pd.DataFrame(rows)


def topk_errors(grp, k=50):
    cols = ['idx', 'y', 'low_pred', 'a_pred', 'diff_pred',
            'pred_gate_new', 'ae_gate_new', 'ae_low', 'ae_a', 'route_new']
    cols = [c for c in cols if c in grp.columns]
    return grp.sort_values('ae_gate_new', ascending=False).head(k).reset_index(drop=True)


# ============================================================
# 全量导出
# ============================================================
def final_train_and_export(X, y, model_dir, seed=0, label_shift=0.0):
    os.makedirs(model_dir, exist_ok=True)
    print('Computing expert OOF for final gate ...')
    kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
    low_oof = np.zeros(len(y), dtype=np.float32)
    a_oof = np.zeros(len(y), dtype=np.float32)
    for tr_i, va_i in kf.split(X):
        Xtr, Xva = X[tr_i], X[va_i]; ytr = y[tr_i]
        lm_ = ytr < LOW_THR; am_ = ytr >= A_THR
        if lm_.sum() < 30: lm_ = np.ones(len(ytr), bool)
        if am_.sum() < 10: am_ = np.ones(len(ytr), bool)
        lm = train_reg(Xtr[lm_], ytr[lm_], seed)
        am = train_reg(Xtr[am_], ytr[am_], seed)
        low_oof[va_i] = lm.predict(Xva)
        a_oof[va_i] = am.predict(Xva)

    G = gate_feat_new(X, low_oof, a_oof)
    diff_target = np.abs(y - low_oof) - np.abs(y - a_oof)
    y_bin = (diff_target > GATE_LABEL_THR).astype(int)

    gkf = KFold(N_INNER, shuffle=True, random_state=seed * 1000)
    prob_oof = np.zeros(len(y), dtype=np.float32)
    for gtr, gva in gkf.split(G):
        gn = train_gate_clf(G[gtr], y_bin[gtr], seed=seed)
        prob_oof[gva] = gate_proba(gn, G[gva])
    tau = pick_tau(prob_oof, y, low_oof, a_oof)
    print(f'Final tau (prob) = {tau:.4f}')

    print('Training final experts ...')
    lm_ = y < LOW_THR; am_ = y >= A_THR
    if lm_.sum() < 30: lm_ = np.ones(len(y), bool)
    if am_.sum() < 10: am_ = np.ones(len(y), bool)
    low_model = train_reg(X[lm_], y[lm_], seed)
    a_model = train_reg(X[am_], y[am_], seed)

    print('Training final gate ...')
    gate_model = train_gate_clf(G, y_bin, seed=seed)

    joblib.dump(low_model, os.path.join(model_dir, 'low_model.pkl'))
    joblib.dump(a_model, os.path.join(model_dir, 'a_model.pkl'))
    joblib.dump(gate_model, os.path.join(model_dir, 'gate_model.pkl'))

    meta = {
        'gate_type': 'binary_classifier',
        'tau': float(tau),
        'gate_label_thr': GATE_LABEL_THR,
        'gate_scale_pos_weight': GATE_SPW,
        'n_features_raw': int(X.shape[1]),
        'n_features_gate': int(G.shape[1]),
        'low_thr': float(LOW_THR),
        'a_thr': float(A_THR),
        'extreme_thr': float(EXTREME_THR),
        'low_thr_base': float(LOW_THR_BASE),
        'a_thr_base': float(A_THR_BASE),
        'extreme_thr_base': float(EXTREME_THR_BASE),
        'label_shift': float(label_shift),
        'weight_high': WEIGHT_HIGH,
        'seed': seed,
        'n_folds': N_FOLDS,
        'gate_feature_layout':
            'X_raw(42) | low_pred | a_pred | diff | |diff| | min(low,a) | max(low,a)',
        'gate_label_rule': 'label = 1 if |y-low|-|y-a| > 0 else 0',
        'gate_route_rule': 'prob = predict_proba(G)[:,1]; route_A if prob > tau',
        'variant': 'plan_B (binary gate, spw=16, ffmpeg decode)',
        'decode_backend': 'ffmpeg'}

    with open(os.path.join(model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'Models saved to {model_dir}/')
    return tau


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
    ap.add_argument('--no-aug', action='store_true')
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--shift', type=float, default=0.0)
    args = ap.parse_args()

    global LOW_THR, A_THR, EXTREME_THR
    LOW_THR = LOW_THR_BASE + args.shift
    A_THR = A_THR_BASE + args.shift
    EXTREME_THR = EXTREME_THR_BASE + args.shift

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
    print(f'aug        = {not args.no_aug}')
    print(f'n_seeds    = {args.n_seeds}')
    print(f'n_jobs     = {args.n_jobs}')
    print(f'shift      = {args.shift:+.2f}')
    print(f'  LOW_THR     = {LOW_THR:+.2f}')
    print(f'  A_THR       = {A_THR:+.2f}')
    print(f'  EXTREME_THR = {EXTREME_THR:+.2f}')

    X, y, orig_paths = build_cache_from_csv(
        csv_path=args.csv, audio_root=args.audio_root,
        cache_path=args.cache, force=args.force_recache,
        aug_enable=not args.no_aug, n_jobs=args.n_jobs)
    idx = np.arange(len(y))

    print(f'\nX={X.shape}  y={y.shape}  n_orig={len(orig_paths)}')
    print(f'y range: {y.min():.3f} ~ {y.max():.3f}')

    seeds = SEEDS[:max(1, args.n_seeds)]
    print('\n' + '=' * 80)
    print(f'Running {len(seeds)}x{N_FOLDS} OOF ...')
    df_oof = run_oof(X, y, idx, seeds)
    df_oof.to_csv(os.path.join(args.eval_dir, 'per_seed_sample.csv'), index=False)
    grp = summarize_oof(df_oof)
    grp.to_csv(os.path.join(args.eval_dir, 'per_idx.csv'), index=False)
    summary = group_mae(grp, shift=args.shift)
    summary.to_csv(os.path.join(args.eval_dir, 'group_summary.csv'), index=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 240)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print('\nGroup MAE:')
    print(summary.to_string(index=False))

    top = topk_errors(grp, k=args.topk)
    top.to_csv(os.path.join(args.eval_dir, f'top{args.topk}_errors.csv'), index=False)
    print(f'\nTop {args.topk} errors  ->  eval_out/top{args.topk}_errors.csv')

    ext = grp[grp['y'] >= EXTREME_THR].sort_values('y', ascending=False)
    ext.to_csv(os.path.join(args.eval_dir, 'extreme_samples.csv'), index=False)
    if len(ext) > 0:
        n_ok = int((ext['route_new'] >= 0.5).sum())
        print(f'[Extreme] route OK: {n_ok}/{len(ext)}   fail: {len(ext)-n_ok}')

    print('\n' + '=' * 80)
    print('Final training & export ...')
    final_train_and_export(X, y, args.model_dir, seed=0, label_shift=args.shift)
    print('\nAll done.')


if __name__ == '__main__':
    main()