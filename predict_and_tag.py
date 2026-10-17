# predict_and_tag.py
# -*- coding: utf-8 -*-
"""
批量预测 ReplayGain 并写标签。多进程 + feat_cache + label_shift。

写入逻辑完全照搬 write_rg_fast.py：
  - FLAC / OGG / Opus：小写键 replaygain_track_gain / replaygain_track_peak
  - MP3：ID3 TXXX frame
  - M4A / MP4：freeform atom
  - peak 从 true_peak 特征取，转线性值写入
"""

import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys

# Windows 控制台默认 GBK，文件名里的 ♯ / ‡ 之类字符会让 print 抛
# UnicodeEncodeError 并把整个推理进程带崩。改成替换而不是抛异常。
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except Exception:
    pass

import csv
import json
import shutil
import subprocess
import hashlib
import argparse
import traceback
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import librosa
import pyloudnorm as pyln

try:
    from mutagen.flac import FLAC
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
    from mutagen.oggvorbis import OggVorbis
    from mutagen.oggopus import OggOpus
    from mutagen.mp4 import MP4
    import mutagen
except ImportError:
    sys.exit("请先 pip install mutagen")


# ============================================================
# 特征提取（内嵌，ffmpeg 优先）
# ============================================================
SR_TARGET = 44100
MAX_DURATION_SEC = 7200
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
        return data, sr_target
    except Exception:
        return librosa.load(path, sr=sr_target, mono=mono,
                            res_type='kaiser_fast')


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
    hop = int(0.1 * sr); frame = int(0.4 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    lseq = 20 * np.log10(rms + 1e-12)
    if len(lseq) >= 2:
        lra = np.percentile(lseq, 95) - np.percentile(lseq, 10)
        std = np.std(lseq)
        skew = pd.Series(lseq).skew(); kurt = pd.Series(lseq).kurt()
        skew = 0.0 if np.isnan(skew) else skew
        kurt = 0.0 if np.isnan(kurt) else kurt
        d_std = np.std(np.diff(lseq)); r_short = np.max(lseq) - np.min(lseq)
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
        harm_e = float(np.sum(y_harm ** 2)); perc_e = float(np.sum(y_perc ** 2))
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
        'crest_factor': crest_factor}


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
    total_weight = 0.0; n_valid = 0
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
        feat_accum += arr * w; total_weight += w; n_valid += 1
    if total_weight == 0 or n_valid == 0:
        return None
    feat_accum /= total_weight
    return feat_accum.astype(np.float32)


# ============================================================
# 推理配置
# ============================================================
MODEL_DIR = 'model_dir'
LOG_DIR = 'inference_logs'
FEAT_CACHE_DIR = 'feat_cache'

WATCH_THR_BASE = -3.0
CONF_EXTREME_BASE = -2.0
CONF_HIGH_BASE = -4.0
CONF_LOW_BASE = -5.0

AUDIO_EXTS = {'.flac', '.mp3', '.m4a', '.mp4', '.m4b', '.m4r',
              '.ogg', '.opus', '.wav', '.aiff', '.aif', '.ape', '.wv'}


def file_key(path):
    raw = os.path.abspath(path)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def try_load_feat_cache(path, cache_dir=FEAT_CACHE_DIR):
    if not cache_dir:
        return None
    key = file_key(path)
    npz_path = os.path.join(cache_dir, key + '.npz')
    if not os.path.isfile(npz_path):
        return None
    try:
        d = np.load(npz_path, allow_pickle=False)
        x = d['x']
        if x.shape != (42,):
            return None
        return np.asarray(x, dtype=np.float32)
    except Exception:
        return None


def load_models(model_dir):
    """返回 (models, meta)。

    model_type == 'single_lgb'  -> models 是 [LGBMRegressor, ...]
    其他（v1 旧格式）           -> models 是 [(low, a, gate), ...]
    """
    meta_path = os.path.join(model_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f'meta.json not found in {model_dir}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    model_type = meta.get('model_type', 'dual_expert_gate')
    n = int(meta.get('n_ensemble', 1))
    models = []

    if model_type == 'single_lgb':
        if n == 1:
            models.append(joblib.load(os.path.join(model_dir, 'model.pkl')))
        else:
            for i in range(n):
                models.append(joblib.load(os.path.join(model_dir, f'model_{i}.pkl')))
        return models, meta

    if model_type == 'two_stage_lgb':
        # 返回两层：models = ([stage1...], [stage2...])
        def _load(sub):
            out = []
            d = os.path.join(model_dir, sub)
            if n == 1:
                out.append(joblib.load(os.path.join(d, 'model.pkl')))
            else:
                for i in range(n):
                    out.append(joblib.load(os.path.join(d, f'model_{i}.pkl')))
            return out
        return (_load('stage1'), _load('stage2')), meta

    # ---- v1 旧格式（双专家 + 门控），保留兼容以便回滚 ----
    if n == 1:
        models.append((
            joblib.load(os.path.join(model_dir, 'low_model.pkl')),
            joblib.load(os.path.join(model_dir, 'a_model.pkl')),
            joblib.load(os.path.join(model_dir, 'gate_model.pkl'))))
    else:
        for i in range(n):
            models.append((
                joblib.load(os.path.join(model_dir, f'low_model_{i}.pkl')),
                joblib.load(os.path.join(model_dir, f'a_model_{i}.pkl')),
                joblib.load(os.path.join(model_dir, f'gate_model_{i}.pkl'))))
    return models, meta


def _f6(low_p, a_p):
    d = a_p - low_p
    return np.column_stack([low_p, a_p, d, np.abs(d),
                            np.minimum(low_p, a_p),
                            np.maximum(low_p, a_p)]).astype(np.float32)


def gate_feat_new(X42, low_p, a_p):
    return np.hstack([X42, _f6(low_p, a_p)]).astype(np.float32)


def predict_one(X42, models, tau, model_type='dual_expert_gate'):
    """返回 (pred, low_p, a_p, prob, route_a, spread)。

    单模型模式下 low_p = a_p = pred，prob = 0，route_a = False，
    spread = N 个 seed 预测的标准差（>1 个模型时有意义，否则 0）。
    """
    X = X42.reshape(1, -1).astype(np.float32)

    if model_type == 'single_lgb':
        preds = [float(m.predict(X)[0]) for m in models]
        pred = float(np.mean(preds))
        spread = float(np.std(preds)) if len(preds) > 1 else 0.0
        return pred, pred, pred, 0.0, False, spread

    lows, aas, probs = [], [], []
    for low, a, gate in models:
        lp = float(low.predict(X)[0])
        ap = float(a.predict(X)[0])
        G = gate_feat_new(X, np.array([lp]), np.array([ap]))
        pr = float(gate.predict_proba(G)[:, 1][0])
        lows.append(lp); aas.append(ap); probs.append(pr)
    low_p = float(np.mean(lows))
    a_p = float(np.mean(aas))
    prob = float(np.mean(probs))
    route_a = prob > tau
    pred = a_p if route_a else low_p
    spread = float(np.std(aas)) if len(aas) > 1 else 0.0
    return pred, low_p, a_p, prob, route_a, spread


# ============================================================
# 标签写入 —— 完全照搬 write_rg_fast.py
# ============================================================
def has_replaygain(path):
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.m4a', '.mp4', '.m4b', '.m4r'):
            audio = MP4(path)
            if audio.tags is None:
                return False
            return '----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN' in audio.tags
        if ext == '.mp3':
            try:
                tags = ID3(path)
            except ID3NoHeaderError:
                return False
            for k in tags.keys():
                if k.upper() == 'TXXX:REPLAYGAIN_TRACK_GAIN':
                    return True
            return False
        audio = mutagen.File(path, easy=True)
        if audio is None:
            return False
        for k in (audio.tags or {}):
            if k.upper() == 'REPLAYGAIN_TRACK_GAIN':
                return True
        return False
    except Exception:
        return False


def write_replaygain_tags(filepath, track_gain, track_peak_db, write_r128=False):
    """完全照搬 write_rg_fast.py 的写入逻辑。"""
    ext = os.path.splitext(filepath)[1].lower()
    gain_str = f"{track_gain:+.2f} dB"
    peak_linear = 10 ** (track_peak_db / 20.0)
    peak_str = f"{peak_linear:.6f}"

    try:
        if ext == '.flac':
            audio = FLAC(filepath)
            audio['replaygain_track_gain'] = gain_str
            audio['replaygain_track_peak'] = peak_str
            audio.save()
        elif ext == '.mp3':
            audio = MP3(filepath, ID3=ID3)
            audio.tags.delall('TXXX:REPLAYGAIN_TRACK_GAIN')
            audio.tags.delall('TXXX:REPLAYGAIN_TRACK_PEAK')
            audio.tags.add(TXXX(encoding=3, desc='REPLAYGAIN_TRACK_GAIN', text=gain_str))
            audio.tags.add(TXXX(encoding=3, desc='REPLAYGAIN_TRACK_PEAK', text=peak_str))
            audio.save()
        elif ext in ('.ogg', '.oga'):
            audio = OggVorbis(filepath)
            audio['replaygain_track_gain'] = gain_str
            audio['replaygain_track_peak'] = peak_str
            audio.save()
        elif ext == '.opus':
            audio = OggOpus(filepath)
            audio['replaygain_track_gain'] = gain_str
            audio['replaygain_track_peak'] = peak_str
            if write_r128:
                r128 = int(round(track_gain * 256))
                audio['R128_TRACK_GAIN'] = [str(r128)]
            audio.save()
        elif ext in ('.mp4', '.m4a', '.m4b', '.m4r'):
            audio = MP4(filepath)
            audio['----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN'] = [gain_str.encode('utf-8')]
            audio['----:com.apple.iTunes:REPLAYGAIN_TRACK_PEAK'] = [peak_str.encode('utf-8')]
            audio.save()
        else:
            return False, f'unsupported: {ext}'
        return True, 'ok'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


# ============================================================
# 扫描 / 分类
# ============================================================
def scan_audio(root):
    root = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in AUDIO_EXTS:
                yield os.path.join(dirpath, fn)


def classify_confidence(pred, ext_thr, high_thr, low_thr):
    if pred >= ext_thr:
        return 'extreme'
    if pred >= high_thr:
        return 'high'
    if pred >= low_thr:
        return 'low'
    return 'normal'


# ============================================================
# Worker
# ============================================================
_W = {}


def _worker_init(model_dir, feat_cache_dir, shift):
    _W['models'], _W['meta'] = load_models(model_dir)
    _W['model_type'] = _W['meta'].get('model_type', 'dual_expert_gate')
    _W['tau'] = float(_W['meta'].get('tau', 0.0))
    _W['feat_cache_dir'] = feat_cache_dir
    _W['watch_thr'] = WATCH_THR_BASE + shift
    _W['conf_extreme'] = CONF_EXTREME_BASE + shift
    _W['conf_high'] = CONF_HIGH_BASE + shift
    _W['conf_low'] = CONF_LOW_BASE + shift


def _worker_extract(task):
    """两阶段模式专用：只负责把 42 维特征取出来（缓存优先）。

    两阶段的组信号必须等全库基础预测齐了才能算，所以把「取特征」和
    「出预测」拆成两趟：本函数只取特征，预测在主进程统一做。
    """
    fp = task
    try:
        x = try_load_feat_cache(fp, _W.get('feat_cache_dir'))
        from_cache = x is not None
        if not from_cache:
            x = extract_features(fp)
            if x is None:
                return {'path': fp, 'ok': False,
                        'status': 'err:extract_features returned None'}
            x = np.asarray(x, dtype=np.float32)
            if x.shape != (42,):
                return {'path': fp, 'ok': False,
                        'status': f'err:shape {x.shape}'}
        return {'path': fp, 'ok': True, 'status': '', 'x': x,
                'from_cache': from_cache}
    except Exception as e:
        return {'path': fp, 'ok': False,
                'status': f'err:{type(e).__name__}: {e}',
                'trace': traceback.format_exc()}


def _worker_task(task):
    fp, skip_tagged, dry_run, write_r128, peak_override = task
    if skip_tagged and has_replaygain(fp):
        return {'path': fp, 'status': 'skipped'}
    try:
        x = try_load_feat_cache(fp, _W.get('feat_cache_dir'))
        from_cache = x is not None
        if not from_cache:
            x = extract_features(fp)
            if x is None:
                return {'path': fp, 'status': 'err:extract_features returned None'}
            x = np.asarray(x, dtype=np.float32)
            if x.shape != (42,):
                return {'path': fp, 'status': f'err:shape {x.shape}'}

        pred, low_p, a_p, prob, route_a, spread = predict_one(
            x, _W['models'], _W['tau'], _W['model_type'])

        # true_peak（dB），即特征第 2 维
        try:
            true_peak_db = float(x[1])
            if not np.isfinite(true_peak_db):
                true_peak_db = -100.0
        except Exception:
            true_peak_db = -100.0

        if peak_override is not None:
            # 用户手动指定线性峰值，转回 dB
            true_peak_db = 20.0 * np.log10(max(float(peak_override), 1e-9))

        watch = int(pred > _W['watch_thr'])
        conf = classify_confidence(pred, _W['conf_extreme'],
                                   _W['conf_high'], _W['conf_low'])
        if dry_run:
            status = 'dry-run'; ok = True
        else:
            ok, msg = write_replaygain_tags(fp, pred, true_peak_db, write_r128)
            status = 'ok' if ok else f'err:{msg}'
        return {'path': fp, 'y_pred': pred, 'spread': spread,
                'low_pred': low_p, 'a_pred': a_p,
                'prob': prob, 'route_a': int(route_a), 'watch': watch,
                'confidence': conf, 'status': status, 'ok': ok,
                'from_cache': from_cache}
    except Exception as e:
        return {'path': fp, 'status': f'err:{type(e).__name__}: {e}',
                'ok': False, 'trace': traceback.format_exc()}


# ============================================================
# Main
# ============================================================
def _run_two_stage(args, files, models, meta, feat_cache_dir, shift,
                   watch_thr, log_writer, log_fh, n_workers):
    """两阶段推理：先全量取特征/基础预测，再算组信号，最后出最终预测。

    组的划分来自 library_meta.csv（目录/专辑/艺人）。缺元数据的歌退化为
    「按所在目录分组」，不会报错。
    """
    import pandas as pd

    stage1, stage2 = models
    use_keys = tuple(meta.get('group_signal_keys') or ('dir', 'album', 'artist'))
    meta_file = meta.get('meta_file', 'library_meta.csv')

    # 组键表：路径 -> (dir, album, artist)
    gmeta = {}
    if os.path.exists(meta_file):
        d = pd.read_csv(meta_file, encoding='utf-8-sig')
        for _, r in d.iterrows():
            k = str(r['path']).replace('\\', '/')
            while '//' in k:
                k = k.replace('//', '/')
            if len(k) > 1 and k[1] == ':':
                k = k[0].lower() + k[1:]
            gmeta[k] = (r['dir_key'], r['album'], r['artist'])
        print(f'  元数据      = {meta_file}  ({len(gmeta)} 条)')
    else:
        print(f'  [WARN] 找不到 {meta_file}，组信号退化为按目录分组')

    def _norm(p):
        k = str(p).replace('\\', '/')
        while '//' in k:
            k = k.replace('//', '/')
        if len(k) > 1 and k[1] == ':':
            k = k[0].lower() + k[1:]
        return k

    def _keys_of(p):
        k = _norm(p)
        if k in gmeta:
            return gmeta[k]
        d = _norm(os.path.dirname(os.path.abspath(p)))
        nm = os.path.basename(os.path.dirname(p)) or d
        return (d, nm, nm)

    # 需要写标签的才处理；跳过的直接记一行
    todo = []
    for fp in files:
        if args.skip_tagged and has_replaygain(fp):
            log_writer.writerow([fp] + [''] * 9 + ['skipped'])
            continue
        todo.append(fp)
    n_skip = len(files) - len(todo)
    total = len(todo)
    print(f'需处理 {total} 首（跳过 {n_skip}）')
    print('=' * 70)
    if total == 0:
        log_fh.close()
        return

    # ---- 1. 取特征 + 一阶段基础预测 ----
    print('阶段 1/3：提取特征 + 基础预测 ...')
    xs = {}
    base = {}
    fails = {}
    n_cache_hit = 0
    done = 0
    with mp.Pool(processes=n_workers, initializer=_worker_init,
                 initargs=(args.model_dir, feat_cache_dir, shift)) as pool:
        for res in pool.imap_unordered(_worker_extract, todo, chunksize=1):
            done += 1
            fp = res['path']
            if not res['ok']:
                fails[fp] = res['status']
            else:
                x = res['x']
                xs[fp] = x
                if res.get('from_cache'):
                    n_cache_hit += 1
                X = x.reshape(1, -1).astype(np.float32)
                base[fp] = float(np.mean([m.predict(X)[0] for m in stage1]))
            if done % 50 == 0 or done == total:
                print(f'  [{done}/{total}]  已取 {len(xs)}  失败 {len(fails)}')

    if not xs:
        print('没有任何可用样本，退出。')
        log_fh.close()
        return

    # ---- 2. 组信号 ----
    print('\n阶段 2/3：计算组信号 ...')
    paths = list(xs.keys())
    keys = {k: [] for k in use_keys}
    for p in paths:
        arr = _keys_of(p)
        for i, k in enumerate(('dir', 'album', 'artist')):
            if k in keys:
                keys[k].append(arr[i] if i < len(arr) else '')
    bp = np.array([base[p] for p in paths], np.float64)
    gcols = []
    for k in use_keys:
        s = pd.Series(keys[k])
        mean = np.full(len(paths), -9.0)
        cnt = np.zeros(len(paths))
        for _, idx in s.groupby(s, sort=False).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) <= 1:
                continue
            tot = bp[idx].sum()
            for i in idx:
                mean[i] = (tot - bp[i]) / (len(idx) - 1)
                cnt[i] = len(idx) - 1
        gcols += [mean, cnt]
        print(f'  {k:<7} 组数 {s.nunique():>4}  用过组信号的样本 {int((cnt>0).sum())}/{len(paths)}')
    G = np.column_stack(gcols) if gcols else np.zeros((len(paths), 0))

    # ---- 3. 二阶段预测 + 写标签 ----
    print('\n阶段 3/3：最终预测 + 写标签 ...')
    Xall = np.vstack([xs[p] for p in paths])
    Wall = np.hstack([Xall, G])
    preds = [np.array([m.predict(Wall) for m in stage2])]  # (n_models, n)
    P = preds[0]
    final = P.mean(0)
    spread = P.std(0) if P.shape[0] > 1 else np.zeros(len(paths))

    n_ok = 0
    for i, fp in enumerate(paths):
        pred = float(final[i])
        tp = float(xs[fp][1])
        if not np.isfinite(tp):
            tp = -100.0
        if args.peak is not None:
            tp = 20.0 * np.log10(max(float(args.peak), 1e-9))
        watch = int(pred > watch_thr)
        conf = classify_confidence(pred, CONF_EXTREME_BASE + shift,
                                   CONF_HIGH_BASE + shift,
                                   CONF_LOW_BASE + shift)
        if args.dry_run:
            status, ok = 'dry-run', True
        else:
            ok, msg = write_replaygain_tags(fp, pred, tp, args.write_r128)
            status = 'ok' if ok else f'err:{msg}'
        if ok:
            n_ok += 1
        log_writer.writerow([fp, f'{pred:.4f}', f'{spread[i]:.4f}',
                             '', '', '', '', watch, conf,
                             1, status])
        if watch:
            print(f'  [WATCH] pred={pred:7.3f}  spread={spread[i]:.3f}  '
                  f'conf={conf}  {os.path.basename(fp)}')
        log_fh.flush()
    for fp, st in fails.items():
        log_writer.writerow([fp] + [''] * 9 + [st])
        print(f'  [ERR] {fp}: {st}')

    print('=' * 70)
    print(f'Done.  ok={n_ok}  fail={len(fails)}  skip={n_skip}  '
          f'cache_hit={n_cache_hit}/{total}')
    print(f'最终预测来自两阶段模型（42维 + {G.shape[1]} 维组信号）')
    log_fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--model-dir', default=MODEL_DIR)
    ap.add_argument('--log-dir', default=LOG_DIR)
    ap.add_argument('--feat-cache', default=FEAT_CACHE_DIR)
    ap.add_argument('--no-feat-cache', action='store_true')
    ap.add_argument('--skip-tagged', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--write-r128', action='store_true')
    ap.add_argument('--peak', type=float, default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--shift', type=float, default=None)
    ap.add_argument('--only-ext', type=str, default='')
    ap.add_argument('--exclude-ext', type=str, default='')
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f'predict_{ts}.csv')
    log_fh = open(log_path, 'w', encoding='utf-8', newline='')
    log_writer = csv.writer(log_fh)
    log_writer.writerow(['path', 'y_pred', 'spread', 'low_pred', 'a_pred',
                         'prob', 'route_a', 'watch', 'confidence',
                         'from_cache', 'status'])

    print('=' * 70)
    print(f'Loading models from: {args.model_dir}')
    models, meta = load_models(args.model_dir)
    model_type = meta.get('model_type', 'dual_expert_gate')
    single_lgb = (model_type == 'single_lgb')
    tau = float(meta.get('tau', 0.0))
    meta_shift = float(meta.get('label_shift', 0.0))
    shift = args.shift if args.shift is not None else meta_shift
    oof = meta.get('oof_stats') or {}

    watch_thr = WATCH_THR_BASE + shift
    print(f'  model_type  = {model_type}')
    print(f'  n_ensemble  = {meta.get("n_ensemble", 1)}')
    if model_type != 'single_lgb':
        print(f'  tau         = {tau:.4f}')
    print(f'  label_shift = {shift:+.2f}')
    if oof:
        print(f'  OOF 指标    = MAE {oof.get("mae", float("nan")):.3f}  '
              f'>2dB {oof.get("gt2", "?")} 首 / {oof.get("n", "?")} 首  '
              f'max {oof.get("max", float("nan")):.2f}')
    print(f'  WATCH_THR   = {watch_thr:+.2f}')
    print(f'  decoder     = {"ffmpeg" if ffmpeg_available() else "soundfile/librosa"}')
    print(f'  MAX_DURATION= {MAX_DURATION_SEC}s')

    feat_cache_dir = None if args.no_feat_cache else args.feat_cache
    if feat_cache_dir and os.path.isdir(feat_cache_dir):
        n_npz = sum(1 for f in os.listdir(feat_cache_dir) if f.endswith('.npz'))
        print(f'  feat_cache  = {feat_cache_dir}  ({n_npz} files)')
    else:
        feat_cache_dir = None

    files = list(scan_audio(args.input))

    def _parse_exts(s):
        exts = set()
        for e in s.split(','):
            e = e.strip().lower()
            if not e:
                continue
            if not e.startswith('.'):
                e = '.' + e
            exts.add(e)
        return exts

    if args.only_ext:
        exts = _parse_exts(args.only_ext)
        if exts:
            before = len(files)
            files = [f for f in files if os.path.splitext(f)[1].lower() in exts]
            print(f'[FILTER] only-ext {sorted(exts)}: {before} -> {len(files)}')
    if args.exclude_ext:
        exts = _parse_exts(args.exclude_ext)
        if exts:
            before = len(files)
            files = [f for f in files if os.path.splitext(f)[1].lower() not in exts]
            print(f'[FILTER] exclude-ext {sorted(exts)}: {before} -> {len(files)}')

    if args.limit > 0:
        files = files[:args.limit]
    total = len(files)
    print(f'Found {total} audio files.')
    if total == 0:
        print('Nothing to do.')
        log_fh.close()
        return

    if args.workers > 0:
        n_workers = args.workers
    else:
        # librosa 的 HPSS 和长音频解码很吃内存。32 核全开实测会把内存打爆
        # （31GB 物理内存、可用 9.8GB 时，16 workers 就会 MemoryError）。
        n_workers = max(1, min(os.cpu_count() or 4, 8))
    n_workers = min(n_workers, total)
    print(f'Workers: {n_workers}')
    print('=' * 70)

    tasks = [(fp, args.skip_tagged, args.dry_run, args.write_r128, args.peak)
             for fp in files]
    n_ok = n_fail = n_skip = n_watch = 0
    n_cache_hit = 0
    done = 0

    if model_type == 'two_stage_lgb':
        _run_two_stage(args, files, models, meta, feat_cache_dir, shift,
                       watch_thr, log_writer, log_fh, n_workers)
        return

    try:
        with mp.Pool(processes=n_workers, initializer=_worker_init,
                     initargs=(args.model_dir, feat_cache_dir, shift)) as pool:
            for res in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                done += 1
                fp = res['path']
                if res.get('status') == 'skipped':
                    n_skip += 1
                    log_writer.writerow([fp] + [''] * 9 + ['skipped'])
                elif res.get('ok'):
                    n_ok += 1
                    if res.get('from_cache'):
                        n_cache_hit += 1
                    if single_lgb:
                        # 单模型模式没有专家/门控，这几列留空
                        mid = ['', '', '', '']
                    else:
                        mid = [f'{res["low_pred"]:.4f}',
                               f'{res["a_pred"]:.4f}',
                               f'{res["prob"]:.4f}', res['route_a']]
                    log_writer.writerow([
                        fp, f'{res["y_pred"]:.4f}', f'{res.get("spread", 0):.4f}',
                        *mid,
                        res['watch'], res['confidence'],
                        int(res.get('from_cache', 0)), res['status']])
                    if res['watch']:
                        n_watch += 1
                        extra = ('' if res.get('route_a') is None and
                                 not res.get('prob') else
                                 f'  route={"A" if res["route_a"] else "L"}'
                                 f'  prob={res["prob"]:.3f}')
                        print(f'  [WATCH] pred={res["y_pred"]:7.3f}'
                              f'  spread={res.get("spread", 0):.3f}'
                              f'{extra}  conf={res["confidence"]}  '
                              f'{os.path.basename(fp)}')
                else:
                    n_fail += 1
                    log_writer.writerow([fp] + [''] * 9 + [res['status']])
                    print(f'  [ERR] {fp}: {res["status"]}')
                    if 'trace' in res:
                        print(res['trace'])
                if done % 50 == 0 or done == total:
                    print(f'  [{done}/{total}]  ok={n_ok}  fail={n_fail}'
                          f'  skip={n_skip}  watch={n_watch}  '
                          f'cache_hit={n_cache_hit}')
                log_fh.flush()
    except KeyboardInterrupt:
        print('\n[INTERRUPTED]')
    finally:
        log_fh.close()

    print('=' * 70)
    print(f'Done.  ok={n_ok}  fail={n_fail}  skip={n_skip}  watch={n_watch}')
    if n_ok > 0:
        print(f'       缓存命中 {n_cache_hit}/{n_ok} '
              f'({n_cache_hit * 100 // max(n_ok, 1)}%)')
    print(f'Log -> {log_path}')


if __name__ == '__main__':
    main()