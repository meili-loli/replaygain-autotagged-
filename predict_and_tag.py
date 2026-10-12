# predict_and_tag.py
# -*- coding: utf-8 -*-
"""
批量预测 ReplayGain 并写标签。多进程 + feat_cache + label_shift + 全格式写入。

写入策略：
  - MP3：ID3 TXXX frame（大写 REPLAYGAIN_TRACK_GAIN，foobar2000 识别）
  - FLAC：Vorbis Comment
  - OGG：Vorbis Comment
  - Opus：Vorbis Comment + 可选 R128_TRACK_GAIN
  - M4A/MP4：MP4 freeform atom (com.apple.iTunes)

只处理指定扩展名：
  python predict_and_tag.py --input "..." --write-r128 --only-ext .mp3
"""

import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
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
    import mutagen
except ImportError:
    sys.exit("请先 pip install mutagen")


# ============================================================
# 特征提取（内嵌，ffmpeg 优先，和 train 完全一致）
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

RG_KEYS = ('REPLAYGAIN_TRACK_GAIN', 'REPLAYGAIN_TRACK_PEAK',
           'REPLAYGAIN_ALBUM_GAIN', 'REPLAYGAIN_ALBUM_PEAK')


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
    meta_path = os.path.join(model_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f'meta.json not found in {model_dir}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    low = joblib.load(os.path.join(model_dir, 'low_model.pkl'))
    a = joblib.load(os.path.join(model_dir, 'a_model.pkl'))
    gate = joblib.load(os.path.join(model_dir, 'gate_model.pkl'))
    return low, a, gate, meta


def _f6(low_p, a_p):
    d = a_p - low_p
    return np.column_stack([low_p, a_p, d, np.abs(d),
                            np.minimum(low_p, a_p),
                            np.maximum(low_p, a_p)]).astype(np.float32)


def gate_feat_new(X42, low_p, a_p):
    return np.hstack([X42, _f6(low_p, a_p)]).astype(np.float32)


def predict_one(X42, low, a, gate, tau):
    X = X42.reshape(1, -1).astype(np.float32)
    low_p = float(low.predict(X)[0])
    a_p = float(a.predict(X)[0])
    G = gate_feat_new(X, np.array([low_p]), np.array([a_p]))
    prob = float(gate.predict_proba(G)[:, 1][0])
    route_a = prob > tau
    pred = a_p if route_a else low_p
    return pred, low_p, a_p, prob, route_a


# ============================================================
# ReplayGain 写入（全格式，MP3 用 TXXX）
# ============================================================
def has_replaygain(path):
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.m4a', '.mp4', '.m4b', '.m4r'):
            from mutagen.mp4 import MP4
            audio = MP4(path)
            if audio.tags is None:
                return False
            return '----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN' in audio.tags
        if ext == '.mp3':
            from mutagen.id3 import ID3, ID3NoHeaderError
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


def write_replaygain(path, gain_db, peak=None, write_r128=False):
    """
    MP3       → ID3 TXXX frame（大写，foobar2000 识别）
    FLAC      → Vorbis Comment
    OGG       → Vorbis Comment
    Opus      → Vorbis Comment + 可选 R128_TRACK_GAIN
    M4A / MP4 → freeform atom
    """
    try:
        ext = os.path.splitext(path)[1].lower()

        if ext in ('.m4a', '.mp4', '.m4b', '.m4r'):
            from mutagen.mp4 import MP4, MP4FreeForm
            audio = MP4(path)
            if audio.tags is None:
                audio.add_tags()
            audio.tags['----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN'] = [
                MP4FreeForm(f'{gain_db:.2f} dB'.encode('utf-8'))]
            if peak is not None:
                audio.tags['----:com.apple.iTunes:REPLAYGAIN_TRACK_PEAK'] = [
                    MP4FreeForm(f'{peak:.6f}'.encode('utf-8'))]
            audio.save()
            return True, 'ok (mp4)'

        if ext == '.mp3':
            from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
            try:
                tags = ID3(path)
            except ID3NoHeaderError:
                tags = ID3()
            for k in list(tags.keys()):
                if k.upper().startswith('TXXX:'):
                    desc = k.split(':', 1)[1].upper() if ':' in k else ''
                    if desc in RG_KEYS:
                        del tags[k]
            tags.add(TXXX(encoding=3, desc='REPLAYGAIN_TRACK_GAIN',
                          text=f'{gain_db:.2f} dB'))
            if peak is not None:
                tags.add(TXXX(encoding=3, desc='REPLAYGAIN_TRACK_PEAK',
                              text=f'{peak:.6f}'))
            tags.save(path)
            return True, 'ok (mp3 id3)'

        if ext == '.flac':
            from mutagen.flac import FLAC
            audio = FLAC(path)
            audio['REPLAYGAIN_TRACK_GAIN'] = f'{gain_db:.2f} dB'
            if peak is not None:
                audio['REPLAYGAIN_TRACK_PEAK'] = f'{peak:.6f}'
            audio.save()
            return True, 'ok (flac)'

        if ext == '.ogg':
            from mutagen.oggvorbis import OggVorbis
            audio = OggVorbis(path)
            audio['REPLAYGAIN_TRACK_GAIN'] = f'{gain_db:.2f} dB'
            if peak is not None:
                audio['REPLAYGAIN_TRACK_PEAK'] = f'{peak:.6f}'
            audio.save()
            return True, 'ok (ogg)'

        if ext == '.opus':
            from mutagen.oggopus import OggOpus
            audio = OggOpus(path)
            audio['REPLAYGAIN_TRACK_GAIN'] = f'{gain_db:.2f} dB'
            if peak is not None:
                audio['REPLAYGAIN_TRACK_PEAK'] = f'{peak:.6f}'
            if write_r128:
                r128 = int(round(gain_db * 256))
                audio['R128_TRACK_GAIN'] = [str(r128)]
            audio.save()
            return True, 'ok (opus)'

        audio = mutagen.File(path, easy=True)
        if audio is None:
            return False, 'unsupported format'
        if audio.tags is None:
            try:
                audio.add_tags()
            except Exception:
                pass
        audio['REPLAYGAIN_TRACK_GAIN'] = f'{gain_db:.2f} dB'
        if peak is not None:
            audio['REPLAYGAIN_TRACK_PEAK'] = f'{peak:.6f}'
        audio.save()
        return True, 'ok (fallback)'
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
    _W['low'], _W['a'], _W['gate'], _W['meta'] = load_models(model_dir)
    _W['tau'] = float(_W['meta']['tau'])
    _W['feat_cache_dir'] = feat_cache_dir
    _W['watch_thr'] = WATCH_THR_BASE + shift
    _W['conf_extreme'] = CONF_EXTREME_BASE + shift
    _W['conf_high'] = CONF_HIGH_BASE + shift
    _W['conf_low'] = CONF_LOW_BASE + shift


def _worker_task(task):
    fp, skip_tagged, dry_run, write_r128, peak = task
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
        pred, low_p, a_p, prob, route_a = predict_one(
            x, _W['low'], _W['a'], _W['gate'], _W['tau'])
        watch = int(pred > _W['watch_thr'])
        conf = classify_confidence(pred, _W['conf_extreme'],
                                   _W['conf_high'], _W['conf_low'])
        if dry_run:
            status = 'dry-run'; ok = True
        else:
            ok, msg = write_replaygain(fp, pred, peak=peak,
                                       write_r128=write_r128)
            status = 'ok' if ok else f'err:{msg}'
        return {'path': fp, 'y_pred': pred, 'low_pred': low_p, 'a_pred': a_p,
                'prob': prob, 'route_a': int(route_a), 'watch': watch,
                'confidence': conf, 'status': status, 'ok': ok,
                'from_cache': from_cache}
    except Exception as e:
        return {'path': fp, 'status': f'err:{type(e).__name__}: {e}',
                'ok': False, 'trace': traceback.format_exc()}


# ============================================================
# Main
# ============================================================
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
    ap.add_argument('--only-ext', type=str, default='',
                    help='只处理这些扩展名，逗号分隔。例：.mp3,.m4a')
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f'predict_{ts}.csv')
    log_fh = open(log_path, 'w', encoding='utf-8', newline='')
    log_writer = csv.writer(log_fh)
    log_writer.writerow(['path', 'y_pred', 'low_pred', 'a_pred', 'prob',
                         'route_a', 'watch', 'confidence', 'from_cache', 'status'])

    print('=' * 70)
    print(f'Loading models from: {args.model_dir}')
    low, a, gate, meta = load_models(args.model_dir)
    tau = float(meta['tau'])
    gate_type = meta.get('gate_type', 'unknown')
    meta_shift = float(meta.get('label_shift', 0.0))
    shift = args.shift if args.shift is not None else meta_shift

    watch_thr = WATCH_THR_BASE + shift
    conf_extreme = CONF_EXTREME_BASE + shift
    conf_high = CONF_HIGH_BASE + shift
    conf_low = CONF_LOW_BASE + shift

    print(f'  gate_type   = {gate_type}')
    print(f'  tau         = {tau:.4f}')
    print(f'  label_shift = {shift:+.2f}')
    print(f'  WATCH_THR   = {watch_thr:+.2f}')
    print(f'  CONF_EXTREME= {conf_extreme:+.2f}')
    print(f'  CONF_HIGH   = {conf_high:+.2f}')
    print(f'  CONF_LOW    = {conf_low:+.2f}')
    print(f'  decoder     = {"ffmpeg" if ffmpeg_available() else "soundfile/librosa"}')
    if gate_type != 'binary_classifier':
        print('  [WARN] gate_type != binary_classifier')

    feat_cache_dir = None if args.no_feat_cache else args.feat_cache
    if feat_cache_dir and os.path.isdir(feat_cache_dir):
        n_npz = sum(1 for f in os.listdir(feat_cache_dir) if f.endswith('.npz'))
        print(f'  feat_cache  = {feat_cache_dir}  ({n_npz} files)')
    else:
        if feat_cache_dir:
            print(f'  feat_cache  = {feat_cache_dir}  (不存在)')
        feat_cache_dir = None

    files = list(scan_audio(args.input))

    # ★ 只处理指定扩展名
    if args.only_ext:
        exts = set()
        for e in args.only_ext.split(','):
            e = e.strip().lower()
            if not e:
                continue
            if not e.startswith('.'):
                e = '.' + e
            exts.add(e)
        if exts:
            before = len(files)
            files = [f for f in files
                     if os.path.splitext(f)[1].lower() in exts]
            print(f'[FILTER] --only-ext {sorted(exts)}: {before} -> {len(files)}')

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
        n_workers = max(1, os.cpu_count() or 4)
    n_workers = min(n_workers, total)
    print(f'Workers: {n_workers}')
    print('=' * 70)

    tasks = [(fp, args.skip_tagged, args.dry_run, args.write_r128, args.peak)
             for fp in files]
    n_ok = n_fail = n_skip = n_watch = 0
    n_cache_hit = 0
    watch_list = []
    done = 0

    try:
        with mp.Pool(processes=n_workers, initializer=_worker_init,
                     initargs=(args.model_dir, feat_cache_dir, shift)) as pool:
            for res in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                done += 1
                fp = res['path']
                if res.get('status') == 'skipped':
                    n_skip += 1
                    log_writer.writerow([fp] + [''] * 8 + ['skipped'])
                elif res.get('ok'):
                    n_ok += 1
                    if res.get('from_cache'):
                        n_cache_hit += 1
                    log_writer.writerow([
                        fp, f'{res["y_pred"]:.4f}', f'{res["low_pred"]:.4f}',
                        f'{res["a_pred"]:.4f}', f'{res["prob"]:.4f}',
                        res['route_a'], res['watch'], res['confidence'],
                        int(res.get('from_cache', 0)), res['status']])
                    if res['watch']:
                        n_watch += 1
                        watch_list.append((
                            fp, res['y_pred'], res['low_pred'], res['a_pred'],
                            res['prob'], res['route_a'], res['confidence']))
                        print(f'  [WATCH] pred={res["y_pred"]:7.3f}  '
                              f'route={"A" if res["route_a"] else "L"}  '
                              f'prob={res["prob"]:.3f}  '
                              f'conf={res["confidence"]}  '
                              f'{os.path.basename(fp)}')
                else:
                    n_fail += 1
                    log_writer.writerow([fp] + [''] * 8 + [res['status']])
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