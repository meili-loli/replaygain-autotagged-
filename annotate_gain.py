# annotate_gain.py
# -*- coding: utf-8 -*-
"""
Delta Gain 标注工具（ffmpeg + sounddevice）。
- ffmpeg 解码（和 feature_extractor 一致）
- 实时 dB 增益，A/B 对比
- 长音频分块加载
- 断点续传
"""

import os
import sys
import gc
import csv
import json
import shutil
import faulthandler
import threading
import subprocess
import argparse
import traceback
from datetime import datetime

_CRASH_LOG = open('annotate_crash.log', 'a', encoding='utf-8', buffering=1)
faulthandler.enable(_CRASH_LOG)
_CRASH_LOG.write(f'\n=== session start {datetime.now().isoformat()} ===\n')


def _excepthook(exc_type, exc_value, exc_tb):
    _CRASH_LOG.write(f'\n=== unhandled exception {datetime.now().isoformat()} ===\n')
    traceback.print_exception(exc_type, exc_value, exc_tb, file=_CRASH_LOG)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook

import numpy as np
import sounddevice as sd

try:
    import pyloudnorm as pyln
except ImportError:
    sys.exit("请先 pip install pyloudnorm")

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QListWidget, QListWidgetItem,
        QDoubleSpinBox, QSplitter, QMessageBox)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QShortcut, QKeySequence
except ImportError:
    sys.exit("请先 pip install PySide6")


AUDIO_EXTS = {'.flac', '.wav', '.ogg', '.opus', '.aiff', '.aif',
              '.mp3', '.m4a', '.mp4', '.m4b', '.aac', '.ape', '.wv'}

GAIN_STEP = 0.5
GAIN_MIN = -20.0
GAIN_MAX = 20.0
LUFS_SCAN_SEC = 120
BLOCKSIZE = 2048
GC_EVERY = 20
WINDOW_SEC = 600


def fmt_time(sec):
    sec = max(0, int(sec))
    return f'{sec // 60:02d}:{sec % 60:02d}'


def ffmpeg_available():
    return shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None


def ffprobe_info(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
         '-show_entries', 'stream=sample_rate,channels',
         '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True, timeout=30)
    vals = [v for v in r.stdout.strip().split('\n') if v.strip()]
    if len(vals) < 3:
        raise RuntimeError(f'ffprobe 无法读取: {os.path.basename(path)}')
    return int(vals[0]), int(vals[1]), float(vals[2])


def load_audio_ffmpeg(path, target_sr=None, mono=False,
                      start_sec=0.0, duration_sec=None):
    if not ffmpeg_available():
        raise RuntimeError('未找到 ffmpeg')
    src_sr, src_ch, _ = ffprobe_info(path)
    sr = target_sr if target_sr else src_sr
    ch = 1 if mono else src_ch

    cmd = ['ffmpeg', '-v', 'error', '-nostdin']
    if start_sec > 0:
        cmd += ['-ss', f'{start_sec:.3f}']
    cmd += ['-i', path]
    if duration_sec is not None:
        cmd += ['-t', f'{duration_sec:.3f}']
    cmd += ['-f', 'f32le', '-acodec', 'pcm_f32le',
            '-ac', str(ch), '-ar', str(sr), '-']
    proc = subprocess.run(cmd, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        msg = proc.stderr.decode('utf-8', 'ignore').strip()
        raise RuntimeError(f'ffmpeg 解码失败: {msg[:200]}')

    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError('ffmpeg 返回空数据')
    if ch == 1:
        out = data.copy()
    else:
        n = data.size // ch
        out = data[:n * ch].reshape(n, ch).copy()
    del proc, data
    return out, sr


def load_audio_for_playback(path):
    return load_audio_ffmpeg(path, target_sr=None, mono=False)


def compute_lufs(path, max_sec=LUFS_SCAN_SEC):
    try:
        data, sr = load_audio_ffmpeg(path, target_sr=44100, mono=True,
                                     start_sec=0.0, duration_sec=max_sec)
        meter = pyln.Meter(sr)
        val = float(meter.integrated_loudness(data))
        del data
        return val if np.isfinite(val) else None
    except Exception:
        return None


class PrecisePlayer:
    def __init__(self, window_sec=WINDOW_SEC):
        self.window_sec = window_sec
        self.path = None
        self.total_dur = 0.0
        self.src_sr = 44100
        self.channels = 1
        self.y = None
        self.sr = None
        self.window_start = 0.0
        self.pos = 0
        self.gain_db = 0.0
        self.playing = False
        self.stream = None
        self.lock = threading.Lock()
        self.eof = False

    def load(self, path):
        with self.lock:
            self.playing = False
            self.pos = 0
            self.eof = False
            self.y = None
        self.path = path
        self.src_sr, src_ch, self.total_dur = ffprobe_info(path)
        self._load_window(0.0)

    def _load_window(self, start_sec):
        start_sec = max(0.0, min(start_sec, max(0.0, self.total_dur - 0.1)))
        y, sr = load_audio_ffmpeg(self.path, target_sr=None, mono=False,
                                  start_sec=start_sec, duration_sec=self.window_sec)
        old = self.y
        self.y = y
        self.sr = sr
        self.channels = 1 if y.ndim == 1 else y.shape[1]
        self.window_start = start_sec
        del old
        self._ensure_stream()

    def _ensure_stream(self):
        if self.stream is not None:
            try:
                if (self.stream.samplerate == self.sr and
                        self.stream.channels == self.channels):
                    return
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr, channels=self.channels, dtype='float32',
                blocksize=BLOCKSIZE, callback=self._callback)
            self.stream.start()
        except Exception:
            if self.sr != 44100:
                import librosa
                if self.y.ndim == 1:
                    self.y = librosa.resample(self.y, orig_sr=self.sr, target_sr=44100)
                else:
                    self.y = np.stack([
                        librosa.resample(self.y[:, c], orig_sr=self.sr, target_sr=44100)
                        for c in range(self.y.shape[1])], axis=1)
                self.sr = 44100
                self.stream = sd.OutputStream(
                    samplerate=self.sr, channels=self.channels, dtype='float32',
                    blocksize=BLOCKSIZE, callback=self._callback)
                self.stream.start()
            else:
                raise

    def _callback(self, outdata, frames, time_info, status):
        if not self.playing or self.y is None:
            outdata.fill(0)
            return
        with self.lock:
            pos = self.pos
            gain = 10.0 ** (self.gain_db / 20.0)
            end = pos + frames
            if end >= len(self.y):
                chunk = self.y[pos:]
                n = len(chunk)
                if chunk.ndim == 1:
                    outdata[:n, 0] = chunk * gain
                    outdata[n:, 0] = 0
                else:
                    outdata[:n, :] = chunk * gain
                    outdata[n:, :] = 0
                self.pos = len(self.y)
                self.playing = False
                self.eof = True
                return
            chunk = self.y[pos:end]
            if chunk.ndim == 1:
                outdata[:, 0] = chunk * gain
            else:
                outdata[:, :] = chunk * gain
            self.pos = end

    def play(self):
        if self.y is None:
            return
        with self.lock:
            if self.pos >= len(self.y):
                return
            self.playing = True

    def pause(self):
        with self.lock:
            self.playing = False

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        with self.lock:
            self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def seek_sec(self, sec):
        if self.y is None or self.sr is None:
            return
        sec = max(0.0, min(sec, max(0.0, self.total_dur - 0.05)))
        win_end = self.window_start + len(self.y) / self.sr
        if self.window_start <= sec < win_end:
            with self.lock:
                self.pos = int((sec - self.window_start) * self.sr)
                self.eof = False
        else:
            new_start = max(0.0, sec - self.window_sec * 0.1)
            self._load_window(new_start)
            with self.lock:
                self.pos = int((sec - new_start) * self.sr)
                self.eof = False

    def seek_rel(self, delta_sec):
        self.seek_sec(self.pos_sec + delta_sec)

    def set_gain_db(self, db):
        self.gain_db = float(db)

    @property
    def duration_sec(self):
        return self.total_dur

    @property
    def pos_sec(self):
        if self.sr is None:
            return 0.0
        with self.lock:
            return self.window_start + self.pos / self.sr


class AnnotateWindow(QMainWindow):
    def __init__(self, audio_dir, csv_path, progress_path, lufs_cache_path):
        super().__init__()
        self.audio_dir = audio_dir
        self.csv_path = csv_path
        self.progress_path = progress_path
        self.lufs_cache_path = lufs_cache_path

        self.files = self.scan_audio(audio_dir)
        if not self.files:
            QMessageBox.critical(self, '错误', f'未找到音频文件：{audio_dir}')
            sys.exit(1)

        self.progress = self.load_progress()
        self.current_idx = max(0, min(self.progress.get('current_idx', 0),
                                      len(self.files) - 1))
        self.annotations = self.progress.get('annotations', {})
        self.lufs_cache = self.load_lufs_cache()

        self.player = PrecisePlayer(window_sec=WINDOW_SEC)
        self.listen_original = False
        self.seeking = False
        self._loading = False
        self._save_counter = 0

        self.init_ui()
        self.build_list_once()
        self.load_current()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(200)

    def scan_audio(self, root):
        files = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.join(dp, fn))
        files.sort()
        return files

    def load_progress(self):
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def load_lufs_cache(self):
        if os.path.exists(self.lufs_cache_path):
            try:
                with open(self.lufs_cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_lufs_cache(self):
        try:
            with open(self.lufs_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.lufs_cache, f, ensure_ascii=False)
        except Exception:
            pass

    def save_progress(self):
        data = {
            'current_idx': self.current_idx,
            'annotations': self.annotations,
            'audio_dir': self.audio_dir,
            'saved_at': datetime.now().isoformat(),
            'total': len(self.files),
            'done': len(self.annotations),
        }
        with open(self.progress_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.export_csv()
        self.update_status()

    def export_csv(self):
        with open(self.csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'delta_gain'])
            for path, gain in sorted(self.annotations.items()):
                w.writerow([os.path.basename(path), gain])

    def get_lufs(self, path):
        if path in self.lufs_cache:
            v = self.lufs_cache[path]
            return v if v is None or isinstance(v, (int, float)) else None
        v = compute_lufs(path)
        self.lufs_cache[path] = v
        self.save_lufs_cache()
        return v

    def init_ui(self):
        self.setWindowTitle('Delta Gain 标注工具（ffmpeg）')
        self.resize(1150, 800)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.info_label = QLabel()
        self.info_label.setFont(QFont('Consolas', 11))
        layout.addWidget(self.info_label)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        self.list_widget = QListWidget()
        self.list_widget.setFont(QFont('Microsoft YaHei', 10))
        self.list_widget.currentRowChanged.connect(self.on_list_select)
        splitter.addWidget(self.list_widget)

        right = QWidget()
        rl = QVBoxLayout(right)

        self.file_label = QLabel()
        self.file_label.setWordWrap(True)
        self.file_label.setFont(QFont('Microsoft YaHei', 13, QFont.Bold))
        rl.addWidget(self.file_label)

        self.time_label = QLabel('00:00 / 00:00')
        self.time_label.setFont(QFont('Consolas', 16))
        rl.addWidget(self.time_label)

        self.lufs_label = QLabel()
        self.lufs_label.setFont(QFont('Consolas', 13))
        self.lufs_label.setStyleSheet(
            'background: #263238; color: #B2FF59; padding: 10px; border-radius: 4px;')
        rl.addWidget(self.lufs_label)

        self.window_label = QLabel('')
        self.window_label.setFont(QFont('Consolas', 10))
        self.window_label.setStyleSheet('color: #888;')
        rl.addWidget(self.window_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderPressed.connect(self.on_slider_press)
        self.slider.sliderMoved.connect(self.on_slider_move)
        self.slider.sliderReleased.connect(self.on_slider_release)
        rl.addWidget(self.slider)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton('播放 / 暂停 (Space)')
        self.btn_play.clicked.connect(self.toggle_play)
        btn_row.addWidget(self.btn_play)
        for label, delta in [('-30s', -30), ('-5s', -5), ('+5s', 5), ('+30s', 30)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, d=delta: self.seek_rel(d))
            btn_row.addWidget(b)
        rl.addLayout(btn_row)

        ab_row = QHBoxLayout()
        self.btn_ab = QPushButton('试听: 调整后 (T 切换 A/B)')
        self.btn_ab.setStyleSheet(
            'background: #1976D2; color: white; padding: 8px; font-size: 13px;')
        self.btn_ab.clicked.connect(self.toggle_ab)
        ab_row.addWidget(self.btn_ab)
        self.gain_label_small = QLabel('gain: 0.00 dB')
        self.gain_label_small.setFont(QFont('Consolas', 10))
        self.gain_label_small.setStyleSheet('color: #666;')
        ab_row.addWidget(self.gain_label_small)
        rl.addLayout(ab_row)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel('delta_gain:'))
        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(GAIN_MIN, GAIN_MAX)
        self.gain_input.setSingleStep(GAIN_STEP)
        self.gain_input.setDecimals(2)
        self.gain_input.setValue(0.0)
        self.gain_input.setFont(QFont('Consolas', 18))
        self.gain_input.setMinimumWidth(150)
        self.gain_input.valueChanged.connect(self.on_gain_changed)
        gain_row.addWidget(self.gain_input)
        for val in [-2.0, -1.0, -0.5, +0.5, +1.0, +2.0]:
            b = QPushButton(f'{val:+.1f}')
            b.clicked.connect(lambda _, v=val: self.adjust_gain(v))
            gain_row.addWidget(b)
        rl.addLayout(gain_row)

        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton('← 上一首 (P)')
        self.btn_prev.clicked.connect(self.prev_song)
        nav_row.addWidget(self.btn_prev)
        self.btn_save = QPushButton('保存并下一首 (S)')
        self.btn_save.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.btn_save.setStyleSheet(
            'background: #4CAF50; color: white; padding: 10px; font-size: 14px;')
        self.btn_save.clicked.connect(self.save_and_next)
        nav_row.addWidget(self.btn_save, 2)
        self.btn_skip = QPushButton('跳过 (N)')
        self.btn_skip.clicked.connect(self.skip)
        nav_row.addWidget(self.btn_skip)
        rl.addLayout(nav_row)

        rl.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([380, 770])

        QShortcut(QKeySequence('Space'), self, self.toggle_play)
        QShortcut(QKeySequence('S'), self, self.save_and_next)
        QShortcut(QKeySequence('N'), self, self.skip)
        QShortcut(QKeySequence('P'), self, self.prev_song)
        QShortcut(QKeySequence('T'), self, self.toggle_ab)
        QShortcut(QKeySequence('Left'), self, lambda: self.seek_rel(-5))
        QShortcut(QKeySequence('Right'), self, lambda: self.seek_rel(5))
        QShortcut(QKeySequence('Ctrl+Left'), self, lambda: self.seek_rel(-30))
        QShortcut(QKeySequence('Ctrl+Right'), self, lambda: self.seek_rel(30))
        QShortcut(QKeySequence('Ctrl+S'), self, self.save_progress)
        QShortcut(QKeySequence('Up'), self, lambda: self.adjust_gain(+GAIN_STEP))
        QShortcut(QKeySequence('Down'), self, lambda: self.adjust_gain(-GAIN_STEP))

    def build_list_once(self):
        self._loading = True
        self.list_widget.clear()
        for i, fp in enumerate(self.files):
            done = fp in self.annotations
            prefix = '✓' if done else '○'
            item = QListWidgetItem(f'{prefix} {os.path.basename(fp)}')
            item.setForeground(QColor('#2E7D32') if done else QColor('#666'))
            self.list_widget.addItem(item)
        if 0 <= self.current_idx < len(self.files):
            self.list_widget.setCurrentRow(self.current_idx)
        self._loading = False

    def update_list_item(self, idx):
        if not (0 <= idx < self.list_widget.count()):
            return
        fp = self.files[idx]
        done = fp in self.annotations
        prefix = '✓' if done else '○'
        item = self.list_widget.item(idx)
        if item is None:
            return
        item.setText(f'{prefix} {os.path.basename(fp)}')
        item.setForeground(QColor('#2E7D32') if done else QColor('#666'))

    def update_status(self):
        done = len(self.annotations)
        total = len(self.files)
        self.info_label.setText(
            f'进度: {done} / {total}  当前: {self.current_idx + 1} / {total}  '
            f'({done * 100 // max(total, 1)}%)')

    def apply_gain(self):
        gain = 0.0 if self.listen_original else float(self.gain_input.value())
        gain = max(GAIN_MIN, min(GAIN_MAX, gain))
        self.player.set_gain_db(gain)
        self.gain_label_small.setText(f'gain: {gain:+.2f} dB')
        self.update_lufs_display()

    def on_gain_changed(self, _):
        self.apply_gain()

    def update_lufs_display(self):
        if not (0 <= self.current_idx < len(self.files)):
            self.lufs_label.setText('LUFS: ---')
            return
        fp = self.files[self.current_idx]
        base_lufs = self.get_lufs(fp)
        gain = float(self.gain_input.value())
        if base_lufs is None:
            self.lufs_label.setText(
                f'LUFS: 读取失败  |  delta_gain: {gain:+.2f} dB')
        else:
            adj = base_lufs + gain
            self.lufs_label.setText(
                f'原始 LUFS: {base_lufs:6.2f}  |  '
                f'delta_gain: {gain:+.2f} dB  |  '
                f'调整后 LUFS: {adj:6.2f}')

    def toggle_ab(self):
        self.listen_original = not self.listen_original
        self.apply_gain()
        if self.listen_original:
            self.btn_ab.setText('试听: 原始 (T 切换 A/B)')
            self.btn_ab.setStyleSheet(
                'background: #F57C00; color: white; padding: 8px; font-size: 13px;')
        else:
            self.btn_ab.setText('试听: 调整后 (T 切换 A/B)')
            self.btn_ab.setStyleSheet(
                'background: #1976D2; color: white; padding: 8px; font-size: 13px;')

    def load_current(self):
        if not (0 <= self.current_idx < len(self.files)):
            self.file_label.setText('全部完成')
            self.player.pause()
            return
        fp = self.files[self.current_idx]
        try:
            self.player.load(fp)
        except Exception as e:
            _CRASH_LOG.write(f'[load fail] {fp}: {e}\n')
            QMessageBox.warning(self, '加载失败', f'{os.path.basename(fp)}\n{e}')
            return
        self.file_label.setText(os.path.basename(fp))
        self.gain_input.setValue(self.annotations.get(fp, self.gain_input.value()))
        self.apply_gain()
        self._loading = True
        self.list_widget.setCurrentRow(self.current_idx)
        self._loading = False
        self.update_status()
        self.update_window_label()

    def update_window_label(self):
        p = self.player
        if p.y is None or p.sr is None:
            self.window_label.setText('')
            return
        win_len = len(p.y) / p.sr
        if p.total_dur <= p.window_sec + 1:
            self.window_label.setText(f'总长 {fmt_time(p.total_dur)}  (完整加载)')
        else:
            self.window_label.setText(
                f'总长 {fmt_time(p.total_dur)}  |  '
                f'当前块 {fmt_time(p.window_start)} ~ '
                f'{fmt_time(p.window_start + win_len)}  (分块加载)')

    def toggle_play(self):
        self.player.toggle()

    def seek_rel(self, delta_sec):
        self.player.seek_rel(delta_sec)
        self.update_window_label()

    def on_slider_press(self):
        self.seeking = True

    def on_slider_move(self, value):
        dur = self.player.duration_sec
        if dur > 0:
            pos = value * dur / 1000
            self.time_label.setText(f'{fmt_time(pos)} / {fmt_time(dur)}')

    def on_slider_release(self):
        dur = self.player.duration_sec
        if dur > 0:
            target = self.slider.value() * dur / 1000
            try:
                self.player.seek_sec(target)
            except Exception as e:
                _CRASH_LOG.write(f'[seek fail] {target}: {e}\n')
            self.update_window_label()
        self.seeking = False

    def update_position(self):
        if self.seeking:
            return
        dur = self.player.duration_sec
        pos = self.player.pos_sec
        if dur > 0:
            self.slider.blockSignals(True)
            self.slider.setValue(int(pos * 1000 / dur))
            self.slider.blockSignals(False)
            self.time_label.setText(f'{fmt_time(pos)} / {fmt_time(dur)}')
        if self.player.eof:
            self.update_window_label()

    def adjust_gain(self, delta):
        self.gain_input.setValue(round(self.gain_input.value() + delta, 2))

    def _maybe_gc(self):
        self._save_counter += 1
        if self._save_counter >= GC_EVERY:
            self._save_counter = 0
            gc.collect()

    def save_and_next(self):
        if not (0 <= self.current_idx < len(self.files)):
            return
        fp = self.files[self.current_idx]
        self.annotations[fp] = round(self.gain_input.value(), 2)
        self.update_list_item(self.current_idx)
        self.current_idx += 1
        self.save_progress()
        self.load_current()
        self._maybe_gc()

    def skip(self):
        if not (0 <= self.current_idx < len(self.files)):
            return
        self.current_idx += 1
        self.save_progress()
        self.load_current()
        self._maybe_gc()

    def prev_song(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current()
            self.update_status()

    def on_list_select(self, row):
        if self._loading:
            return
        if row == self.current_idx:
            return
        if not (0 <= row < len(self.files)):
            return
        if 0 <= self.current_idx < len(self.files):
            fp_cur = self.files[self.current_idx]
            cur_val = round(self.gain_input.value(), 2)
            if fp_cur in self.annotations and abs(self.annotations[fp_cur] - cur_val) > 1e-6:
                self.annotations[fp_cur] = cur_val
                self.update_list_item(self.current_idx)
                self.save_progress()
        self.current_idx = row
        self.load_current()
        self.update_status()

    def closeEvent(self, event):
        self.save_progress()
        self.save_lufs_cache()
        self.player.stop()
        gc.collect()
        event.accept()


def main():
    ap = argparse.ArgumentParser(description='Delta Gain 标注工具（ffmpeg）')
    ap.add_argument('--input', required=True)
    ap.add_argument('--csv', default='delta_gain.csv')
    ap.add_argument('--progress', default='annotate_progress.json')
    ap.add_argument('--lufs-cache', default='lufs_cache.json')
    ap.add_argument('--window-sec', type=int, default=WINDOW_SEC)
    ap.add_argument('--reset', action='store_true')
    args = ap.parse_args()

    if not ffmpeg_available():
        sys.exit('未找到 ffmpeg 或 ffprobe，请安装后加入 PATH。')
    if args.reset and os.path.exists(args.progress):
        os.remove(args.progress)

    app = QApplication(sys.argv)
    win = AnnotateWindow(args.input, args.csv, args.progress, args.lufs_cache)
    win.player.window_sec = max(60, int(args.window_sec))
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()