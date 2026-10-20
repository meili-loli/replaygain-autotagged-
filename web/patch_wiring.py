# -*- coding: utf-8 -*-
"""一次性补丁：app.js 数据层接真 API"""
p = 'js/app.js'
a = open(p, encoding='utf-8').read()

# 1) 静态 tracks → 空（真数据）
import re
m = re.search(r'  const tracks = \[.*?\];\n(  let curIdx = -1;)?', a, re.S)
if not m:
    raise SystemExit("tracks block not found")
a = a[:m.start()] + "  let tracks = [];\n  let curIdx = -1;" + a[m.end():]

# 2) 行点击：跳标注台视图
a2 = a.replace(
    '''      row.addEventListener("click", () => selectTrack(i));''',
    '''      row.addEventListener("click", () => { selectTrack(i); goView("annotate"); });''', 1)
assert a2 != a, "row click"
a = a2

# 3) gain 列显示 mark 值
a2 = a.replace(
    """        '<span>' + fmtDb(t.gain) + '</span>' +""",
    """        '<span>' + (t.mark !== null && t.mark !== undefined ? fmtDb(t.mark) : "—") + '</span>' +""", 1)
assert a2 != a, "mark col"
a = a2

# 4) selectTrack：真歌进播放器 + mark 恢复
old = """    fader.value = t.gain;
    renderGain();
  }"""
new = """    fader.value = (t.mark !== null && t.mark !== undefined) ? t.mark : -3.0;
    renderGain();
    /* 播放器加载真歌喵 */
    if (t.rel) {
      currTrack.src = "/music-files/" + t.rel.split("/").map(encodeURIComponent).join("/");
      currTrack.load();
      isPlaying = false;
      playpauseBtn.innerHTML = "&#9654;";
      trackName.textContent = t.file;
      trackArtist.textContent = "第 " + t.no + " 首 · " + t.state;
    }
  }"""
assert old in a, "sel tail"
a = a.replace(old, new, 1)

# 5) mark → POST /api/mark
old = '''  document.getElementById("btn-mark").addEventListener("click", () => {
    const t = tracks[curIdx];
    t.state = "READY";
    addLog(markLog, "已记录 " + t.file + " @ " + fmtDb(fader.value) + " dB 喵 ♡");
    advance();
  });'''
new = '''  document.getElementById("btn-mark").addEventListener("click", async () => {
    const t = tracks[curIdx];
    t.state = "READY";
    addLog(markLog, "写入 delta_gain.csv：" + t.file + " @ " + fmtDb(fader.value) + " dB 喵 ♡");
    try {
      const r = await fetch("/api/mark", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: t.file, gain: Number(fader.value) }) });
      const d = await r.json();
      addLog(markLog, "已保存 ✓ 目前共标注 " + d.total + " 首");
    } catch (e) {
      addLog(markLog, "保存失败喵（后端没开？）");
    }
    advance();
  });'''
assert old in a, "mark"
a = a.replace(old, new, 1)

# 6) train → 真脚本 + 轮询
old = '''  let training = false;
  document.getElementById("btn-train").addEventListener("click", () => {
    if (training) return;
    training = true;
    let w = 0;
    trainTxt.textContent = "training… 42 features · 5 seeds 喵";
    const timer = setInterval(() => {
      w += 4 + Math.random() * 6;
      if (w >= 100) {
        w = 100; clearInterval(timer); training = false;
        trainTxt.textContent = "done! model saved · MAE 0.666 喵♡";
      }
      trainBar.style.width = w + "%";
    }, 180);
  });'''
new = '''  let training = false;
  function pollJob(name, url, onTick) {
    const timer = setInterval(async () => {
      try {
        const st = await (await fetch(url)).json();
        onTick(st);
        if (!st.running) clearInterval(timer);
      } catch (e) { clearInterval(timer); }
    }, 1500);
  }
  document.getElementById("btn-train").addEventListener("click", async () => {
    if (training) return;
    training = true;
    trainBar.style.width = "100%";
    trainTxt.textContent = "starting… 特征提取比较久，别关页面喵";
    try {
      const d = await (await fetch("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })).json();
      if (!d.ok) { trainTxt.textContent = "启动失败喵：" + (d.detail || "?"); training = false; return; }
      pollJob("train", "/api/train/status", st => {
        if (st.running) { trainTxt.textContent = "running… " + (st.tail.slice(-1)[0] || ""); }
        else { trainTxt.textContent = st.exit === 0 ? "done! model saved ♡" : "出错了喵 · 看 server_logs/train.log"; }
      });
    } catch (e) { training = false; }
  });'''
assert old in a, "train"
a = a.replace(old, new, 1)

# 7) execute → 真写入 dry-run
old = '''  document.getElementById("btn-execute").addEventListener("click", () => {
    addLog(writeLog, "dry-run：将改写 131 个文件的标签（演示，未动文件）· NYAN CERTIFIED 喵");
  });'''
new = '''  document.getElementById("btn-execute").addEventListener("click", () => {
    addLog(writeLog, "启动 predict_and_tag.py …（没训练过模型会报错，先去 ③ 训练喵）");
    fetch("/api/write/start", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dry_run: true }) }).catch(() => {});
    pollJob("write", "/api/write/status", st => {
      if (!st.running) { addLog(writeLog, st.tail.slice(-1)[0] || "done ♡"); }
    });
  });'''
assert old in a, "exec"
a = a.replace(old, new, 1)

# 8) init：播放音量实时随校准 + boot 拉真曲库
old = '''  renderLibrary();
  renderGain();
})();'''
new = '''  /* 播放音量实时反映校准：应用后=按增益缩放音量（耳朵直接听差别） */
  function applyVolume() {
    if (!isPlaying) return;
    const v = Number(fader.value);
    const scale = abMode === "applied" ? Math.max(0.2, 1 + v / 26) : 1;
    currTrack.volume = Math.max(0, Math.min(1, (volumeSlider.value / 100) * scale));
  }
  volumeSlider.addEventListener("input", applyVolume);
  fader.addEventListener("input", () => { renderGain(); applyVolume(); });
  abOrig.addEventListener("click", () => { if (isPlaying) currTrack.volume = Math.min(1, volumeSlider.value / 100); });
  abApplied.addEventListener("click", () => { renderGain(); applyVolume(); });

  async function boot() {
    try {
      const cfg = await (await fetch("/api/config")).json();
      if (!cfg.valid) {
        libScroll.innerHTML = '<div style="padding:10px;font-size:11px;line-height:1.9">' +
          '还没设置音乐目录喵 ♡ 在下面填你的音乐文件夹路径，保存后自动扫描：</div>' +
          '<div style="display:flex;gap:6px;margin-top:8px">' +
          '<input id="music-dir-input" placeholder="D:\\\\Music" style="flex:1;min-width:0">' +
          '<button class="browser-button" id="music-dir-save" type="button">保存并扫描</button></div>';
        document.getElementById("music-dir-save").addEventListener("click", async () => {
          const v = document.getElementById("music-dir-input").value.trim();
          const r = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ music_dir: v }) });
          if (r.ok) location.reload();
          else alert((await r.json()).detail || "出错了喵");
        });
        renderGain();
        return;
      }
      const d = await (await fetch("/api/library")).json();
      tracks = d.tracks.map(t => ({
        no: t.no, file: t.file, rel: t.rel, fmt: t.fmt, duration: t.duration,
        mark: t.mark,
        gain: (t.mark !== null && t.mark !== undefined) ? t.mark : -3.0,
        conf: t.state === "READY" ? "高" : "—",
        state: t.state
      }));
      renderLibrary();
    } catch (e) {}
  }
  boot();

  renderGain();
})();'''
assert old in a, "init"
a = a.replace(old, new, 1)

open(p, 'w', encoding='utf-8').write(a)
print("app.js wired ✓")
