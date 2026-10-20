/* ============================================================
   replaygain-nyan 网页端脚本（FastAPI 版）
   樱花/音效引擎/彩虹字/便签 逻辑 1:1 改编自 bishoujosenshi.neocities.org
   ============================================================ */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const fmtDb = v => { v = Number(v); return (v > 0 ? "+" : "") + v.toFixed(1); };

  /* ================= 分区视图切换 ================= */
  function goView(name) {
    document.querySelectorAll(".nav-view").forEach(x => x.classList.toggle("active", x.dataset.goview === name));
    document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + name));
    window.scrollTo(0, 0);
  }
  document.querySelectorAll("[data-goview]").forEach(el => {
    el.addEventListener("click", e => { e.preventDefault(); goView(el.dataset.goview); });
  });

  /* ================= 全局状态 ================= */
  let tracks = [];
  let curIdx = -1;
  const fader = $("gain-fader");
  const volumeSlider = $("volume-slider");
  const seekSlider = $("seek-slider");
  const currTrack = $("music-audio");
  const playpauseBtn = $("playpause-track");
  const soundToggle = $("sound-toggle");
  const markLog = $("mark-log");
  const writeLog = $("write-log");
  const abOrig = $("ab-orig");
  const abApplied = $("ab-applied");
  let abMode = "orig";
  let isPlaying = false;

  function addLog(box, text) {
    const d = document.createElement("div");
    d.className = "ln";
    d.textContent = text;
    box.insertBefore(d, box.firstChild);
    while (box.children.length > 4) box.removeChild(box.lastChild);
  }

  /* ================= 播放器：放的就是真歌 ================= */
  function applyVolume() {
    if (!isPlaying) return;
    const base = volumeSlider.value / 100;
    const scale = abMode === "applied" ? Math.max(0.15, 1 + Number(fader.value) / 26) : 1;
    currTrack.volume = Math.max(0, Math.min(1, base * scale));
  }
  volumeSlider.addEventListener("input", applyVolume);
  fader.addEventListener("input", () => { renderGain(); applyVolume(); });
  abOrig.addEventListener("click", () => { abMode = "orig"; abOrig.classList.add("active"); abApplied.classList.remove("active"); applyVolume(); });
  abApplied.addEventListener("click", () => { abMode = "applied"; abApplied.classList.add("active"); abOrig.classList.remove("active"); applyVolume(); });
  playpauseBtn.addEventListener("click", () => {
    // 还没加载过音源时：自动选当前曲目（曲库为空则提示）
    if (!currTrack.src && tracks.length) {
      selectTrack(curIdx === -1 ? 0 : curIdx);
    } else if (!currTrack.src) {
      addLog(markLog, "曲库是空的喵：先「☁ 上传歌曲」或点 ♥ 更换音乐目录");
      return;
    }
    if (currTrack.paused) currTrack.play().catch(() => {}); else currTrack.pause();
  });
  currTrack.addEventListener("play", () => { playpauseBtn.innerHTML = "&#9646;"; isPlaying = true; });
  currTrack.addEventListener("pause", () => { playpauseBtn.innerHTML = "&#9654;"; isPlaying = false; });
  currTrack.addEventListener("error", () => {
    addLog(markLog, "这首歌浏览器播不了喵（格式限制）——训练和写标签不受影响 ♡");
  });

  /* 播放器时间 / 进度条接线 */
  let seekingBar = false;
  const fmtTime = s => {
    if (!isFinite(s) || s < 0) s = 0;
    s = Math.floor(s);
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  };
  currTrack.addEventListener("loadedmetadata", () => {
    seekSlider.max = Math.floor(currTrack.duration) || 0;
    const td = $("total-duration");
    if (td) td.textContent = fmtTime(currTrack.duration);
  });
  currTrack.addEventListener("timeupdate", () => {
    const ct = $("current-time");
    if (ct) ct.textContent = fmtTime(currTrack.currentTime);
    if (!seekingBar) seekSlider.value = Math.floor(currTrack.currentTime || 0);
  });
  seekSlider.addEventListener("pointerdown", () => { seekingBar = true; });
  seekSlider.addEventListener("pointerup", () => {
    seekingBar = false;
    if (isFinite(currTrack.duration) && currTrack.duration > 0) {
      currTrack.currentTime = Number(seekSlider.value);
    }
  });
  currTrack.addEventListener("ended", advance);   // 播完自动切下一首待标注

  /* ================= 曲库 + 标注台联动 ================= */
  function confCls(t) { return t.mark !== null && t.mark !== undefined ? "ok" : "mid"; }
  function renderLibrary() {
    const box = $("lib-scroll");
    box.innerHTML = "";
    tracks.forEach((t, i) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "lrow" + (i === curIdx ? " sel" : "");
      const cc = t.mark !== null && t.mark !== undefined ? "ok" : "mid";
      let stateHtml = t.state;
      if (t.format_ok === false) {
        stateHtml = '<b style="color:#c2456b">格式不支持</b>';
      } else if (t.playable === false) {
        stateHtml = t.state + '<br><span style="font-size:9px;color:#a58a97">浏览器不试听</span>';
      }
      row.innerHTML =
        '<span>' + t.no + '</span>' +
        '<span class="tname" title="' + String(t.file).replace(/&/g, "&amp;").replace(/"/g, "&quot;") + '">' + t.file + '</span>' +
        '<span>' + t.fmt + '</span>' +
        '<span>' + (t.mark !== null && t.mark !== undefined ? fmtDb(t.mark) : "—") + '</span>' +
        '<span class="' + cc + '">' + t.conf + '</span>' +
        '<span>' + stateHtml + '</span>';
      row.addEventListener("click", () => { selectTrack(i); goView("annotate"); });
      box.appendChild(row);
    });
  }
  function selectTrack(i) {
    curIdx = i;
    renderLibrary();
    const t = tracks[i];
    $("mark-count").textContent = t.no;
    $("track-fmt").textContent = t.fmt + (t.duration ? " · " + t.duration + "s" : "");
    const tc = $("track-conf");
    tc.textContent = t.conf;
    tc.className = t.mark !== null && t.mark !== undefined ? "ok" : "";
    heroTitle.textContent = t.file.replace(/\.[^.]+$/, "");
    splitRainbow(heroTitle);
    fader.value = (t.mark !== null && t.mark !== undefined) ? t.mark : -3.0;
    renderGain();
    applyVolume();
    currTrack.src = "/music-files/" + t.rel.split("/").map(encodeURIComponent).join("/");
    currTrack.load();
    isPlaying = false;
    playpauseBtn.innerHTML = "&#9654;";
    const tn = $("track-name");
    tn.textContent = t.file;
    tn.title = t.file;
    const ta = $("track-artist");
    if (ta) ta.textContent = "第 " + t.no + " 首 · 点播放开始校准 ♪";
  }
  function advance() {
    let n = curIdx;
    for (let k = 0; k < tracks.length; k++) {
      n = (n + 1) % tracks.length;
      if (tracks[n].state !== "SKIP" && tracks[n].state !== "HOLD") break;
    }
    selectTrack(n);
  }

  /* ================= 增益读数 ================= */
  const heroGain = $("hero-gain");
  const lufsOut = $("lufs-out");
  function renderGain() {
    const v = Number(fader.value);
    heroGain.textContent = fmtDb(v);
    heroGain.classList.toggle("warn", v >= 2 || v <= -10);
    lufsOut.textContent = (-14.2 + v).toFixed(1) + " LUFS";
  }
  fader.addEventListener("input", renderGain);

  /* ================= mark / skip / flag / execute ================= */
  $("btn-mark").addEventListener("click", async () => {
    const t = tracks[curIdx];
    const gain = Number(fader.value);
    addLog(markLog, "写入 delta_gain.csv：" + t.file + " @ " + fmtDb(gain) + " dB 喵 ♡");
    try {
      const r = await fetch("/api/mark", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: t.file, gain }) });
      const d = await r.json();
      if (d.ok) { t.mark = gain; t.state = "READY"; addLog(markLog, "已保存 ✓ 共标注 " + d.total + " 首"); advance(); }
      else addLog(markLog, "保存失败喵：" + (d.detail || "?"));
    } catch (e) { addLog(markLog, "保存失败喵（后端没开？）"); }
  });
  $("btn-skip").addEventListener("click", () => {
    addLog(markLog, "skip → 下一首喵");
    advance();
  });
  $("btn-flag").addEventListener("click", () => {
    addLog(markLog, tracks[curIdx].file + " 已打旗 · 稍后人工复核");
  });
  let writePoll = null;
  $("btn-execute").addEventListener("click", () => {
    const dry = $("chk-dry").checked;
    const payload = { dry_run: dry };
    const src = document.querySelector('input[name="model-src"]:checked');
    if (src && src.value === "hub") {
      const sel = document.querySelector('input[name="hub-model"]:checked');
      const dl = sel && hubDownloaded.find(x => x.id === sel.value);
      if (!dl) { addLog(writeLog, "先在模型列表里挑一个已下载的模型喵"); return; }
      if (dl.compatible === false) { addLog(writeLog, "那个模型和本地特征管线版本不兼容喵"); return; }
      payload.model_dir = dl.model_dir;
      addLog(writeLog, "用 hub 模型 " + dl.name + " 写标签喵");
    }
    addLog(writeLog, dry ? "dry-run 演练中…（不动文件）" : "真写入中… 别关页面喵");
    fetch("/api/write/start", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) }).catch(() => {});
    writePoll = setInterval(async () => {
      try {
        const st = await (await fetch("/api/write/status")).json();
        if (!st.running) {
          clearInterval(writePoll); writePoll = null;
          addLog(writeLog, st.exit === 0 ? "done ♡" : "出错了喵 · 看 server_logs/write.log");
        }
      } catch (e) { clearInterval(writePoll); }
    }, 1500);
  });

  /* ================= 训练（真脚本 + 轮询） ================= */
  const trainBar = document.querySelector("#train-prog i");
  const trainTxt = $("train-txt");
  let trainPoll = null;
  $("btn-train").addEventListener("click", async () => {
    if (trainPoll) return;
    trainBar.style.width = "100%";
    trainTxt.textContent = "starting… 特征提取比较久，别关页面喵";
    try {
      const r = await fetch("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      const d = await r.json();
      if (!d.ok) { trainTxt.textContent = "启动失败喵：" + (d.detail || "?"); trainPoll = null; return; }
      trainPoll = setInterval(async () => {
        try {
          const st = await (await fetch("/api/train/status")).json();
          if (st.running) trainTxt.textContent = "running… " + (st.tail.slice(-1)[0] || "");
          else {
            clearInterval(trainPoll); trainPoll = null;
            trainTxt.textContent = st.exit === 0 ? "done! model saved ♡" : "出错了喵 · 看 server_logs/train.log";
            if (st.exit === 0) loadModelCard();
          }
        } catch (e) { clearInterval(trainPoll); }
      }, 1500);
    } catch (e) { trainTxt.textContent = "启动失败喵"; }
  });

  /* ================= UI 音效引擎（原站 1:1，默认关） ================= */
  const SOUND_CFG = { hover: null, click: null, hoverVol: 0.42, clickVol: 0.55 };
  let soundsOn = false, hoverSound = null, clickSound = null;
  let audioUnlocked = false, lastHoverTarget = null, hoverCooldown = false;
  const clickableSelector = 'a, button, [role="button"], input, select, summary, .lrow';
  function clickableFrom(node) { return node && node.closest ? node.closest(clickableSelector) : null; }
  function initSounds() {
    if (!SOUND_CFG.hover || !SOUND_CFG.click) return false;
    hoverSound = new Audio(SOUND_CFG.hover); clickSound = new Audio(SOUND_CFG.click);
    hoverSound.preload = "auto"; clickSound.preload = "auto";
    hoverSound.volume = SOUND_CFG.hoverVol; clickSound.volume = SOUND_CFG.clickVol;
    return true;
  }
  function unlockAudio() {
    if (audioUnlocked || !soundsOn) return;
    audioUnlocked = true;
    [hoverSound, clickSound].forEach(a => {
      const vol = a === hoverSound ? SOUND_CFG.hoverVol : SOUND_CFG.clickVol;
      const old = a.volume; a.volume = 0;
      const p = a.play();
      if (p && p.then) p.then(() => { a.pause(); a.currentTime = 0; a.volume = vol; }).catch(() => { a.volume = old || vol; });
      else a.volume = vol;
    });
  }
  function playSound(a) {
    if (!a) return;
    try { a.pause(); a.currentTime = 0; const p = a.play(); if (p && p.catch) p.catch(() => {}); } catch (_) {}
  }
  document.addEventListener("pointerover", e => {
    if (!soundsOn) return;
    const t = clickableFrom(e.target);
    if (!t || t === lastHoverTarget) return;
    lastHoverTarget = t; unlockAudio();
    if (audioUnlocked && !hoverCooldown) {
      hoverCooldown = true; playSound(hoverSound);
      setTimeout(() => { hoverCooldown = false; }, 70);
    }
  }, true);
  document.addEventListener("pointerdown", () => { if (soundsOn) { unlockAudio(); playSound(clickSound); } }, true);
  soundToggle.addEventListener("click", () => {
    soundsOn = !soundsOn;
    if (soundsOn && !hoverSound) {
      if (!initSounds()) { soundToggle.textContent = "no mp3 config"; soundsOn = false; return; }
    }
    soundToggle.textContent = soundsOn ? "on ♪" : "off";
    soundToggle.closest(".window-head").querySelector("span:last-child").textContent = soundsOn ? "on" : "off";
  });

  /* ================= 樱花花瓣（原站 1:1） ================= */
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const petals = [];
    const rand = (min, max) => Math.random() * (max - min) + min;
    const makePetal = (delay = 0) => {
      const petal = document.createElement("div");
      petal.className = "sakura-petal";
      const size = Math.round(rand(8, 15));
      petal.style.width = size + "px";
      petal.style.height = Math.max(7, size - 2) + "px";
      petal.style.left = rand(-5, 100) + "vw";
      petal.style.animationDuration = rand(9, 18) + "s";
      petal.style.animationDelay = delay + "s";
      petal.style.setProperty("--drift1", rand(-12, 12) + "vw");
      petal.style.setProperty("--drift2", rand(-22, 22) + "vw");
      petal.style.setProperty("--drift3", rand(-16, 16) + "vw");
      petal.style.setProperty("--drift4", rand(-30, 30) + "vw");
      petal.style.animationName = "sakuraFall";
      petal.style.animationTimingFunction = "linear";
      petal.style.animationIterationCount = "1";
      document.body.appendChild(petal);
      petals.push(petal);
      petal.addEventListener("animationend", () => {
        petal.remove();
        const i = petals.indexOf(petal);
        if (i !== -1) petals.splice(i, 1);
        makePetal(0);
      }, { once: true });
    };
    for (let i = 0; i < 22; i++) makePetal(-rand(0, 18));
  }

  /* ================= 彩虹字（原站 1:1） ================= */
  const rainbowColors = ["#ff7fa8", "#f49bc7", "#b88de8", "#7fa8ff", "#78c8d8", "#86cda1", "#f2c766"];
  const heroTitle = $("hero-rainbow");
  let heroCharLen = 0;
  function splitRainbow(el) {
    const t = el.textContent;
    el.textContent = "";
    const inner = document.createElement("span");
    inner.className = "tt-inner";
    const seg = document.createElement("span");
    seg.className = "tt-seg";
    [...t].forEach(ch => { const sp = document.createElement("span"); sp.textContent = ch; seg.appendChild(sp); });
    inner.appendChild(seg);
    el.appendChild(inner);
    heroCharLen = [...t].length;
    setupMarquee(el);
  }
  function setupMarquee(el) {
    const inner = el.querySelector(".tt-inner");
    const seg = el.querySelector(".tt-seg");
    if (!inner || !seg) return;
    const apply = () => {
      inner.getAnimations().forEach(a => a.cancel());
      const w = seg.scrollWidth;
      if (w > el.clientWidth - 8 && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        el.classList.add("is-scroll");
        if (inner.querySelectorAll(".tt-seg").length < 2) inner.appendChild(seg.cloneNode(true));
        const dur = Math.max(6000, Math.round(w / 45) * 1000); // 约 45px/秒
        inner.animate([{ transform: "translateX(0)" }, { transform: `translateX(-${w}px)` }],
          { duration: dur, iterations: Infinity });
      } else {
        el.classList.remove("is-scroll");
      }
    };
    apply();
    // Web 字体异步加载会改变文字宽度：字体就绪后重新测量一次
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(apply).catch(() => {});
    }
    // 悬停暂停，方便细看歌名
    el.addEventListener("mouseenter", () => inner.getAnimations().forEach(a => a.pause()));
    el.addEventListener("mouseleave", () => inner.getAnimations().forEach(a => a.play()));
  }
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    splitRainbow(heroTitle);
    let offset = 0;
    setInterval(() => {
      heroTitle.querySelectorAll(".tt-seg > span").forEach((letter, i) => {
        letter.style.color = rainbowColors[((i % heroCharLen) + offset) % rainbowColors.length];
      });
      offset = (offset + 1) % rainbowColors.length;
    }, 900);
  } else {
    // 减少动态偏好：静态彩虹着色，不滚动
    splitRainbow(heroTitle);
    heroTitle.querySelectorAll(".tt-seg > span").forEach((letter, i) => {
      letter.style.color = rainbowColors[i % rainbowColors.length];
    });
  }

  /* ================= 小贴士轮播 ================= */
  const tipEl = $("tip-text");
  const tips = [
    "「待复核 / 打滑?」的标签先人工复核，别直接喂给模型学坏喵",
    "dry-run 永远是你的朋友——先预览要改哪些再动手 ♡",
    "组信号：同专辑的歌会互相参考，标注时别孤立地听",
    "低可信标签堆太多会拉低模型质量，宁可少标几首"
  ];
  let tipIdx = 0;
  setInterval(() => {
    if (!tipEl) return;
    tipIdx = (tipIdx + 1) % tips.length;
    tipEl.style.opacity = 0;
    setTimeout(() => {
      tipEl.innerHTML = "<b>tip " + (tipIdx + 1) + "：</b>" + tips[tipIdx];
      tipEl.style.opacity = 1;
    }, 350);
  }, 6000);

  /* ================= 便签（原站 1:1） ================= */
  document.querySelectorAll(".sticky-text[data-note-key]").forEach(area => {
    const key = "beezy-" + area.dataset.noteKey;
    area.value = localStorage.getItem(key) || "";
    area.addEventListener("input", () => localStorage.setItem(key, area.value));
  });

  /* ================= 公告（自动弹出） ================= */
  const notice = $("notice");
  const EVER = "rg-notice-ever";
  try { if (!localStorage.getItem(EVER)) notice.hidden = false; } catch (e) { notice.hidden = false; }
  function closeNotice(forever) {
    notice.hidden = true;
    try { if (forever) localStorage.setItem(EVER, "1"); } catch (e) {}
  }
  $("notice-x").addEventListener("click", () => closeNotice(false));
  $("notice-ok").addEventListener("click", () => closeNotice($("notice-forever").checked));
  $("notice-open").addEventListener("click", () => { notice.hidden = false; });

  /* ================= 启动：读配置 + 拉真曲库 ================= */
  function dirFormHTML(lead) {
    return '<div style="padding:10px;font-size:11px;line-height:1.9">' + (lead || "") +
      "填你的音乐文件夹路径（如 D:\\Music），保存后自动扫描：</div>" +
      '<div style="display:flex;gap:6px;margin-top:8px">' +
      '<input id="music-dir-input" placeholder="D:\\Music" style="flex:1;min-width:0">' +
      '<button class="browser-button" id="music-dir-save" type="button">保存并扫描</button></div>' +
      '<p class="ntxt" id="dir-form-msg" style="min-height:14px;margin:4px 0 0"></p>';
  }
  function wireDirForm() {
    $("music-dir-save").addEventListener("click", async () => {
      const v = $("music-dir-input").value.trim();
      const msg = $("dir-form-msg");
      if (!v) { msg.textContent = "先填一个真实存在的文件夹路径喵"; return; }
      const save = $("music-dir-save");
      save.disabled = true;
      const oldText = save.textContent;
      save.textContent = "扫描中…";
      msg.textContent = "";
      try {
        const r = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ music_dir: v }) });
        if (r.ok) {
          location.reload();
        } else {
          const d = await r.json().catch(() => ({}));
          save.disabled = false; save.textContent = oldText;
          msg.textContent = "保存失败喵：" + (d.detail || r.status);
        }
      } catch (e) {
        save.disabled = false; save.textContent = oldText;
        msg.textContent = "连不上后端喵";
      }
    });
  }
  $("btn-music-dir").addEventListener("click", () => {
    const panel = $("music-dir-panel");
    if (!panel.hidden) { panel.hidden = true; panel.innerHTML = ""; return; }
    panel.hidden = false;
    panel.innerHTML = dirFormHTML("");
    wireDirForm();
  });

  /* ================= 云端会话：上传 / 下载 / 退出 ================= */
  /* ================= 模型市场 ================= */
  let hubItems = [];
  let hubDownloaded = [];
  const escHtml = s => { const d = document.createElement("div"); d.textContent = String(s ?? ""); return d.innerHTML; };
  const fmtMae = v => (v == null || v === "" ? "?" : Number(v).toFixed(2));

  async function loadModelCard() {
    try {
      const d = await (await fetch("/api/model/current")).json();
      const box = $("model-card-box");
      if (!d.exists) { box.hidden = true; return; }
      box.hidden = false;
      const m = d.meta || {}, c = d.card || {};
      const st = (c.oof_stats && Object.keys(c.oof_stats).length) ? c.oof_stats : (m.oof_stats || {});
      $("model-card-body").innerHTML =
        "类型 <b>" + escHtml(m.model_type || "?") + "</b> · " + (m.n_ensemble || 1) +
        " seed · 样本 <b>" + (m.n_train_samples || "?") + "</b> 首 · MAE <b>" +
        (st.mae != null ? Number(st.mae).toFixed(3) : "?") + "</b> dB<br>" +
        "特征管线 v" + (m.feature_pipeline_version ?? "?") + " · lightgbm " +
        escHtml(m.lightgbm || "?") + " · 训练于 " + escHtml(m.trained_at || "?");
      $("mine-model-hint").textContent = "MAE " + (st.mae != null ? Number(st.mae).toFixed(2) : "?");
      // 训练页指标卡
      $("stat-mae").textContent = st.mae != null ? Number(st.mae).toFixed(3) + " dB" : "—";
      $("stat-seeds").textContent = (m.n_ensemble || "—") + " seed";
      $("stat-features").textContent = (m.n_features_raw || "—") + " 维";
      $("stat-stage").textContent = m.group_signal_keys ? "2 (group)" : "1";
    } catch (e) { /* 后端没开时忽略 */ }
  }

  function renderHubList() {
    const listEl = $("hub-list");
    const parts = [];
    hubDownloaded.forEach(dl => {
      parts.push('<label class="chk-row"><input type="radio" name="hub-model" value="' +
        escHtml(dl.id) + '">✔ <b>' + escHtml(dl.name) + '</b>（已就绪）' +
        '<span class="dim"> 样本 ' + (dl.samples ?? "?") + ' · MAE ' + fmtMae(dl.mae) +
        (dl.compatible === false ? ' · <b>版本不兼容</b>' : '') + '</span></label>');
    });
    hubItems.forEach(it => {
      if (hubDownloaded.some(x => x.id === it.id)) return;
      parts.push('<div class="chk-row">' + escHtml(it.name || it.id) +
        ' <span class="dim">样本 ' + (it.sample_count ?? "?") + ' · MAE ' + fmtMae(it.mae) +
        (it.compatible === false ? ' · <b>版本不兼容</b>' : '') + '</span>' +
        (it.compatible === false ? "" :
          ' <button class="browser-button small" data-hub-dl="' + escHtml(it.id) +
          '" type="button">' + (it.downloaded ? "重新下载" : "下载") + "</button>") + "</div>");
    });
    listEl.innerHTML = parts.join("") ||
      '<div class="chk-row dim">hub 里还没有模型喵（RG_HUB_URL 没配或索引为空）</div>';
    listEl.querySelectorAll("[data-hub-dl]").forEach(b => {
      b.addEventListener("click", async () => {
        b.disabled = true; b.textContent = "下载中…";
        try {
          const r = await fetch("/api/hub/download", { method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: b.dataset.hubDl }) });
          const d = await r.json().catch(() => ({}));
          $("model-src-txt").textContent = r.ok && d.ok
            ? "下载好啦 ♡ 勾选它就能用了"
            : "下载失败喵：" + (d.detail || r.status);
        } catch (e) {
          $("model-src-txt").textContent = "下载失败喵（网络？）";
        }
        await refreshHub();
        renderHubList();
      });
    });
  }

  async function refreshHub() {
    try {
      const h = await (await fetch("/api/hub/models")).json();
      hubItems = h.models || [];
      $("hub-count").textContent = h.configured ? ("hub " + hubItems.length + " 个") : "未配置";
      const l = await (await fetch("/api/model/list-local")).json();
      hubDownloaded = l.models || [];
    } catch (e) { /* 后端没开时忽略 */ }
  }

  const UPLOAD_BATCH = 8;   // 每批文件数：整文件夹上传时分批请求，避免超大请求体

  function uploadFilesBatch(files, onProgress) {
    return new Promise(res => {
      const fd = new FormData();
      files.forEach(f => fd.append("files", f, f.name));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/session/upload");
      xhr.upload.addEventListener("progress", e => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      });
      xhr.addEventListener("load", () => {
        let d = {};
        try { d = JSON.parse(xhr.responseText); } catch (_) {}
        res({ ok: xhr.status === 200 && !!d.ok, saved: (d.saved || []).length, detail: d });
      });
      xhr.addEventListener("error", () => res({ ok: false }));
      xhr.send(fd);
    });
  }

  async function uploadInBatches(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    const prog = $("upload-prog"), bar = prog.querySelector("i"), txt = $("upload-txt");
    prog.style.display = "block";
    bar.style.width = "0%";
    const batches = Math.ceil(files.length / UPLOAD_BATCH);
    let savedTotal = 0;
    for (let b = 0; b < batches; b++) {
      const chunk = files.slice(b * UPLOAD_BATCH, (b + 1) * UPLOAD_BATCH);
      const base = (b * UPLOAD_BATCH) / files.length;
      txt.textContent = `上传批次 ${b + 1}/${batches}…`;
      let r = { ok: false };
      try {
        r = await uploadFilesBatch(chunk, pc => {
          const ratio = base + pc * (chunk.length / files.length);
          bar.style.width = Math.round(ratio * 100) + "%";
          txt.textContent = `上传中… 约 ${Math.round(ratio * files.length)}/${files.length} 首`;
        });
      } catch (e) { /* 网络异常按失败处理 */ }
      if (!r.ok) {
        prog.style.display = "none";
        txt.textContent = "有一批上传失败喵（网络/文件超上限？），已传部分不受影响";
        await refreshLibrary();
        return;
      }
      savedTotal += r.saved;
    }
    prog.style.display = "none";
    txt.textContent = `全部传完啦 ♡ 曲库已刷新（会话内新增 ${savedTotal} 首）`;
    await refreshLibrary();
  }

  $("btn-upload").addEventListener("click", () => {
    const b = $("btn-upload");
    b.textContent = "已打开文件选择框…";
    setTimeout(() => { b.textContent = "☁ 上传歌曲"; }, 2500);
    $("upload-input").click();
  });
  $("upload-input").addEventListener("change", () => {
    const fs = $("upload-input").files;
    $("upload-input").value = "";
    uploadInBatches(fs);
  });
  $("btn-upload-folder").addEventListener("click", () => $("folder-input").click());
  $("folder-input").addEventListener("change", () => {
    const fs = $("folder-input").files;
    $("folder-input").value = "";
    uploadInBatches(fs);
  });
  $("btn-download-zip").addEventListener("click", async () => {
    const btn = $("btn-download-zip");
    btn.disabled = true;
    const old = btn.textContent;
    btn.textContent = "打包中…";
    try {
      const r = await fetch("/api/session/download");
      if (!r.ok) {
        $("upload-txt").textContent = "下载不了喵：还没有云端会话（先上传几首歌）";
        return;
      }
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "replaygain-tagged.zip";
      a.click();
      URL.revokeObjectURL(a.href);
      $("upload-txt").textContent = "打包下载完成 ♡ 记得退出会话再走喵";
    } catch (e) {
      $("upload-txt").textContent = "下载失败喵（网络/后端？）";
    } finally {
      btn.disabled = false;
      btn.textContent = "⬇ 下载打完标签";
    }
  });
  $("btn-session-reset").addEventListener("click", async () => {
    if (!confirm("退出会话会清空你上传的歌和全部标注数据喵，确定？")) return;
    const btn = $("btn-session-reset");
    btn.disabled = true;
    try {
      await fetch("/api/session/reset", { method: "POST" });
      $("upload-txt").textContent = "已退出云端会话，回到演示曲库喵";
      refreshLibrary();
    } finally {
      btn.disabled = false;
    }
  });

  /* ---- hub 列表开合 / 导入 / 打包发布 ---- */
  document.querySelectorAll('input[name="model-src"]').forEach(r => {
    r.addEventListener("change", () => {
      $("hub-list").hidden =
        document.querySelector('input[name="model-src"]:checked').value !== "hub";
    });
  });
  $("import-input").addEventListener("change", async () => {
    const f = $("import-input").files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    $("model-src-txt").textContent = "导入中…";
    try {
      const r = await fetch("/api/hub/import", { method: "POST", body: fd });
      const d = await r.json().catch(() => ({}));
      if (r.ok && d.ok) {
        $("model-src-txt").textContent = "导入成功 ♡ " + (d.name || "");
        document.querySelector('input[name="model-src"][value="hub"]').checked = true;
        $("hub-list").hidden = false;
      } else {
        $("model-src-txt").textContent = "导入失败喵：" + (d.detail || r.status);
      }
    } catch (e) {
      $("model-src-txt").textContent = "导入失败喵（后端没开？）";
    }
    $("import-input").value = "";
    await refreshHub();
    renderHubList();
  });
  $("btn-pack-model").addEventListener("click", async () => {
    const btn = $("btn-pack-model");
    btn.disabled = true;
    btn.textContent = "打包中…";
    try {
      const r = await fetch("/api/model/pack", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          author: $("card-author").value.trim(),
          note: $("card-note").value.trim(),
          publish_tracks: $("card-publish-tracks").checked }) });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        btn.textContent = "打包失败喵：" + (d.detail || r.status);
        btn.disabled = false;
        return;
      }
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = r.headers.get("X-Model-Name") || "rg-model.zip";
      a.click();
      URL.revokeObjectURL(a.href);
      btn.textContent = "已打包下载 ♡ 传到 hub 的 Release 再提 PR 就能上架";
      setTimeout(() => {
        btn.textContent = "⬇ 打包模型（生成发布 zip）";
        btn.disabled = false;
      }, 4000);
    } catch (e) {
      btn.textContent = "打包失败喵（后端没开？）";
      btn.disabled = false;
    }
  });

  /* ================= 访问码验证 ================= */
  function showAuth() {
    $("auth-mask").hidden = false;
    setTimeout(() => { try { $("auth-code").focus(); } catch (_) {} }, 60);
  }
  async function checkAuth() {
    try {
      const st = await (await fetch("/api/auth/status")).json();
      if (st.required && !st.authed) { showAuth(); return false; }
    } catch (e) { /* 后端没开时让 boot 自己报错 */ }
    return true;
  }
  async function submitAuth() {
    const code = $("auth-code").value;
    try {
      const r = await fetch("/api/auth", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }) });
      if (r.ok) {
        $("auth-mask").hidden = true;
        $("auth-msg").textContent = "";
        $("auth-code").value = "";
        boot();
      } else {
        $("auth-msg").textContent = "访问码不对喵，再想想 ♡";
      }
    } catch (e) {
      $("auth-msg").textContent = "连不上后端喵";
    }
  }
  $("auth-ok").addEventListener("click", submitAuth);
  $("auth-code").addEventListener("keydown", e => { if (e.key === "Enter") submitAuth(); });
  $("auth-x").addEventListener("click", () => { $("auth-mask").hidden = true; });

  async function refreshLibrary() {
    const r = await fetch("/api/library");
    if (r.status === 401) { showAuth(); return; }
    const d = await r.json();
    tracks = d.tracks.map(t => ({
      no: t.no, file: t.file, rel: t.rel, fmt: t.fmt, duration: t.duration,
      mark: t.mark,
      conf: t.mark !== null && t.mark !== undefined ? "高" : "待标",
      state: t.mark !== null && t.mark !== undefined ? "READY" : "待标注"
    }));
    renderLibrary();
    const lc = $("lib-count");
    if (lc) lc.textContent = "♡ library.db · " + d.total + " 首" +
      (d.mode === "cloud" ? " · ☁ 云端会话" : "");
    const lt = $("lib-total");
    if (lt) lt.textContent = String(d.total).padStart(3, "0");
    // 首次加载自动选中第一首，播放器即有真实音源可播
    if (tracks.length && curIdx === -1) selectTrack(0);
  }
  async function boot() {
    // 部署形态：localhost = 本地版（显示换目录），其余 = 云端版（显示上传/下载）
    const IS_LOCAL = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
    const lt = $("local-tools"), ct = $("cloud-tools");
    if (lt && ct) {
      lt.style.display = IS_LOCAL ? "" : "none";
      ct.style.display = IS_LOCAL ? "none" : "";
    }
    if (!(await checkAuth())) return;
    try {
      const cfg = await (await fetch("/api/config")).json();
      if (!cfg.valid) {
        $("lib-scroll").innerHTML = dirFormHTML("音乐目录还没设置（或路径失效了）喵 ♡ ");
        wireDirForm();
        return;
      }
      await refreshLibrary();
      loadModelCard();
      refreshHub().then(renderHubList);
    } catch (e) {
      const box = $("lib-scroll");
      box.innerHTML = '<div style="padding:10px;font-size:11px;line-height:1.9">后端没连上喵 ♡ 曲库/标注都要 server/app.py 提供，请先运行 <b>python server/app.py</b>，再访问 <b>http://127.0.0.1:8792</b></div>';
    }
  }
  boot();
})();
