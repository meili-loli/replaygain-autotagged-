# replaygain-web

`replaygain-nyan` 网页端：**复古个人站皮肤**（bishoujosenshi.neocities.org 设计系统 + Pixelsafari 像素素材）× **FastAPI 真后端**（曲库扫描 / 人工标注 / 训练 / 批量写 ReplayGain 标签）。

## 素材

`assets/gif/` 下的动图两类来源，全部本地化存储、无外链：

**Pixelsafari**（[pixelsafari.neocities.org](https://pixelsafari.neocities.org/)，neocities 社区免费像素素材库，原站自己也用）：导航贴纸（蝴蝶结/星月/小花/爱心弓/闪光）、分隔线（爱心串/锯齿/心形条）、blinkie 页脚横幅。

**原站同款二次元素材**（与 bishoujosenshi 所用的外链同源，便于 1:1 对标）：
| 文件 | 原始来源 | 用途 |
|---|---|---|
| `banner.gif` | i.pinimg.com（原站横幅同款初音 GIF） | 页头横幅 |
| `character.png` | i.ibb.co（原站右下角看板立绘同款） | 右栏看板窗 |
| `anime1.gif` | media3.giphy.com（原站 watchlist 同款 Angel Beats!） | desktop buddy |
| `anime2.gif` | c.tenor.com（原站 watchlist 同款 Toradora!） | todo 配图 |
| `anime3.gif` | c.tenor.com（原站 watchlist 同款） | visitor note |
| `cat10-15.gif` 等 | pixelsafari animals/cat | 猫猫图鉴、宠物头像 |

**复古小件**：访客计数器（绿字数码管，CSS 手写）、webring、88×31 图章墙、gify.pet 电子宠物 iframe（原站同款服务，宠物名 NYAN）——均为老 web 标配，纯 CSS/HTML 实现。

**版权提醒**：以上二次元 GIF 的画作者均不可考（Pinterest/Tenor/Giphy 转载链），本项目仅供个人/学习使用；若要开源发布或商用，请替换为自有授权素材或取得画师授权。

## 运行（本地一键启动）

Windows：**双击 `启动.bat`**（首次运行自动装依赖，之后秒开；浏览器自动打开
http://127.0.0.1:8792 ；关闭那个黑窗口 = 停止服务）。

手动方式（其他系统 / 想了解细节）：

```bash
cd replaygain-web
pip install -r requirements.txt
# 相邻目录有 replaygain-autotagged/ 时，训练/写标签还要装它的依赖：
pip install -r ../replaygain-autotagged/requirements.txt
python server/app.py     # 打开 http://127.0.0.1:8792
```

- 后端 `server/app.py`（FastAPI）负责曲库扫描、标注 CSV（`replaygain-autotagged/delta_gain.csv`）、训练、写标签和音频播放，前端静态页由它一并托管，前后端同源无跨域
- 音乐目录：`server_config.json` 的 `music_dir` 优先；**留空或不配置时自动使用自带的 `music/` 示例曲**，开箱即用。换自己的收藏 → 曲库窗口点「♥ 更换音乐目录」填路径，或直接改 `server_config.json` 后刷新
- 训练 / 写标签依赖 `replaygain-autotagged/` 目录里的真脚本（`train_eval_export.py` / `predict_and_tag.py`）

## 迁移到别的电脑

1. 整个 `replaygain-web/` 文件夹拷走即可（含 `music/` 示例曲约 2MB，全部素材本地化无外链；连同旁边的 `replaygain-autotagged/` 一起拷才能训练/写标签）
2. 目标机装 Python 3.9+，`pip install -r requirements.txt`
3. `python server/app.py` → 打开 http://127.0.0.1:8792 就能用
4. `server_config.json` 是运行时状态文件，迁移前可以删掉或清空 `music_dir`（会自动回落到自带示例曲；绑定 127.0.0.1，不弹防火墙提示）
5. 想开 UI 音效：往 `assets/sfx/` 放两个 mp3，并在 `js/app.js` 的 `SOUND_CFG` 填上文件名

## 出处对照表（哪些代码是从哪来的）

### 1:1 取自 bishoujosenshi.neocities.org（内联样式/脚本原样搬运）

| 内容 | 位置 | 说明 |
|---|---|---|
| body 底纹 | `css/style.css` body 规则 | 白玻璃 00s 四层像素底纹（radial 点阵 + 45° 格纹），取自原站 inline.css |
| 窗口铬 | `.retro-window` / `.window-head` | 2px 粉边框、白玻璃 `rgba(255,255,255,.62)` + blur(4px)、`2px 2px 0` 硬投影 |
| 四色标题栏渐变 | `.pink-head` `.blue-head` `.peach-head` `.lilac-head` | 渐变色值原样（`#ffb9d0→#ef88ad` 等） |
| 字距悬停微交互 | `.window-head>span:hover{letter-spacing:2px}` | 原站招牌细节，transition .45s |
| 樱花飘落 | `@keyframes sakuraFall` + `js/app.js` makePetal | 22 片、--drift1~4 随机漂移变量、animationend 重生，全部原样 |
| 扫描线 | `@keyframes scanlines` | keyframes 原样；覆盖层透明度调为 .12（原站元素层用法未暴露） |
| 播放器逻辑 | `js/app.js` 播放器段 | trackList 数组、--track-cover 封面变量、seek/volume、is-minimized 最小化、错误框、键盘可达——逐行对应 |
| UI 音效引擎 | `js/app.js` 音效段 | 悬停/点击双音轨 + 首次交互解锁 + 70ms 冷却，逐行对应；**默认关闭**、未配置 mp3 时显示 no mp3 config |
| 彩虹字 | `js/app.js` rainbow 段 | 7 色 900ms 循环逐字染色，用于品牌名与曲名 |
| 便签 | `js/app.js` sticky 段 | data-note-key + localStorage 自动保存 |
| 字体栈 | body | `"DotGothic16","MS UI Gothic","MS Gothic",Tahoma` |
| 页头横幅结构 | `#topheader` `.header-art` `.site-banner` `.browser-strip` | 180px 高、底部 17px 圆角图槽 + `object-fit:cover` + 27px 假地址栏丝带（`::after` 欢迎语右对齐 `#c77a94`），全部原样；原图是 pinimg 的 GIF（版权素材），`assets/banner.svg` 为本项目自绘占位（粉条纹+星星+贴纸字+原创双马尾耳机娘），直接替换 `<img src>` 即可换任何横幅 GIF |

### 取自 Musical 模板（musical.framer.media，实际渲染为 MONRAE 皮肤）

| 内容 | 说明 |
|---|---|
| 固定左侧栏骨架 | 品牌块（名字/简介/chips）+ 分区导航列表（标题+描述，当前项高亮）——MONRAE 的 Explore/About/Tour/Music Release 结构 |
| 巨字 hero 排版 | 大标题 / 发丝分隔线 / 小标签 / 发丝线 / 巨型数值 的节奏 |
| 分区清单 | 标注 / 曲库 / 训练 / 写标签 对应 MONRAE 的分区导航语义 |

### 本项目新增（工具逻辑，风格同源）

- 增益推子 + 大读数 + LUFS 联动、A/B 对比 chip、mark/flag/skip 日志
- 曲库 dry-run 表格、训练 todo、写标签清单
- 占位音频为运行时生成的静音 wav（不外链任何素材）

## 接下来

- 音效：`SOUND_CFG` 填两个 mp3（悬停音 + 点击音），mp3 放 `assets/sfx/`
- `patch_wiring.py` 是历史一次性补丁（已打进 app.js），发布打包时可剔除；`server_logs/` 同理
- 若要开源发布：先处理 `assets/gif/` 里画师不可考的二次元 GIF 授权（见下方版权提醒）

## 云端部署（上传/下载模式，适合非程序员用户）

同一套代码可以直接部署成云端网页：用户打开网页 → ☁ 上传歌曲 → 标注/训练/写标签全在服务器上 → ⬇ 打包下载。会话按 cookie 隔离，互相看不见对方的文件；「✕ 退出云端会话」或到期后自动清空。

```bash
# 任何有 Docker 的地方（VPS / Railway / Fly.io / HF Spaces 均可）
docker build -t replaygain-nyan .
docker run -p 8792:8792 replaygain-nyan
```

- 裸 VPS 不用 Docker 也行：`apt install ffmpeg` + `pip install -r requirements.txt` + `RG_HOST=0.0.0.0 python server/app.py`（特征提取依赖 ffmpeg，Docker 镜像里已装）
- 上限用环境变量调：`RG_MAX_FILE_MB`（单文件，默认 200）、`RG_SESSION_MB`（每会话，默认 1024）、`RG_SESSION_DAYS`（会话保留天数，默认 7，启动时自动清理过期目录）
- 注意：Render / HF Spaces 免费档的容器重启会清空 `uploads/` —— 打完标签记得及时下载；公开部署目前**接口无鉴权**，建议套一层访问控制（如 Cloudflare Access）再对外
- 训练/写标签在云端按会话独立出 `model_dir`，不同用户的口味模型互不干扰

## 部署到 Cloudflare（Tunnel + Access，推荐）

Cloudflare Pages/Workers 跑不了 Python 后端，所以正确姿势是 **Cloudflare Tunnel**：后端继续跑在自己电脑上（音乐文件不出本机），隧道把它安全地发布成一个公网网址，朋友直接打开就用；鉴权用两层——

1. **Cloudflare Access**（第一层，在 CF 后台配置）：Zero Trust → Access → Applications → 自托管应用，套在你的隧道域名上，策略选 Email OTP（或 Google 登录）。只有通过验证的请求才会到达你的电脑
2. **应用访问码**（第二层，代码已内置）：设了 `RG_ACCESS_CODE` 环境变量后，打开网页会先弹访问码输入框，填对才能进（cookie 记住 30 天）

```bash
# 一次性准备：安装 cloudflared → cloudflared tunnel login → cloudflared tunnel create rg-nyan
# 配置样例见 deploy/cloudflared.example.yml；一键启动脚本见 deploy/start-cloud.bat
```

应用侧还能再校验 Access 传来的 JWT（防止有人绕过边缘直连源站端口）：

```bash
set CF_ACCESS_TEAM=你的team名.cloudflareaccess.com   # 必填才启用
set CF_ACCESS_AUD=Application Audience (AUD) 标签的值 # 可选，更严格
```

## 模型市场（本地训练 → 云端分享）

爱折腾的用户自己听歌标注、训练模型；不想折腾的直接挑别人训练好的口味模型下载套用。规范详见 `../replaygain-autotagged/MODEL_HUB.md`。

- **训练产物**：LightGBM 原生 `.txt` 格式（v3）+ `meta.json` + `MODEL_CARD.json` 模型卡（样本数/MAE/版本/口味摘要，歌单默认隐藏、`--publish-tracks` 或打包时勾选才公开）。原生格式加载时不反序列化代码，才能安全分发
- **训练页**：训练完成自动显示模型卡；填署名/备注 → 「打包模型」下载发布 zip
- **写标签页**：模型来源三选一——我训练的 / 从 hub 挑一个（`RG_HUB_URL` 环境变量指向 models.json，GitHub raw 地址即可）/ 导入别人的模型包 zip
- **安全三道门**：只接受原生 .txt（绝不 joblib.load）、zip 白名单解压 + sha256 校验、特征管线版本不符拒绝加载

## 素材与授权注意

- bishoujosenshi 站点内容（文案/图片/音乐）© 原作者 beezykins，本项目只复用了其 CSS/JS 设计系统代码与布局语言，未搬运任何图片、音频或文案素材
- MONRAE 为 Framer 商店模板：布局思路复刻，未搬运其字体文件/背景图/文案
- DotGothic16 为 Google Fonts 开源字体（OFL）
