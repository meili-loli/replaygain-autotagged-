# 部署指南（选一个平台）

> 2026-10 更新：Hugging Face 免费档已不含 Docker/Gradio（只剩静态页），
> 所以这份指南改成通用 Docker 部署。前端壳 / 模型索引（models.json）这类
> 纯静态文件仍然可以放 HF Static 或 Cloudflare Pages（免费），只是"干活的
> 后端"需要下面任一平台。

按省事程度排序：

## 方案 A · Render 免费档（最容易，5 分钟）

免费给 512MB 内存 / 0.1 CPU，闲置 15 分钟休眠（有人访问自动唤醒，约等 1 分钟）。
适合朋友间小规模使用；大曲库训练会慢，记得配 `RG_WORKERS=1`。

1. 把本文件夹推到一个 GitHub 仓库（新建仓库 → Upload files 或 git push）
2. 登录 https://render.com（可直接用 GitHub 账号登录）
3. New → Web Service → 连接刚才的仓库 → Runtime 会自动识别为 Docker
4. Instance Type 选 **Free**；Environment → Add environment variable：
   - `RG_ACCESS_CODE` = 你定的访问码
   - `RG_WORKERS` = `1`
5. Create → 等 build（首次约 5-10 分钟）→ 得到 `xxx.onrender.com` 链接
6. 打开链接 → 输访问码 → 开用；把链接发给朋友

注意：免费档磁盘是临时的，重启清空上传——会话制设计不受影响；
15 分钟无人访问会休眠，首次唤醒慢。

## 方案 B · Oracle Cloud 永久免费 VM（最强，永不休眠）

永久免费 4 核 ARM + 24GB 内存 + 200GB 盘，是真服务器，能扛大曲库训练。

1. 注册 https://oracle.com/cloud/free/（要信用卡验证，不扣费）
2. 建 ARM 实例（A1.Flex，建议 2 核 8G 起步；热门区可能显示缺容量，换区/多试几次）
3. SSH 进去后：
```bash
curl -fsSL https://get.docker.com | sh
git clone <你的仓库> && cd <仓库>
docker build -t rg-nyan .
docker run -d --name rg-nyan -p 80:8792 -e RG_HOST=0.0.0.0 \
  -e RG_ACCESS_CODE=你的访问码 rg-nyan
```
4. 开安全组 80 端口（实例 → 子网 → 安全列表），浏览器访问 `http://实例公网IP`

## 方案 C · Google Cloud e2-micro 永久免费（备选）

美国区 e2-micro 永久免费（1GB 内存），部署同方案 B（apt 装 ffmpeg +
python3.11 -m venv + pip install 或 Docker）。1GB 内存适合小规模，
训练大曲库会吃紧。

## 方案 D · Cloudflare Tunnel + 自己电脑（零服务器）

`deploy/start-cloud.bat` 一键启动（前一个贡献包里），电脑开着就有公网链接。
适合临时给朋友演示，不适合 7x24。

## 免费额度速查

| 平台 | 配置 | 休眠策略 | 门槛 |
|---|---|---|---|
| Render Free | 512MB / 0.1CPU | 闲置 15 分钟 | GitHub 仓库即可 |
| Oracle Always Free | 4 核 ARM / 24GB | 不休眠 | 信用卡验证 + 蹲容量 |
| GCP e2-micro | 1GB | 不休眠 | 信用卡验证（美国区） |
| Cloudflare Tunnel | 取决于你的电脑 | 跟电脑走 | cloudflared 一条命令 |
| HF Spaces | ~~Docker 免费~~ | — | Docker 已转付费（仅剩 Static） |
