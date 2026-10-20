@echo off
REM ============================================================
REM replaygain-nyan: start backend + cloudflared tunnel together
REM Prerequisites (one-time):
REM   1. cloudflared installed and on PATH
REM   2. "cloudflared tunnel login" done
REM   3. "cloudflared tunnel create rg-nyan" done
REM   4. copy deploy\cloudflared.example.yml to %USERPROFILE%\.cloudflared\config.yml
REM      and fill in your tunnel UUID / credentials / hostname
REM   5. set your own access code below
REM Then in Cloudflare Zero Trust dashboard: add a self-hosted Access
REM application on your hostname with an email-OTP policy.
REM ============================================================
chcp 65001 >nul
set RG_ACCESS_CODE=change-me-nyan
REM Optional: verify the Cloudflare Access JWT in the app as well
REM set CF_ACCESS_TEAM=your-team.cloudflareaccess.com
REM set CF_ACCESS_AUD=<aud from the Access app>

cd /d %~dp0..
start "rg-nyan backend" python server\app.py
cloudflared tunnel run rg-nyan
