# 🎬 Telegram Video Upscaler Bot

Upscale video **quality** (AI/ONNX models) and/or **FPS** (frame interpolation) — directly in Telegram.

---

## Features

- 🔍 **Quality upscale** using your custom ONNX models
- 🎬 **FPS upscale** via FFmpeg minterpolate (type any target FPS)
- ✨ **Both together** in one go
- ⏭️ **Skip either step** — fully optional
- 📦 Supports up to **100 MB** videos
- ⏱️ **Duration stays the same** — only quality/FPS changes

---

## Models Supported

| Key | Model Name |
|-----|-----------|
| `soft` | JV_Soft_Compact |
| `strong` | JV_Strong_V3_Compact |
| `strong_base` | JV_Strong_Compact |
| `ultra` | JV_SSJ_UltraCompact |
| `superultra` | JV_SSJ_SuperUltraCompact |

Place `.onnx` files in the `models/` folder.

---

## Local Setup

```bash
# 1. Install ffmpeg
sudo apt install ffmpeg      # Linux
brew install ffmpeg          # macOS

# 2. Install Python deps
pip install -r requirements.txt

# 3. Place your .onnx models in models/
cp *.onnx models/

# 4. Set your bot token
export TELEGRAM_BOT_TOKEN=your_token_here

# 5. Run
python bot.py
```

---

## Deploy to Render (Free Tier)

1. Push this folder to a **GitHub repo**
2. Go to [render.com](https://render.com) → New → **Background Worker**
3. Connect your repo
4. Set environment variable: `TELEGRAM_BOT_TOKEN = your_token`
5. Build command: `apt-get update && apt-get install -y ffmpeg && pip install -r requirements.txt`
6. Start command: `python bot.py`

> ⚠️ **Important:** Upload your `.onnx` model files to the `models/` folder in your repo before deploying.
> `.engine` files require a GPU and won't work on Render free tier.

---

## Bot Usage

```
1. Send a video (≤100 MB)
2. Bot shows video info (resolution, FPS)
3. Choose:
   [🔍 Quality Only]
   [🎬 FPS Only]
   [✨ Both]
4. If quality: pick an AI model
5. If FPS: type your target FPS (e.g. 60)
6. Wait for processing ☕
7. Receive upscaled video!
```

---

## Notes

- Processing time depends on video length and resolution
- On Render free tier, expect ~2-10 min per video
- Use `/cancel` to abort at any time
