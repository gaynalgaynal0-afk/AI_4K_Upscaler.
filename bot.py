import os
import logging
import asyncio
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from pyrogram import Client, filters
from pyrogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
API_ID    = int(os.environ["API_ID"])
API_HASH  = os.environ["API_HASH"]
BOT_TOKEN = os.environ["BOT_TOKEN"]

app = Client("upscaler_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
AVAILABLE_MODELS = {
    "soft":       ("JV_Soft_Compact",         "🟢 Soft Compact — smooth, natural"),
    "strong":     ("JV_Strong_V3_Compact",     "🔴 Strong V3 Compact — sharp detail"),
    "ultra":      ("JV_SSJ_UltraCompact",      "⚡ SSJ Ultra Compact — fast & crisp"),
    "superultra": ("JV_SSJ_SuperUltraCompact", "🚀 SSJ Super Ultra — max speed"),
}

def get_onnx_path(key: str) -> Path | None:
    name, _ = AVAILABLE_MODELS[key]
    p = MODELS_DIR / f"{name}.onnx"
    return p if p.exists() else None

# ── User session storage ──────────────────────────────────────────────────────
# { user_id: { input_path, mode, quality_key, target_fps, video_info, state } }
sessions = {}

STATE_CHOOSE_MODE  = "choose_mode"
STATE_CHOOSE_MODEL = "choose_model"
STATE_WAIT_FPS     = "wait_fps"

# ── Video Info ────────────────────────────────────────────────────────────────
def get_video_info(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0", path
    ]
    out = subprocess.check_output(cmd).decode().strip().split(",")
    width, height = int(out[0]), int(out[1])
    num, den = map(int, out[2].split("/"))
    fps = round(num / den, 3)
    return {"width": width, "height": height, "fps": fps}

# ── ONNX Upscale ──────────────────────────────────────────────────────────────
def upscale_frames_onnx(input_path: str, output_path: str, model_key: str) -> bool:
    import onnxruntime as ort
    import cv2

    model_path = get_onnx_path(model_key)
    if not model_path:
        logger.error(f"Model not found: {model_key}")
        return False

    info    = get_video_info(input_path)
    fps     = info["fps"]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    with tempfile.TemporaryDirectory() as frame_dir:
        frames_in  = os.path.join(frame_dir, "in_%06d.png")
        frames_out = os.path.join(frame_dir, "out_%06d.png")

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"fps={fps}", "-pix_fmt", "rgb24",
            frames_in
        ], check=True, capture_output=True)

        in_files = sorted(Path(frame_dir).glob("in_*.png"))
        if not in_files:
            return False

        logger.info(f"Processing {len(in_files)} frames...")

        for idx, img_path in enumerate(in_files, start=1):
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            tensor = img_rgb.astype(np.float16) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)

            result = session.run(["output"], {"input": tensor})[0]
            result = np.squeeze(result, axis=0)
            result = np.transpose(result, (1, 2, 0))
            result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(frame_dir, f"out_{idx:06d}.png"), result_bgr)

            if idx % 50 == 0:
                logger.info(f"  {idx}/{len(in_files)} frames done")

        temp_noaudio = output_path + "_noaudio.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frames_out,
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            temp_noaudio
        ], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_noaudio, "-i", input_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a?",
        "-shortest", output_path
    ], check=True, capture_output=True)

    os.remove(temp_noaudio)
    return True

# ── FPS Interpolation ─────────────────────────────────────────────────────────
def upscale_fps(input_path: str, output_path: str, target_fps: float) -> bool:
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", (
            f"minterpolate=fps={target_fps}:"
            "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        ),
        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
        "-c:a", "copy", output_path
    ], check=True, capture_output=True)
    return True

# ── Keyboards ─────────────────────────────────────────────────────────────────
def mode_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 Quality Only (AI 2×)", callback_data="mode_quality")],
        [InlineKeyboardButton("🎬 FPS Only",              callback_data="mode_fps")],
        [InlineKeyboardButton("✨ Both (Quality + FPS)",  callback_data="mode_both")],
    ])

def model_keyboard():
    rows = []
    for key, (_, label) in AVAILABLE_MODELS.items():
        rows.append([InlineKeyboardButton(label, callback_data=f"model_{key}")])
    return InlineKeyboardMarkup(rows)

# ── /start ────────────────────────────────────────────────────────────────────
@app.on_message(filters.command("start"))
async def start(client: Client, message: Message):
    await message.reply(
        "👋 *Welcome to the Video Upscaler Bot!*\n\n"
        "Send me a video (up to **2 GB**) and I'll:\n"
        "• 🔍 Upscale *quality* with real AI models (2× resolution)\n"
        "• 🎬 Upscale *FPS* via frame interpolation\n"
        "• ✨ Or *both* at once\n\n"
        "Just send a video to get started. Use /cancel to stop anytime.",
        parse_mode="markdown"
    )

# ── /cancel ───────────────────────────────────────────────────────────────────
@app.on_message(filters.command("cancel"))
async def cancel(client: Client, message: Message):
    uid = message.from_user.id
    sess = sessions.pop(uid, {})
    p = sess.get("input_path")
    if p and os.path.exists(p):
        try: os.remove(p)
        except: pass
    await message.reply("❌ Cancelled. Send a new video whenever you're ready.")

# ── Receive Video ─────────────────────────────────────────────────────────────
@app.on_message(filters.video | filters.document)
async def receive_video(client: Client, message: Message):
    uid  = message.from_user.id
    media = message.video or message.document

    if not media:
        return

    # Check it's a video document
    if message.document and not message.document.mime_type.startswith("video"):
        return

    size_mb = media.file_size / (1024 * 1024)
    if size_mb > 2048:
        await message.reply("❌ File too large. Max is 2 GB.")
        return

    status = await message.reply(f"📥 Downloading your video ({size_mb:.1f} MB)…")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()

    await client.download_media(message, file_name=tmp.name)

    try:
        info = get_video_info(tmp.name)
        info_text = (
            f"📹 *Video detected*\n"
            f"• Resolution: `{info['width']}×{info['height']}`\n"
            f"• FPS: `{info['fps']}`\n\n"
        )
    except Exception:
        info      = {}
        info_text = "📹 Video received.\n\n"

    sessions[uid] = {
        "input_path":  tmp.name,
        "video_info":  info,
        "mode":        None,
        "quality_key": None,
        "target_fps":  None,
        "state":       STATE_CHOOSE_MODE,
    }

    await status.delete()
    await message.reply(
        info_text + "What would you like to upscale?",
        reply_markup=mode_keyboard(),
        parse_mode="markdown"
    )

# ── Callback: Mode & Model ────────────────────────────────────────────────────
@app.on_callback_query()
async def handle_callback(client: Client, query: CallbackQuery):
    uid  = query.from_user.id
    data = query.data
    sess = sessions.get(uid)

    if not sess:
        await query.answer("Session expired. Please send a new video.", show_alert=True)
        return

    await query.answer()

    # ── Mode selection ──
    if data.startswith("mode_"):
        sess["mode"]  = data
        sess["state"] = STATE_CHOOSE_MODEL

        if data == "mode_fps":
            info     = sess.get("video_info", {})
            fps_text = f" (current: `{info.get('fps', '?')}` fps)" if info else ""
            sess["state"] = STATE_WAIT_FPS
            await query.edit_message_text(
                f"🎬 *FPS Upscale*{fps_text}\n\nType your target FPS (e.g. `60`):",
                parse_mode="markdown"
            )
        else:
            await query.edit_message_text(
                "🔍 *Choose your AI upscale model:*\n_(all models upscale 2×)_",
                reply_markup=model_keyboard(),
                parse_mode="markdown"
            )

    # ── Model selection ──
    elif data.startswith("model_"):
        model_key        = data.replace("model_", "")
        sess["quality_key"] = model_key
        _, label         = AVAILABLE_MODELS[model_key]

        if sess["mode"] == "mode_both":
            info     = sess.get("video_info", {})
            fps_text = f" (current: `{info.get('fps', '?')}` fps)" if info else ""
            sess["state"] = STATE_WAIT_FPS
            await query.edit_message_text(
                f"✅ Model: *{label}*\n\n"
                f"🎬 Now type your target FPS{fps_text} (e.g. `60`):",
                parse_mode="markdown"
            )
        else:
            # Quality only — start processing
            sess["state"] = None
            await query.edit_message_text(
                f"✅ Model: *{label}*\n\n⚙️ Processing… this may take a few minutes ☕",
                parse_mode="markdown"
            )
            asyncio.create_task(process_video(client, uid, query.message.chat.id))

# ── Text: FPS input ───────────────────────────────────────────────────────────
@app.on_message(filters.text & ~filters.command(["start", "cancel"]))
async def handle_text(client: Client, message: Message):
    uid  = message.from_user.id
    sess = sessions.get(uid)

    if not sess or sess.get("state") != STATE_WAIT_FPS:
        return

    try:
        target_fps = float(message.text.strip())
        if not (1 <= target_fps <= 240):
            raise ValueError
    except ValueError:
        await message.reply("⚠️ Enter a valid FPS between 1 and 240.")
        return

    info       = sess.get("video_info", {})
    orig_fps   = info.get("fps", 0)
    multiplier = f"×{round(target_fps / orig_fps, 2)}" if orig_fps else ""

    sess["target_fps"] = target_fps
    sess["state"]      = None

    await message.reply(
        f"✅ Target FPS: `{target_fps}` {multiplier}\n\n"
        "⚙️ Processing… this may take a few minutes ☕",
        parse_mode="markdown"
    )
    asyncio.create_task(process_video(client, uid, message.chat.id))

# ── Core Processing ───────────────────────────────────────────────────────────
async def process_video(client: Client, uid: int, chat_id: int):
    sess        = sessions.get(uid, {})
    input_path  = sess.get("input_path")
    mode        = sess.get("mode")
    quality_key = sess.get("quality_key")
    target_fps  = sess.get("target_fps")
    loop        = asyncio.get_event_loop()
    current     = input_path

    try:
        # Step 1: Quality upscale
        if mode in ("mode_quality", "mode_both") and quality_key:
            await client.send_message(chat_id, "🔍 Running AI upscale (2×)… extracting frames...")
            q_out = input_path + "_quality.mp4"
            ok    = await loop.run_in_executor(
                None, upscale_frames_onnx, current, q_out, quality_key
            )
            if not ok:
                await client.send_message(chat_id, "❌ Quality upscale failed — model file missing.")
                return
            current = q_out
            await client.send_message(chat_id, "✅ Quality upscale done!")

        # Step 2: FPS interpolation
        if mode in ("mode_fps", "mode_both") and target_fps:
            await client.send_message(chat_id, f"🎬 Interpolating to {target_fps} FPS…")
            f_out = input_path + "_fps.mp4"
            await loop.run_in_executor(None, upscale_fps, current, f_out, target_fps)
            current = f_out
            await client.send_message(chat_id, "✅ FPS upscale done!")

        # Upload result
        size_mb = os.path.getsize(current) / (1024 * 1024)
        await client.send_message(chat_id, f"📤 Uploading result ({size_mb:.1f} MB)…")

        await client.send_video(
            chat_id,
            video=current,
            caption="✅ *Done! Here's your upscaled video.*",
            parse_mode="markdown",
            supports_streaming=True,
        )

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error: {stderr}")
        await client.send_message(chat_id, "❌ Processing failed. Check the video format.")
    except Exception as e:
        logger.exception("Unexpected error")
        await client.send_message(chat_id, f"❌ Error: {str(e)}")
    finally:
        for f in [
            input_path,
            input_path + "_quality.mp4",
            input_path + "_fps.mp4",
            input_path + "_quality.mp4_fps.mp4",
        ]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
        sessions.pop(uid, None)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("🚀 Bot started with 2GB support via Pyrogram.")
    app.run()
