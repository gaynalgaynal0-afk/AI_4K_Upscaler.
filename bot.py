import os
import logging
import asyncio
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters, ConversationHandler
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── States ───────────────────────────────────────────────────────────────────
CHOOSE_MODE, CHOOSE_MODEL, WAIT_FPS = range(3)

# ── Models ───────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
AVAILABLE_MODELS = {
    "soft":        ("JV_Soft_Compact",          "🟢 Soft Compact — smooth, natural"),
    "strong":      ("JV_Strong_V3_Compact",      "🔴 Strong V3 Compact — sharp detail"),
    "ultra":       ("JV_SSJ_UltraCompact",       "⚡ SSJ Ultra Compact — fast & crisp"),
    "superultra":  ("JV_SSJ_SuperUltraCompact",  "🚀 SSJ Super Ultra — max speed"),
}

def get_onnx_path(key: str) -> Path | None:
    name, _ = AVAILABLE_MODELS[key]
    p = MODELS_DIR / f"{name}.onnx"
    return p if p.exists() else None

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

# ── Real ONNX Upscale ─────────────────────────────────────────────────────────
def upscale_frames_onnx(input_path: str, output_path: str, model_key: str) -> bool:
    """
    Extract every frame → run through ONNX super-resolution (float16, NCHW, RGB 0-1, 2x) → rebuild video.
    Audio is copied from original. Duration is identical.
    """
    import onnxruntime as ort
    import cv2

    model_path = get_onnx_path(model_key)
    if model_path is None:
        logger.error(f"ONNX model not found: {model_key}")
        return False

    info    = get_video_info(input_path)
    fps     = info["fps"]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    with tempfile.TemporaryDirectory() as frame_dir:
        frames_pattern_in  = os.path.join(frame_dir, "in_%06d.png")
        frames_pattern_out = os.path.join(frame_dir, "out_%06d.png")

        # 1. Extract frames at original fps
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"fps={fps}",
            "-pix_fmt", "rgb24",          # ensure consistent color space
            frames_pattern_in
        ], check=True, capture_output=True)

        in_files = sorted(Path(frame_dir).glob("in_*.png"))
        if not in_files:
            logger.error("No frames extracted")
            return False

        logger.info(f"Processing {len(in_files)} frames with model {model_key}...")

        for idx, img_path in enumerate(in_files, start=1):
            # Read as RGB uint8
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Normalize to float16, shape (1, 3, H, W)
            tensor = img_rgb.astype(np.float16) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))      # HWC → CHW
            tensor = np.expand_dims(tensor, axis=0)        # → (1, 3, H, W)

            # Run inference
            result = session.run(["output"], {"input": tensor})[0]

            # Back to uint8 BGR
            result = np.squeeze(result, axis=0)            # (3, H, W)
            result = np.transpose(result, (1, 2, 0))       # CHW → HWC
            result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            out_file = os.path.join(frame_dir, f"out_{idx:06d}.png")
            cv2.imwrite(out_file, result_bgr)

            if idx % 50 == 0:
                logger.info(f"  {idx}/{len(in_files)} frames done")

        # 2. Rebuild video from upscaled frames (no audio)
        temp_noaudio = output_path + "_noaudio.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frames_pattern_out,
            "-c:v", "libx264", "-crf", "16", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            temp_noaudio
        ], check=True, capture_output=True)

    # 3. Merge original audio back (duration preserved)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_noaudio,
        "-i", input_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a?",
        "-shortest",
        output_path
    ], check=True, capture_output=True)

    os.remove(temp_noaudio)
    return True

# ── FPS Interpolation ─────────────────────────────────────────────────────────
def upscale_fps(input_path: str, output_path: str, target_fps: float) -> bool:
    """Interpolate to target_fps. Duration stays the same."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", (
            f"minterpolate=fps={target_fps}:"
            "mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        ),
        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ], check=True, capture_output=True)
    return True

# ── /start ────────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to the Video Upscaler Bot!*\n\n"
        "Send me a video (up to 100 MB) and I'll:\n"
        "• 🔍 Upscale *quality* with real AI models (2× resolution)\n"
        "• 🎬 Upscale *FPS* via frame interpolation\n"
        "• ✨ Or *both* at once\n\n"
        "Just send a video to get started. Use /cancel to stop anytime.",
        parse_mode="Markdown"
    )

# ── Receive Video ─────────────────────────────────────────────────────────────
async def receive_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg   = update.message
    video = msg.video or msg.document

    if not video:
        await msg.reply_text("⚠️ Please send a video file.")
        return ConversationHandler.END

    size_mb = video.file_size / (1024 * 1024)
    if size_mb > 100:
        await msg.reply_text(f"❌ File too large ({size_mb:.1f} MB). Max is 100 MB.")
        return ConversationHandler.END

    status = await msg.reply_text("📥 Downloading your video...")

    tg_file = await context.bot.get_file(video.file_id)
    suffix  = ".mp4"
    if hasattr(video, "file_name") and video.file_name:
        suffix = Path(video.file_name).suffix or ".mp4"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    await tg_file.download_to_drive(tmp.name)
    tmp.close()

    context.user_data.clear()
    context.user_data["input_path"]  = tmp.name
    context.user_data["quality_key"] = None
    context.user_data["target_fps"]  = None

    try:
        info = get_video_info(tmp.name)
        context.user_data["video_info"] = info
        info_text = (
            f"📹 *Video detected*\n"
            f"• Resolution: `{info['width']}×{info['height']}`\n"
            f"• FPS: `{info['fps']}`\n\n"
        )
    except Exception:
        info_text = "📹 Video received.\n\n"
        context.user_data["video_info"] = {}

    await status.delete()

    keyboard = [
        [InlineKeyboardButton("🔍 Quality Only (AI 2×)",     callback_data="mode_quality")],
        [InlineKeyboardButton("🎬 FPS Only",                  callback_data="mode_fps")],
        [InlineKeyboardButton("✨ Both (Quality + FPS)",      callback_data="mode_both")],
    ]
    await msg.reply_text(
        info_text + "What would you like to upscale?",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown"
    )
    return CHOOSE_MODE

# ── Mode Selection ─────────────────────────────────────────────────────────────
async def choose_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    mode  = query.data
    context.user_data["mode"] = mode

    if mode == "mode_fps":
        info     = context.user_data.get("video_info", {})
        fps_text = f" (current: `{info.get('fps', '?')}` fps)" if info else ""
        await query.edit_message_text(
            f"🎬 *FPS Upscale*{fps_text}\n\nType your target FPS (e.g. `60`):",
            parse_mode="Markdown"
        )
        return WAIT_FPS

    await query.edit_message_text(
        "🔍 *Choose your AI upscale model:*\n_(all models are 2× resolution)_",
        reply_markup=model_keyboard(),
        parse_mode="Markdown"
    )
    return CHOOSE_MODEL

def model_keyboard():
    rows = []
    for key, (_, label) in AVAILABLE_MODELS.items():
        rows.append([InlineKeyboardButton(label, callback_data=f"model_{key}")])
    return InlineKeyboardMarkup(rows)

# ── Model Selection ───────────────────────────────────────────────────────────
async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query     = update.callback_query
    await query.answer()
    model_key = query.data.replace("model_", "")
    context.user_data["quality_key"] = model_key

    _, label  = AVAILABLE_MODELS[model_key]
    mode      = context.user_data.get("mode")

    if mode == "mode_both":
        info     = context.user_data.get("video_info", {})
        fps_text = f" (current: `{info.get('fps', '?')}` fps)" if info else ""
        await query.edit_message_text(
            f"✅ Model: *{label}*\n\n"
            f"🎬 Now type your target FPS{fps_text} (e.g. `60`):",
            parse_mode="Markdown"
        )
        return WAIT_FPS

    # quality only
    await query.edit_message_text(
        f"✅ Model: *{label}*\n\n⚙️ Processing… this may take a few minutes ☕",
        parse_mode="Markdown"
    )
    await process_video(context, query.message.chat_id)
    return ConversationHandler.END

# ── FPS Input ─────────────────────────────────────────────────────────────────
async def wait_fps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    try:
        target_fps = float(text)
        if not (1 <= target_fps <= 240):
            raise ValueError
    except ValueError:
        await update.message.reply_text("⚠️ Enter a valid FPS between 1 and 240.")
        return WAIT_FPS

    info       = context.user_data.get("video_info", {})
    orig_fps   = info.get("fps", 0)
    multiplier = f"×{round(target_fps / orig_fps, 2)}" if orig_fps else ""

    context.user_data["target_fps"] = target_fps

    await update.message.reply_text(
        f"✅ Target FPS: `{target_fps}` {multiplier}\n\n"
        "⚙️ Processing… this may take a few minutes ☕",
        parse_mode="Markdown"
    )
    await process_video(context, update.message.chat_id)
    return ConversationHandler.END

# ── Core Processing ───────────────────────────────────────────────────────────
async def process_video(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    bot         = context.bot
    input_path  = context.user_data["input_path"]
    mode        = context.user_data.get("mode")
    quality_key = context.user_data.get("quality_key")
    target_fps  = context.user_data.get("target_fps")
    loop        = asyncio.get_event_loop()
    current     = input_path

    try:
        # Step 1: Quality upscale with real ONNX model
        if mode in ("mode_quality", "mode_both") and quality_key:
            await bot.send_message(chat_id, "🔍 Running AI upscale (2×)… extracting frames...")
            q_out = input_path + "_quality.mp4"
            ok    = await loop.run_in_executor(
                None, upscale_frames_onnx, current, q_out, quality_key
            )
            if not ok:
                await bot.send_message(chat_id, "❌ Quality upscale failed — model file missing from models/ folder.")
                return
            current = q_out
            await bot.send_message(chat_id, "✅ Quality upscale done!")

        # Step 2: FPS interpolation
        if mode in ("mode_fps", "mode_both") and target_fps:
            await bot.send_message(chat_id, f"🎬 Interpolating to {target_fps} FPS…")
            f_out = input_path + "_fps.mp4"
            await loop.run_in_executor(
                None, upscale_fps, current, f_out, target_fps
            )
            current = f_out
            await bot.send_message(chat_id, "✅ FPS upscale done!")

        # Send result
        size_mb = os.path.getsize(current) / (1024 * 1024)
        await bot.send_message(chat_id, f"📤 Uploading ({size_mb:.1f} MB)…")

        with open(current, "rb") as f:
            await bot.send_video(
                chat_id,
                video=f,
                caption="✅ *Done! Here's your upscaled video.*",
                parse_mode="Markdown",
                supports_streaming=True,
                read_timeout=300,
                write_timeout=300,
            )

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error: {stderr}")
        await bot.send_message(chat_id, "❌ Processing failed. Make sure the video format is supported.")
    except Exception as e:
        logger.exception("Unexpected error during processing")
        await bot.send_message(chat_id, f"❌ Unexpected error: {str(e)}")
    finally:
        # Clean up all temp files
        for f in [input_path,
                  input_path + "_quality.mp4",
                  input_path + "_fps.mp4",
                  input_path + "_quality.mp4_fps.mp4"]:
            if f and os.path.exists(f):
                try: os.remove(f)
                except: pass
        context.user_data.clear()

# ── Cancel ────────────────────────────────────────────────────────────────────
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("❌ Cancelled. Send a new video whenever you're ready.")
    return ConversationHandler.END

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app   = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[
            MessageHandler(filters.VIDEO | filters.Document.VIDEO, receive_video)
        ],
        states={
            CHOOSE_MODE:  [CallbackQueryHandler(choose_mode,  pattern="^mode_")],
            CHOOSE_MODEL: [CallbackQueryHandler(choose_model, pattern="^model_")],
            WAIT_FPS:     [MessageHandler(filters.TEXT & ~filters.COMMAND, wait_fps)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)

    logger.info("🚀 Bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
