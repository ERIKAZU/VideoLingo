import os, subprocess, time
from core._1_ytdlp import find_video_files
import cv2
import numpy as np
import platform
from core.utils import *

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

# Linux need to install google noto fonts: apt-get install fonts-noto
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
# Mac OS has different font names
elif platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"
    
def check_gpu_available():
    try:
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
        return 'h264_nvenc' in result.stdout
    except:
        return False

def merge_subtitles_to_video():
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Check resolution
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    if not os.path.exists(SRC_SRT) or not os.path.exists(TRANS_SRT):
        rprint("Subtitle files not found in the 'output' directory.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    # Build force_style strings with escaped commas for FFmpeg filter parsing
    # In FFmpeg subtitles filter, commas in force_style must be escaped
    escaped_font = FONT_NAME.replace(' ', r'\ ')
    escaped_trans_font = TRANS_FONT_NAME.replace(' ', r'\ ')
    ESC = r'\,'  # escaped comma for FFmpeg

    src_parts = [
        "FontSize=" + str(SRC_FONT_SIZE),
        "FontName=" + escaped_font,
        "PrimaryColour=" + SRC_FONT_COLOR,
        "OutlineColour=" + SRC_OUTLINE_COLOR,
        "OutlineWidth=" + str(SRC_OUTLINE_WIDTH),
        "ShadowColour=" + SRC_SHADOW_COLOR,
        "Shadow=1",
        "BorderStyle=1",
    ]
    src_style = ESC.join(src_parts)

    trans_parts = [
        "FontSize=" + str(TRANS_FONT_SIZE),
        "FontName=" + escaped_trans_font,
        "PrimaryColour=" + TRANS_FONT_COLOR,
        "OutlineColour=" + TRANS_OUTLINE_COLOR,
        "OutlineWidth=" + str(TRANS_OUTLINE_WIDTH),
        "BackColour=" + TRANS_BACK_COLOR,
        "Shadow=1",
        "Alignment=2",
        "MarginV=27",
        "BorderStyle=4",
    ]
    trans_style = ESC.join(trans_parts)

    vf_filter = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"subtitles={SRC_SRT}:force_style={src_style},"
        f"subtitles={TRANS_SRT}:force_style={trans_style}"
    )

    # Add vocabulary overlay if it exists (ASS file handles its own styling)
    VOCAB_ASS = f"{OUTPUT_DIR}/vocab.ass"
    if os.path.exists(VOCAB_ASS):
        vf_filter += f",subtitles={VOCAB_ASS}"
        rprint("[bold cyan]📚 Including vocabulary overlay in video[/bold cyan]")

    ffmpeg_cmd = [
        'ffmpeg', '-i', video_file,
        '-vf', vf_filter,
    ]

    ffmpeg_gpu = load_key("ffmpeg_gpu")
    if ffmpeg_gpu:
        rprint("[bold green]will use GPU acceleration.[/bold green]")
        ffmpeg_cmd.extend(['-c:v', 'h264_nvenc'])
    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
        if process.returncode == 0:
            rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
        else:
            rprint("\n❌ FFmpeg execution error")
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    merge_subtitles_to_video()