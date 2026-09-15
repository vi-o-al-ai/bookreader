"""bookreader.audio.export - optional MP3 export through an ffmpeg binary found on PATH."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT_S = 600


def ffmpeg_path() -> str | None:
    """Absolute path of the ``ffmpeg`` executable, or None when it is not on PATH."""
    return shutil.which("ffmpeg")


def export_mp3(wav: Path, mp3: Path) -> bool:
    """Encode *wav* to *mp3* with libmp3lame (VBR quality 4). Returns True on success.

    Never raises: a missing ffmpeg, a non-zero exit, a timeout or an OS error is logged and yields False.
    The caller decides whether export is wanted at all (``settings.mp3 != 'off'``).
    """
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        log.info("ffmpeg not found on PATH; skipping mp3 export of %s", wav)
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "4", str(mp3)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("mp3 export of %s failed: %s", wav, exc)
        return False
    if result.returncode != 0 or not Path(mp3).exists():
        log.warning("mp3 export of %s failed (exit %d): %s", wav, result.returncode, result.stderr.strip())
        return False
    return True
