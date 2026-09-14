"""Local family manifest: open-source models (Piper/Kokoro TTS, MusicGen, AudioGen). Constants only."""

FAMILY = "local"
PROVIDERS = {
    "tts": "bookreader.providers.local.tts:LocalTTS",
    "music": "bookreader.providers.local.music:MusicGenMusic",
    "sfx": "bookreader.providers.local.sfx:LocalSFX",
}
EXTRA = "local"
REQUIRED_SECRETS: tuple[str, ...] = ()
