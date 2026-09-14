"""ElevenLabs family manifest: TTS, sound effects, music. Constants only - never import the SDK here."""

FAMILY = "elevenlabs"
PROVIDERS = {
    "tts": "bookreader.providers.elevenlabs.tts:ElevenLabsTTS",
    "music": "bookreader.providers.elevenlabs.music:ElevenLabsMusic",
    "sfx": "bookreader.providers.elevenlabs.sfx:ElevenLabsSFX",
}
EXTRA = "elevenlabs"
REQUIRED_SECRETS: tuple[str, ...] = ("ELEVENLABS_API_KEY",)
