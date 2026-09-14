"""Mock family manifest: deterministic offline providers. Constants only - never import modules here."""

FAMILY = "mock"
PROVIDERS = {
    "analysis": "bookreader.providers.mock.analysis:HeuristicAnalyzer",
    "tts": "bookreader.providers.mock.tts:MockTTS",
    "music": "bookreader.providers.mock.music:MockMusic",
    "sfx": "bookreader.providers.mock.sfx:ProceduralSfx",
}
EXTRA = None
REQUIRED_SECRETS: tuple[str, ...] = ()
