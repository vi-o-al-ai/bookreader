"""Anthropic family manifest: Claude text analysis. Constants only - never import the SDK here."""

FAMILY = "anthropic"
PROVIDERS = {
    "analysis": "bookreader.providers.anthropic.analysis:ClaudeAnalyzer",
}
EXTRA = "anthropic"
REQUIRED_SECRETS: tuple[str, ...] = ("ANTHROPIC_API_KEY",)
