"""Provider families and the registry that resolves them by name.

Import from here in application code::

    from bookreader.providers import build_providers, validate_providers, Providers
"""
from bookreader.providers.base import (
    BUILTIN_FAMILIES,
    CAPABILITIES,
    FamilyManifest,
    MusicGenerator,
    NullUsage,
    Provider,
    ProviderReport,
    Providers,
    SfxGenerator,
    TextAnalyzer,
    UsageSink,
    VoiceSynthesizer,
    build_provider,
    build_providers,
    describe_providers,
    load_manifest,
    resolve,
    selected_family,
    validate_providers,
    warmup_providers,
)

__all__ = [
    "BUILTIN_FAMILIES",
    "CAPABILITIES",
    "FamilyManifest",
    "MusicGenerator",
    "NullUsage",
    "Provider",
    "ProviderReport",
    "Providers",
    "SfxGenerator",
    "TextAnalyzer",
    "UsageSink",
    "VoiceSynthesizer",
    "build_provider",
    "build_providers",
    "describe_providers",
    "load_manifest",
    "resolve",
    "selected_family",
    "validate_providers",
    "warmup_providers",
]
