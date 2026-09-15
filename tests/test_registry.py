"""Provider registry end to end: manifests, resolution, validation, building, describing, warmup."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from typing import Any

import pytest

from bookreader.providers import base
from bookreader.settings import Settings
from bookreader.types import CastBible, Chunk, ChunkAnalysis, ProviderConfigError

MOCK_ANALYSIS = "bookreader.providers.mock.analysis"
SDK_MODULES = ("anthropic", "elevenlabs", "torch", "transformers", "piper", "kokoro", "audiocraft")


@pytest.fixture
def mock_analysis(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """The real heuristic analyzer module (WP2). Until it lands, a minimal stand-in with the same
    contract (family 'mock', check() -> [], from_settings(settings, usage) -> cls()) is installed
    so the registry behaviour under test does not depend on another package's schedule."""
    try:
        return importlib.import_module(MOCK_ANALYSIS)
    except ModuleNotFoundError as exc:
        if exc.name != MOCK_ANALYSIS:
            raise
    stub = types.ModuleType(MOCK_ANALYSIS)

    class HeuristicAnalyzer:
        family = "mock"
        cache_version = "1"
        model_id = "heuristic-1"

        @classmethod
        def check(cls, settings: Settings) -> list[str]:
            return []

        @classmethod
        def from_settings(cls, settings: Settings, usage: Any = None) -> "HeuristicAnalyzer":
            return cls()

        def warmup(self) -> None:
            return None

        def analyze_chunk(self, chunk: Chunk, bible: CastBible) -> ChunkAnalysis:
            return ChunkAnalysis()

    stub.HeuristicAnalyzer = HeuristicAnalyzer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, MOCK_ANALYSIS, stub)
    return stub


def install_family(monkeypatch: pytest.MonkeyPatch, package: str, family: str, providers: dict[str, type], extra: str | None = None) -> None:
    """Install a third-party family package plus one module per capability in sys.modules."""
    pkg = types.ModuleType(package)
    pkg.FAMILY = family  # type: ignore[attr-defined]
    pkg.PROVIDERS = {cap: f"{package}.{cap}:{cls.__name__}" for cap, cls in providers.items()}  # type: ignore[attr-defined]
    pkg.EXTRA = extra  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, package, pkg)
    for cap, cls in providers.items():
        mod = types.ModuleType(f"{package}.{cap}")
        setattr(mod, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, f"{package}.{cap}", mod)


# --------------------------------------------------------------------------- manifests and resolution


def test_mock_family_resolves_all_four_capabilities(mock_analysis: types.ModuleType):
    manifest = base.load_manifest("mock")
    assert manifest.family == "mock" and manifest.extra is None and manifest.required_secrets == ()
    assert set(manifest.providers) == set(base.CAPABILITIES)
    classes = {cap: base.resolve(cap, "mock") for cap in base.CAPABILITIES}
    assert classes["analysis"].__name__ == "HeuristicAnalyzer"
    assert classes["tts"].__name__ == "MockTTS"
    assert classes["music"].__name__ == "MockMusic"
    assert classes["sfx"].__name__ == "ProceduralSfx"
    assert all(cls.family == "mock" for cls in classes.values())


def test_builtin_manifests_declare_extras_and_secrets():
    assert base.load_manifest("elevenlabs").providers == {
        "tts": "bookreader.providers.elevenlabs.tts:ElevenLabsTTS",
        "music": "bookreader.providers.elevenlabs.music:ElevenLabsMusic",
        "sfx": "bookreader.providers.elevenlabs.sfx:ElevenLabsSFX",
    }
    assert base.load_manifest("elevenlabs").required_secrets == ("ELEVENLABS_API_KEY",)
    local = base.load_manifest("local")
    assert local.extra == "local" and local.required_secrets == () and set(local.providers) == {"tts", "music", "sfx"}
    for capability in ("tts", "music", "sfx"):
        assert base.resolve(capability, "elevenlabs").family == "elevenlabs"
        assert base.resolve(capability, "local").family == "local"


def test_unknown_family_error_lists_builtin_families():
    with pytest.raises(ProviderConfigError) as exc:
        base.load_manifest("acme")
    message = str(exc.value)
    assert "unknown provider family 'acme'" in message
    for family in base.BUILTIN_FAMILIES:
        assert family in message
    with pytest.raises(ProviderConfigError):
        base.resolve("tts", "does.not.exist")


def test_elevenlabs_does_not_provide_analysis():
    with pytest.raises(ProviderConfigError) as exc:
        base.resolve("analysis", "elevenlabs")
    message = str(exc.value)
    assert "provider family 'elevenlabs' does not provide 'analysis'" in message
    assert "music, sfx, tts" in message
    with pytest.raises(ProviderConfigError) as exc2:
        base.resolve("analysis", "local")
    assert "does not provide 'analysis'" in str(exc2.value)


def test_dotted_third_party_family_via_sys_modules(monkeypatch: pytest.MonkeyPatch):
    class AcmeTTS:
        family = "acme"
        cache_version = "acme-1"
        max_chars = 1000

        @classmethod
        def check(cls, settings: Settings) -> list[str]:
            return ["acme is beta"]

        @classmethod
        def from_settings(cls, settings: Settings, usage: Any = None) -> "AcmeTTS":
            return cls()

    install_family(monkeypatch, "acme_voices.family", "acme", {"tts": AcmeTTS}, extra="acme")
    manifest = base.load_manifest("acme_voices.family")
    assert manifest.family == "acme" and manifest.extra == "acme" and manifest.providers == {"tts": "acme_voices.family.tts:AcmeTTS"}
    assert base.resolve("tts", "acme_voices.family") is AcmeTTS

    settings = Settings.from_env({"BOOKREADER_TTS_PROVIDER": "acme_voices.family"})
    assert isinstance(base.build_provider("tts", settings), AcmeTTS)
    entries = {e["capability"]: e for e in base.describe_providers(settings)}
    assert entries["tts"] == {"capability": "tts", "family": "acme_voices.family", "ok": True,
                              "warnings": ["acme is beta"], "error": None, "class": "AcmeTTS"}
    with pytest.raises(ProviderConfigError) as exc:
        base.resolve("music", "acme_voices.family")
    assert "does not provide 'music'" in str(exc.value)


def test_family_package_without_providers_dict_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "not_a_family", types.ModuleType("not_a_family"))
    with pytest.raises(ProviderConfigError) as exc:
        base.load_manifest("not_a_family")
    assert "no PROVIDERS dict" in str(exc.value)


# --------------------------------------------------------------------------- validation


def test_validate_providers_aggregates_two_missing_keys_into_one_error(mock_analysis: types.ModuleType):
    settings = Settings.from_env({"BOOKREADER_ANALYSIS_PROVIDER": "anthropic", "BOOKREADER_TTS_PROVIDER": "elevenlabs"})
    with pytest.raises(ProviderConfigError) as exc:
        base.validate_providers(settings)
    message = str(exc.value)
    assert message.startswith("provider configuration invalid:")
    assert "[analysis=anthropic]" in message and "ANTHROPIC_API_KEY" in message and "bookreader[anthropic]" in message
    assert "[tts=elevenlabs]" in message and "ELEVENLABS_API_KEY" in message and "bookreader[elevenlabs]" in message
    assert "[music=mock]" not in message and "[sfx=mock]" not in message


def test_validate_providers_reports_ok_for_all_mock_and_elevenlabs_with_key(mock_analysis: types.ModuleType, monkeypatch: pytest.MonkeyPatch):
    reports = base.validate_providers(Settings.from_env({}))
    assert [(r.capability, r.family, r.ok, r.warnings, r.error) for r in reports] == [
        (cap, "mock", True, [], None) for cap in base.CAPABILITIES
    ]
    assert [r.class_name for r in reports] == ["HeuristicAnalyzer", "MockTTS", "MockMusic", "ProceduralSfx"]

    settings = Settings.from_env({"BOOKREADER_MUSIC_PROVIDER": "elevenlabs", "BOOKREADER_SFX_PROVIDER": "elevenlabs", "ELEVENLABS_API_KEY": "k"})
    ok = {r.capability: r for r in base.validate_providers(settings)}
    assert ok["music"].class_name == "ElevenLabsMusic" and ok["sfx"].class_name == "ElevenLabsSFX" and ok["sfx"].ok

    monkeypatch.setitem(sys.modules, "elevenlabs", None)
    with pytest.raises(ProviderConfigError) as exc:
        base.validate_providers(settings)
    assert str(exc.value).count("bookreader[elevenlabs]") == 2


def test_validate_providers_surfaces_local_warnings_and_errors(mock_analysis: types.ModuleType, monkeypatch: pytest.MonkeyPatch):
    for name in ("piper", "torch", "transformers", "audiocraft"):
        monkeypatch.setitem(sys.modules, name, None)
    settings = Settings.from_env({"BOOKREADER_SFX_PROVIDER": "local"})
    sfx = {r.capability: r for r in base.validate_providers(settings)}["sfx"]
    assert sfx.ok and sfx.class_name == "LocalSFX" and sfx.warnings == ["audiocraft not installed; local sfx uses procedural synthesis"]

    everything_local = Settings.from_env({"BOOKREADER_TTS_PROVIDER": "local", "BOOKREADER_MUSIC_PROVIDER": "local", "BOOKREADER_SFX_PROVIDER": "local",
                                          "BOOKREADER_LOCAL_SFX_FALLBACK": "0"})
    with pytest.raises(ProviderConfigError) as exc:
        base.validate_providers(everything_local)
    message = str(exc.value)
    assert "[tts=local]" in message and "[music=local]" in message and "[sfx=local]" in message
    assert message.count("bookreader[local]") == 2 and "audiocraft" in message


# --------------------------------------------------------------------------- building


def test_build_providers_on_all_mock_settings_yields_mock_singletons(mock_analysis: types.ModuleType, mock_settings: Settings):
    providers = base.build_providers(mock_settings)
    assert isinstance(providers, base.Providers)
    for capability in base.CAPABILITIES:
        provider = providers.get(capability)
        assert provider.family == "mock"
        assert provider is getattr(providers, capability)
    assert isinstance(providers.tts, base.VoiceSynthesizer)
    assert isinstance(providers.music, base.MusicGenerator)
    assert isinstance(providers.sfx, base.SfxGenerator)
    assert isinstance(providers.analysis, base.TextAnalyzer)
    described = providers.describe()
    assert set(described) == set(base.CAPABILITIES)
    assert all(entry["family"] == "mock" and entry["cache_version"] for entry in described.values())


BUILD_SCRIPT = r"""
import json, sys, types
MOCK_ANALYSIS = "bookreader.providers.mock.analysis"
try:
    __import__(MOCK_ANALYSIS)
except ModuleNotFoundError as exc:      # WP2 not landed yet: same stand-in as the pytest fixture
    if exc.name != MOCK_ANALYSIS:
        raise
    stub = types.ModuleType(MOCK_ANALYSIS)
    class HeuristicAnalyzer:
        family = "mock"; cache_version = "1"; model_id = "heuristic-1"
        @classmethod
        def check(cls, settings): return []
        @classmethod
        def from_settings(cls, settings, usage=None): return cls()
        def warmup(self): return None
        def analyze_chunk(self, chunk, bible): raise NotImplementedError
    stub.HeuristicAnalyzer = HeuristicAnalyzer
    sys.modules[MOCK_ANALYSIS] = stub
from bookreader.providers import base, build_providers, validate_providers, warmup_providers
from bookreader.settings import Settings
settings = Settings.from_env({})
validate_providers(settings)
providers = build_providers(settings)
warmup_providers(providers)
loaded = sorted(name for name, mod in sys.modules.items() if mod is not None and name.split(".")[0] in %s)
print(json.dumps({"loaded": loaded, "families": [providers.get(c).family for c in base.CAPABILITIES]}))
"""


def test_all_mock_build_never_imports_third_party_sdks():
    """Run in a fresh interpreter: this pytest process may already have imported the elevenlabs SDK."""
    script = BUILD_SCRIPT % repr(SDK_MODULES)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["loaded"] == []
    assert payload["families"] == ["mock", "mock", "mock", "mock"]


# --------------------------------------------------------------------------- describing and warmup


def test_describe_providers_never_raises(mock_analysis: types.ModuleType, monkeypatch: pytest.MonkeyPatch):
    for name in ("elevenlabs", "piper", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)
    settings = Settings.from_env({
        "BOOKREADER_ANALYSIS_PROVIDER": "anthropic",       # module not importable without the SDK / key
        "BOOKREADER_TTS_PROVIDER": "nope",                  # unknown family
        "BOOKREADER_MUSIC_PROVIDER": "local",               # transformers/torch nulled out
        "BOOKREADER_SFX_PROVIDER": "elevenlabs",            # SDK nulled out
    })
    entries = {e["capability"]: e for e in base.describe_providers(settings)}
    assert set(entries) == set(base.CAPABILITIES)
    assert all(set(e) >= {"capability", "family", "ok", "warnings", "error"} for e in entries.values())
    assert entries["tts"]["ok"] is False and "unknown provider family 'nope'" in entries["tts"]["error"]
    assert entries["music"]["ok"] is False and "bookreader[local]" in entries["music"]["error"]
    assert entries["sfx"]["ok"] is False and "bookreader[elevenlabs]" in entries["sfx"]["error"]
    assert isinstance(entries["analysis"]["ok"], bool) and (entries["analysis"]["ok"] or entries["analysis"]["error"])

    healthy = {e["capability"]: e for e in base.describe_providers(Settings.from_env({}))}
    assert all(e["ok"] and e["error"] is None and e["warnings"] == [] for e in healthy.values())
    assert healthy["sfx"]["class"] == "ProceduralSfx"


def test_warmup_providers_calls_warmup_on_each(monkeypatch: pytest.MonkeyPatch):
    warmed: list[str] = []

    def make(capability: str, with_warmup: bool = True) -> type:
        namespace: dict[str, Any] = {
            "family": "spy", "cache_version": "1",
            "check": classmethod(lambda cls, settings: []),
            "from_settings": classmethod(lambda cls, settings, usage=None: cls()),
        }
        if with_warmup:
            namespace["warmup"] = lambda self: warmed.append(capability)
        return type(f"Spy{capability.title()}", (), namespace)

    providers = base.Providers(analysis=make("analysis")(), tts=make("tts")(), music=make("music")(), sfx=make("sfx")())
    base.warmup_providers(providers)
    assert warmed == list(base.CAPABILITIES)

    warmed.clear()
    partial = base.Providers(analysis=make("analysis", with_warmup=False)(), tts=make("tts")(), music=make("music")(), sfx=make("sfx")())
    base.warmup_providers(partial)
    assert warmed == ["tts", "music", "sfx"]

    install_family(monkeypatch, "spy_family", "spy", {"analysis": make("analysis"), "tts": make("tts"), "music": make("music"), "sfx": make("sfx")})
    settings = Settings.from_env({f"BOOKREADER_{cap.upper()}_PROVIDER": "spy_family" for cap in base.CAPABILITIES})
    warmed.clear()
    base.warmup_providers(base.build_providers(settings))
    assert warmed == list(base.CAPABILITIES)
