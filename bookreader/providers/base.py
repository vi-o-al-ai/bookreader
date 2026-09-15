"""bookreader.providers.base - provider Protocols and the family registry.

A *family* (mock, anthropic, elevenlabs, local, or any dotted third-party module path) is a
package whose ``__init__`` is a constants-only manifest::

    FAMILY = "elevenlabs"
    PROVIDERS = {"tts": "bookreader.providers.elevenlabs.tts:ElevenLabsTTS", ...}
    EXTRA = "elevenlabs"                    # pip extra that installs the SDK, or None
    REQUIRED_SECRETS = ("ELEVENLABS_API_KEY",)

Importing a manifest costs nothing; the provider module is imported only when its
capability is selected, and the third-party SDK is imported only inside ``check``/``from_settings``
(always ``try: import x`` - never ``importlib.util.find_spec``, so tests can stub or null out
``sys.modules`` entries).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from bookreader.types import (
    AudioClip,
    CastBible,
    Chunk,
    ChunkAnalysis,
    MusicRequest,
    ProviderConfigError,
    SfxRequest,
    TTSRequest,
    VoiceInfo,
)

if TYPE_CHECKING:  # avoid an import cycle; Settings lives in bookreader.settings
    from bookreader.settings import Settings

CAPABILITIES: tuple[str, ...] = ("analysis", "tts", "music", "sfx")
BUILTIN_FAMILIES: tuple[str, ...] = ("mock", "anthropic", "elevenlabs", "local")


# --------------------------------------------------------------------------- usage sink
class UsageSink(Protocol):
    """Implemented by bookreader.usage.UsageLedger; providers call it after every external call."""

    def record(
        self,
        capability: str,
        family: str,
        unit_type: str,
        units: float,
        *,
        cache_hit: bool = False,
        duration_ms: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> None: ...


class NullUsage:
    """Default sink: drops everything. Used by unit tests and the mock family."""

    def record(self, capability: str, family: str, unit_type: str, units: float, **_: Any) -> None:
        return None


# --------------------------------------------------------------------------- provider protocols
class Provider(Protocol):
    """Attributes and lifecycle every capability shares."""

    family: ClassVar[str]          # "mock" | "anthropic" | "elevenlabs" | "local"
    cache_version: str             # model id / DSP recipe version; part of every cache key

    @classmethod
    def check(cls, settings: "Settings") -> list[str]:
        """Startup validation. Raise ProviderConfigError with the exact remedy
        (pip extra, env var, path) or return a list of warnings. Must not download models."""
        ...

    @classmethod
    def from_settings(cls, settings: "Settings", usage: UsageSink | None = None) -> Any:
        """Construct the singleton for this process. Third-party clients are injected through
        __init__ so tests can pass fakes; from_settings is the only place that builds real ones."""
        ...

    def warmup(self) -> None:
        """Called at startup when settings.warmup is true: load local weights, fetch voice
        catalogs, so missing model artifacts fail before any job is accepted."""
        ...


@runtime_checkable
class TextAnalyzer(Protocol):
    family: ClassVar[str]
    cache_version: str
    model_id: str

    def analyze_chunk(self, chunk: Chunk, bible: CastBible) -> ChunkAnalysis:
        """Label every quote span in *chunk* and emit cues/character updates.
        Must be deterministic for the mock family. May raise ProviderTransientError /
        ProviderPermanentError; must never raise on refusal (fall back to the heuristic)."""
        ...


@runtime_checkable
class VoiceSynthesizer(Protocol):
    family: ClassVar[str]
    cache_version: str
    max_chars: int                 # longest text accepted per synthesize call; the render stage splits above it

    def list_voices(self) -> list[VoiceInfo]:
        """Full catalog (bounded by settings.max_voices). Cached per process by the caller."""
        ...

    def synthesize(self, req: TTSRequest) -> AudioClip:
        """Return audio at any sample rate; the pipeline resamples to SAMPLE_RATE."""
        ...


@runtime_checkable
class MusicGenerator(Protocol):
    family: ClassVar[str]
    cache_version: str
    min_duration_ms: int
    max_duration_ms: int           # the timeline clamps requests; longer regions are looped at mix time

    def compose(self, req: MusicRequest) -> AudioClip: ...


@runtime_checkable
class SfxGenerator(Protocol):
    family: ClassVar[str]
    cache_version: str
    max_duration_ms: int

    def generate(self, req: SfxRequest) -> AudioClip: ...


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class FamilyManifest:
    family: str
    providers: dict[str, str]                 # capability -> "module.path:ClassName"
    extra: str | None = None
    required_secrets: tuple[str, ...] = ()


@dataclass
class ProviderReport:
    capability: str
    family: str
    class_name: str = ""
    ok: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Providers:
    analysis: TextAnalyzer
    tts: VoiceSynthesizer
    music: MusicGenerator
    sfx: SfxGenerator

    def get(self, capability: str) -> Any:
        return getattr(self, capability)

    def describe(self) -> dict[str, dict[str, str]]:
        return {
            cap: {"family": getattr(self.get(cap), "family", "?"), "cache_version": str(getattr(self.get(cap), "cache_version", ""))}
            for cap in CAPABILITIES
        }


def family_module_names(family: str) -> list[str]:
    """Candidate module names: the built-in package first, then the name as given (third-party)."""
    if "." in family:
        return [family]
    return [f"bookreader.providers.{family}", family]


def load_manifest(family: str) -> FamilyManifest:
    """Import the family package (constants only) and read its manifest."""
    mod = None
    tried: list[str] = []
    for module_name in family_module_names(family):
        tried.append(module_name)
        try:
            mod = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as exc:
            if exc.name != module_name:        # the family exists but one of its own imports failed
                raise ProviderConfigError(f"provider family '{family}' failed to import: {exc}") from exc
    if mod is None:
        raise ProviderConfigError(
            f"unknown provider family '{family}' (tried {', '.join(tried)}); "
            f"built-in families: {', '.join(BUILTIN_FAMILIES)}"
        )
    providers = getattr(mod, "PROVIDERS", None)
    if not isinstance(providers, dict):
        raise ProviderConfigError(f"{module_name} is not a provider family: it has no PROVIDERS dict")
    return FamilyManifest(
        family=str(getattr(mod, "FAMILY", family)),
        providers=dict(providers),
        extra=getattr(mod, "EXTRA", None),
        required_secrets=tuple(getattr(mod, "REQUIRED_SECRETS", ())),
    )


def resolve(capability: str, family: str) -> type:
    """Return the provider class for (capability, family), importing its module lazily."""
    if capability not in CAPABILITIES:
        raise ProviderConfigError(f"unknown capability '{capability}'; expected one of {', '.join(CAPABILITIES)}")
    manifest = load_manifest(family)
    if capability not in manifest.providers:
        raise ProviderConfigError(
            f"provider family '{family}' does not provide '{capability}'; "
            f"it provides: {', '.join(sorted(manifest.providers)) or 'nothing'}"
        )
    module_path, _, class_name = manifest.providers[capability].partition(":")
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ProviderConfigError(f"cannot import {module_path} for {capability}={family}: {exc}") from exc
    try:
        return getattr(mod, class_name)
    except AttributeError as exc:
        raise ProviderConfigError(f"{module_path} has no class {class_name}") from exc


def selected_family(settings: "Settings", capability: str) -> str:
    return str(getattr(settings, f"{capability}_provider"))


def validate_providers(settings: "Settings") -> list[ProviderReport]:
    """Run at startup (API lifespan, CLI). Aggregates every problem into one ProviderConfigError
    so the operator sees all fixes at once; returns reports (with warnings) when everything is ok."""
    reports: list[ProviderReport] = []
    problems: list[str] = []
    for capability in CAPABILITIES:
        family = selected_family(settings, capability)
        report = ProviderReport(capability=capability, family=family)
        try:
            manifest = load_manifest(family)
            missing = [k for k in manifest.required_secrets if not settings.secrets.get(k)]
            if missing:
                hint = f" (pip install 'bookreader[{manifest.extra}]' if not installed)" if manifest.extra else ""
                raise ProviderConfigError(
                    f"{capability} provider '{family}' needs environment variable(s) {', '.join(missing)}{hint}"
                )
            cls = resolve(capability, family)
            report.class_name = cls.__name__
            report.warnings = list(cls.check(settings))
            report.ok = True
        except ProviderConfigError as exc:
            report.error = str(exc)
            problems.append(f"[{capability}={family}] {exc}")
        reports.append(report)
    if problems:
        raise ProviderConfigError("provider configuration invalid:\n  " + "\n  ".join(problems))
    return reports


def build_provider(capability: str, settings: "Settings", usage: UsageSink | None = None) -> Any:
    cls = resolve(capability, selected_family(settings, capability))
    return cls.from_settings(settings, usage or NullUsage())


def build_providers(settings: "Settings", usage: UsageSink | None = None) -> Providers:
    """Instantiate one provider per capability. Call validate_providers() first at startup."""
    sink = usage or NullUsage()
    return Providers(
        analysis=build_provider("analysis", settings, sink),
        tts=build_provider("tts", settings, sink),
        music=build_provider("music", settings, sink),
        sfx=build_provider("sfx", settings, sink),
    )


def warmup_providers(providers: Providers) -> None:
    for capability in CAPABILITIES:
        provider = providers.get(capability)
        warm = getattr(provider, "warmup", None)
        if callable(warm):
            warm()


def describe_providers(settings: "Settings") -> list[dict[str, Any]]:
    """Non-raising variant for /api/health and `bookreader providers`."""
    out: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        family = selected_family(settings, capability)
        entry: dict[str, Any] = {"capability": capability, "family": family, "ok": True, "warnings": [], "error": None}
        try:
            manifest = load_manifest(family)
            missing = [k for k in manifest.required_secrets if not settings.secrets.get(k)]
            if missing:
                hint = f" (pip install 'bookreader[{manifest.extra}]' if not installed)" if manifest.extra else ""
                raise ProviderConfigError(
                    f"{capability} provider '{family}' needs environment variable(s) {', '.join(missing)}{hint}"
                )
            cls = resolve(capability, family)
            entry["class"] = cls.__name__
            entry["warnings"] = list(cls.check(settings))
        except ProviderConfigError as exc:
            entry.update(ok=False, error=str(exc))
        out.append(entry)
    return out
