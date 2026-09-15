"""Registry behaviour that does not depend on any concrete provider module."""
import sys
import types

import pytest

from bookreader.providers import base
from bookreader.settings import Settings
from bookreader.types import ProviderConfigError


def test_manifests_load_for_every_builtin_family():
    for family in base.BUILTIN_FAMILIES:
        m = base.load_manifest(family)
        assert m.family == family
        assert set(m.providers) <= set(base.CAPABILITIES)
    assert base.load_manifest("mock").required_secrets == ()
    assert base.load_manifest("anthropic").required_secrets == ("ANTHROPIC_API_KEY",)
    assert base.load_manifest("elevenlabs").extra == "elevenlabs"
    assert "analysis" not in base.load_manifest("local").providers


def test_unknown_family_and_missing_capability_errors():
    with pytest.raises(ProviderConfigError) as exc:
        base.load_manifest("nope")
    assert "mock" in str(exc.value) and "elevenlabs" in str(exc.value)
    with pytest.raises(ProviderConfigError) as exc2:
        base.resolve("analysis", "elevenlabs")
    assert "does not provide 'analysis'" in str(exc2.value)
    with pytest.raises(ProviderConfigError):
        base.resolve("video", "mock")


def test_third_party_dotted_family_via_sys_modules(monkeypatch):
    pkg = types.ModuleType("acme_voices")
    pkg.FAMILY = "acme"
    pkg.PROVIDERS = {"tts": "acme_voices.tts:AcmeTTS"}
    pkg.EXTRA = "acme"
    mod = types.ModuleType("acme_voices.tts")

    class AcmeTTS:
        family = "acme"

        @classmethod
        def check(cls, settings):
            return ["beta"]

    mod.AcmeTTS = AcmeTTS
    monkeypatch.setitem(sys.modules, "acme_voices", pkg)
    monkeypatch.setitem(sys.modules, "acme_voices.tts", mod)
    assert base.resolve("tts", "acme_voices") is AcmeTTS
    s = Settings.from_env({"BOOKREADER_TTS_PROVIDER": "acme_voices"})
    entries = {e["capability"]: e for e in base.describe_providers(s)}
    assert entries["tts"]["ok"] is True and entries["tts"]["warnings"] == ["beta"]


def test_validate_aggregates_missing_secrets_into_one_error():
    s = Settings.from_env({"BOOKREADER_ANALYSIS_PROVIDER": "anthropic", "BOOKREADER_TTS_PROVIDER": "elevenlabs"})
    with pytest.raises(ProviderConfigError) as exc:
        base.validate_providers(s)
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg and "ELEVENLABS_API_KEY" in msg
    assert "bookreader[anthropic]" in msg and "bookreader[elevenlabs]" in msg


def test_describe_reports_missing_secret_as_not_ok():
    s = Settings.from_env({"BOOKREADER_TTS_PROVIDER": "elevenlabs"})
    entries = {e["capability"]: e for e in base.describe_providers(s)}
    assert entries["tts"]["ok"] is False
    assert "ELEVENLABS_API_KEY" in entries["tts"]["error"]
    assert entries["analysis"]["ok"] is True
