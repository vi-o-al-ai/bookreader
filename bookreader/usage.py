"""bookreader.usage - the per-job usage ledger (a ``UsageSink``) and cost estimates.

Providers call ``record`` after every external call; the render stage records cache hits with
``cache_hit=True`` and zero cost so the cache-hit ratio is visible. Rows are priced from
``settings.prices`` (``"<family>:<unit_type>"`` -> USD per unit) and stored in ``usage_events``.

Providers are process singletons while ledgers are per job, so :class:`UsageRouter` is the sink
handed to ``build_providers`` at startup: it forwards to whichever ledger is bound in the
current context (``run_job`` binds its ledger; the render stage propagates the context to its
worker threads).
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from bookreader.jobs.paths import write_json
from bookreader.types import Estimate, UsageEvent, UsageSummary

if TYPE_CHECKING:
    from bookreader.jobs.db import JobStore

log = logging.getLogger(__name__)

# Rough pre-flight assumptions (see UsageLedger.estimate_cost).
ANALYSIS_OUTPUT_TOKENS_PER_CHUNK = 1500      # labels, cues and character updates for one chunk
ANALYSIS_OUTPUT_TOKENS_PER_QUOTE = 25        # one label object per quote span
MUSIC_SECONDS_PER_CHAPTER = 120.0            # a couple of beds per chapter, capped at 120 s each
SFX_SECONDS_PER_PARAGRAPH = 1.0              # sparse impacts plus a few ambient beds


class UsageLedger:
    """Records priced usage rows for one job into the store. Implements ``UsageSink``."""

    def __init__(self, store: "JobStore", job_id: str, prices: Mapping[str, float]) -> None:
        self.store = store
        self.job_id = job_id
        self.prices = dict(prices)

    def price(self, family: str, unit_type: str) -> float:
        return float(self.prices.get(f"{family}:{unit_type}", 0.0))

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
    ) -> None:
        """Store one usage row; cache hits cost nothing."""
        cost = 0.0 if cache_hit else float(units) * self.price(family, unit_type)
        event = UsageEvent(
            capability=capability,
            family=family,
            unit_type=unit_type,
            units=float(units),
            cache_hit=cache_hit,
            cost_usd=cost,
            duration_ms=int(duration_ms),
            meta=dict(meta or {}),
        )
        self.store.add_usage(self.job_id, event)

    def estimate_cost(self, estimate: Estimate, families: Mapping[str, str]) -> dict[str, float]:
        """Pre-flight cost per capability (USD) from the ingest counts and the selected families.

        Analysis is priced on the estimated input tokens plus a nominal output volume per chunk
        and per quote span; TTS on the characters to synthesize; music and sfx on a nominal
        number of audio seconds per chapter / paragraph. Families without prices cost 0.
        """
        analysis = families.get("analysis", "")
        output_tokens = estimate.chunks * ANALYSIS_OUTPUT_TOKENS_PER_CHUNK + estimate.quote_spans * ANALYSIS_OUTPUT_TOKENS_PER_QUOTE
        costs = {
            "analysis": estimate.analysis_input_tokens_est * self.price(analysis, "input_tokens")
            + output_tokens * self.price(analysis, "output_tokens"),
            "tts": estimate.tts_chars * self.price(families.get("tts", ""), "characters"),
            "music": estimate.chapters * MUSIC_SECONDS_PER_CHAPTER * self.price(families.get("music", ""), "audio_seconds"),
            "sfx": estimate.paragraphs * SFX_SECONDS_PER_PARAGRAPH * self.price(families.get("sfx", ""), "audio_seconds"),
        }
        return {capability: round(value, 6) for capability, value in costs.items()}

    def summary(self) -> UsageSummary:
        return self.store.usage_summary(self.job_id)

    def write_json(self, path: Path) -> Path:
        """Write the current summary as ``usage.json``."""
        return write_json(path, self.summary())


_active_ledger: contextvars.ContextVar[UsageLedger | None] = contextvars.ContextVar("bookreader_active_ledger", default=None)


class UsageRouter:
    """Process-wide sink that forwards to the ledger bound in the current context (else drops)."""

    def record(self, capability: str, family: str, unit_type: str, units: float, **kwargs: Any) -> None:
        ledger = _active_ledger.get()
        if ledger is None:
            log.debug("usage row dropped (no active ledger): %s/%s %s=%s", capability, family, unit_type, units)
            return
        ledger.record(capability, family, unit_type, units, **kwargs)

    @staticmethod
    def active() -> UsageLedger | None:
        return _active_ledger.get()

    @staticmethod
    @contextmanager
    def bind(ledger: UsageLedger) -> Iterator[None]:
        """Route every ``record`` call made in this context (and contexts copied from it) to *ledger*."""
        token = _active_ledger.set(ledger)
        try:
            yield
        finally:
            _active_ledger.reset(token)
