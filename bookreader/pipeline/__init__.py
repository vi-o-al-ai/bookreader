"""bookreader.pipeline - the five-stage job pipeline (ingest, analyze, cast, render, finalize).

Import ``run_job`` and ``create_job`` from :mod:`bookreader.pipeline.run`; the stage functions live
in :mod:`bookreader.pipeline.stages` and share a :class:`bookreader.pipeline.context.JobContext`.
This package imports nothing at package level so that ``bookreader.pipeline.cache`` can be used
without pulling in the providers.
"""
