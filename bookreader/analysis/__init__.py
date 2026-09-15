"""bookreader.analysis - the analyze stage's pure building blocks.

* :mod:`bookreader.analysis.chunker` packs whole paragraphs into analyzer-sized chunks.
* :mod:`bookreader.analysis.bible` threads character identity across chunks (the cast bible).
* :mod:`bookreader.analysis.validate` repairs or rejects an analyzer's output for one chunk.
* :mod:`bookreader.analysis.assemble` turns validated chunk analyses into a ChapterScript.

This package deliberately imports nothing at package level so that provider modules can import
its submodules without cycles.
"""
