"""Benchmark parsing and core statistics on synthetic chats of increasing size.

Usage:  python scripts/benchmark.py [sizes...]
Prints a Markdown table of wall-clock time (time.perf_counter, measured without
tracing) and, in a separate pass, peak Python heap allocation (tracemalloc;
excludes some native allocations, so treat memory as a lower bound). Process
peak RSS is also reported. Generation time is excluded.
"""

from __future__ import annotations

import gc
import platform
import resource
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wa_analyzer import analytics as an  # noqa: E402
from wa_analyzer.parsing import parse_chat  # noqa: E402
from wa_analyzer.sample import generate_chat  # noqa: E402


def measure(fn):
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024 / 1024


def core_stats(df):
    f = an.Filters()
    an.overview(df)
    an.participant_summary(df)
    an.daily_counts(df)
    an.monthly_counts(df)
    an.weekday_hour_matrix(df)
    an.response_gaps(df, f)
    an.sessions(df, f)
    an.silence_gaps(df, f)
    an.word_frequencies(df)
    an.top_emoji(df)


def main(sizes: list[int]) -> None:
    print(f"Python {platform.python_version()} on {platform.machine()} ({platform.system()})\n")
    print(
        "| messages | export size (MB) | parse time (s) | parse peak heap (MB) | core stats time (s) | stats peak heap (MB) |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for n in sizes:
        text = generate_chat(n, seed=11, participants=6)
        result, t_parse, m_parse = measure(lambda text=text: parse_chat(text))
        _, t_stats, m_stats = measure(lambda df=result.df: core_stats(df))
        size_mb = len(text.encode("utf-8")) / 1024 / 1024
        print(f"| {n:,} | {size_mb:.1f} | {t_parse:.2f} | {m_parse:.0f} | {t_stats:.2f} | {m_stats:.0f} |")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\nProcess peak RSS during the run: {rss:.0f} MB (includes interpreter, pandas and all sizes)")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1_000, 10_000, 100_000, 300_000])
