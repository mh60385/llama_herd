#!/usr/bin/env python3
"""Smoke test: compare full text extraction vs snippets-only.

Usage: python test_snippets_vs_fulltext.py --query "your query" --num 5
"""

import time
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))

from src.config import get_search_config
from src.search import SearchManager
from src.source_reader import SourceReader


def test_approaches(query: str, num_results: int) -> None:
    search = SearchManager(get_search_config(), max_results=num_results)
    results = search.search(query)
    
    if not results:
        print("No results found")
        return
    
    print(f"\nTesting: {query}")
    print(f"Results to test: {num_results}\n")
    
    total_time = 0.0
    success = 0
    failed = 0
    snippet_chars = 0
    full_chars = 0
    
    for i, result in enumerate(results[:num_results]):
        reader = SourceReader(max_chars=4000)
        start = time.time()
        text, error = reader.read(result)
        elapsed = time.time() - start
        total_time += elapsed
        
        snippet_chars += len(result.snippet)
        full_chars += len(text)
        
        if error:
            failed += 1
            print(f"  [{i+1}] FAIL: {result.title[:45]} ({elapsed:.1f}s) - {str(error)[:50]}")
        else:
            success += 1
            print(f"  [{i+1}] OK:   {result.title[:45]} ({elapsed:.1f}s, {len(text)} chars)")
    
    avg_full = full_chars / num_results if num_results > 0 else 0
    avg_snippet = snippet_chars / num_results if num_results > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Success: {success}/{num_results}, Failed: {failed}/{num_results}")
    print(f"Avg chars: full={avg_full:.0f}, snippet={avg_snippet:.0f}")
    print(f"Chars ratio: {avg_full/avg_snippet:.1f}x more with full text")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test full text vs snippets")
    parser.add_argument("--query", default="Visual arts and gender in non-fiction history")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()
    test_approaches(args.query, args.num)
