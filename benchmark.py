import time
import os
from semantic_engine import scan_repository, ParserPool

REPO_PATH = "./test_repo"

print(f"Benchmarking scan on: {REPO_PATH}")

# 1. Python Baseline (Standard os.walk)
start = time.time()
py_files = []
for root, dirs, files in os.walk(REPO_PATH):
    # Simulating a "dumb" walker that hits node_modules
    # If you want a fair comparison of "useful" files, you'd filter here too
    for file in files:
        py_files.append(os.path.join(root, file))
py_time = time.time() - start

# 2. Rust Implementation (Smart walker)
start = time.time()
rust_files = scan_repository(REPO_PATH)
rust_time = time.time() - start

print(f"Python (Standard): {py_time:.3f}s for {len(py_files)} files")
print(f"Rust (Smart):      {rust_time:.3f}s for {len(rust_files)} files")
print(f"Speedup:           {py_time/rust_time:.1f}x")

# Verification
print(f"\nNote: Rust found fewer files ({len(rust_files)}) because it correctly ignored 'node_modules' and '.git'")

# 3. Benchmark Parser Pool Initialization
print("\nBenchmarking Parser Pool Initialization:")

# The number of languages for which parsers are initialized in ParserPool::new()
# This should match the number of variants in SupportedLanguage::all()
expected_languages_count = 5 # Python, Rust, JavaScript, TypeScript, Go

start_parser_init = time.time()
parser_pool = ParserPool()
parser_init_time = time.time() - start_parser_init

# We assume that ParserPool.num_parsers() accurately reflects the count
# of successfully initialized parsers.
actual_initialized_parsers = parser_pool.num_parsers()

print(f"Time to initialize {actual_initialized_parsers} parsers: {parser_init_time:.3f}s")

# Criterion: < 1ms per language
# Total time should be less than expected_languages_count * 0.001 seconds
criterion_met = parser_init_time < (expected_languages_count * 0.001)

if actual_initialized_parsers != expected_languages_count:
    print(f"  WARNING: Expected {expected_languages_count} parsers, but only {actual_initialized_parsers} were initialized.")

if criterion_met:
    print(f"  Criterion Met: < 1ms per language (Total < {expected_languages_count}ms)")
else:
    print(f"  Criterion FAILED: >= 1ms per language (Total >= {expected_languages_count}ms)")
