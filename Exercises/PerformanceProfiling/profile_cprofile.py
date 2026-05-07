"""Section 1 driver — profile vae_mnist.py with cProfile, then visualize.

This script runs the clean VAE training script under cProfile and writes
the resulting stats to `vae.prof`. Open the `.prof` file with snakeviz
(`snakeviz vae.prof`) to inspect it interactively in your browser.

Run:
    python profile_cprofile.py
    snakeviz vae.prof

Equivalent one-liner using `python -m cProfile` directly (works the same):
    python -m cProfile -o vae.prof -s cumtime vae_mnist.py
    snakeviz vae.prof

What you should be able to answer after this section
-----------------------------------------------------
1. Which function takes the most cumulative time?
2. Which function takes the most "self" time (tottime)?
3. When are tottime and cumtime equal, and when do they differ?

Hints
-----
- cumtime is "time spent in this function and everything it called".
- tottime is "time spent in this function alone, excluding subcalls".
- For a leaf function (one that doesn't call other Python functions),
  the two are equal.
"""
import cProfile
import pstats

from vae_mnist import main

OUTPUT = "vae.prof"

# TODO: (optional) adjust epochs in vae_mnist.py before running.
# Default is 5; drop to 1 for a quick smoke profile.

print(f"Profiling vae_mnist.main() and writing stats to {OUTPUT} ...")
profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()
profiler.dump_stats(OUTPUT)

# Brief textual summary in the terminal so you don't have to open snakeviz
# just to see the top-line numbers.
print("\nTop 10 functions by cumulative time:")
pstats.Stats(OUTPUT).sort_stats("cumulative").print_stats(10)

print(f"\nWrote {OUTPUT}. To visualize:")
print(f"    snakeviz {OUTPUT}")
