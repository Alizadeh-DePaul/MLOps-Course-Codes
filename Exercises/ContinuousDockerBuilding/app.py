"""Tiny entrypoint that proves the image was built and runs.

Imports numpy and rich so the requirements.txt layer is genuinely exercised
(not just installed and discarded).
"""

import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()


def main() -> None:
    rng = np.random.default_rng(seed=489)
    sample = rng.standard_normal(5).round(3)
    console.print(
        Panel.fit(
            f"[bold green]SE 489 — Continuous Docker Building[/]\n"
            f"[dim]Image built and running.[/]\n"
            f"Sample numpy array: {sample.tolist()}",
            title="hello from the container",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()
