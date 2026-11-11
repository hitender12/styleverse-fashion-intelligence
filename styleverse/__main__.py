"""STYLEVERSE CLI.

Usage:
    python -m styleverse            # run the end-to-end demo
    python -m styleverse evaluate   # run the retrieval evaluation
    python -m styleverse catalog    # (re)generate the synthetic catalog
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from .demo import main as demo_main
        demo_main()
        return 0
    if args[0] == "evaluate":
        from .evaluate import main as eval_main
        eval_main()
        return 0
    if args[0] == "catalog":
        from .catalog import generate_catalog
        products = generate_catalog()
        print(f"generated {len(products)} products in catalog/")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
