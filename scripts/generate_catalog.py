#!/usr/bin/env python3
"""(Re)generate the synthetic catalog into ./catalog/."""

from styleverse.catalog import generate_catalog

if __name__ == "__main__":
    products = generate_catalog()
    print(f"generated {len(products)} products in catalog/")
