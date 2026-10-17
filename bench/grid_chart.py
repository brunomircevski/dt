#!/usr/bin/env python3
"""Put finished chart images side by side in a grid, unchanged: row by row,
`--cols` per row, on white, each cell as large as the largest image.

  bench/.venv/bin/python bench/grid_chart.py a.png b.png c.png d.png --out grid.png
"""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--gap", type=int, default=20, help="pixels between the images")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    images = [Image.open(path).convert("RGB") for path in args.images]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = -(-len(images) // args.cols)
    grid = Image.new("RGB", (args.cols * width + (args.cols - 1) * args.gap,
                             rows * height + (rows - 1) * args.gap), "white")
    for index, image in enumerate(images):
        row, col = divmod(index, args.cols)
        grid.paste(image, (col * (width + args.gap), row * (height + args.gap)))
    grid.save(args.out, dpi=images[0].info.get("dpi", (150, 150)))
    print(args.out)


if __name__ == "__main__":
    main()
