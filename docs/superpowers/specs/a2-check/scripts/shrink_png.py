"""Copy a PNG under a size limit: a 256-colour palette first (no resize), then downscale steps if still too big.

usage: shrink_png.py SRC DST [LIMIT_KB]
"""
import os
import sys

from PIL import Image


def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) * 1024 if len(sys.argv) > 3 else 490 * 1024
    im = Image.open(src).convert("RGB")
    for scale in (1.0, 0.85, 0.7, 0.6, 0.5):
        cur = im if scale == 1.0 else im.resize((round(im.width * scale), round(im.height * scale)),
                                                Image.Resampling.LANCZOS)
        q = cur.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q.save(dst, optimize=True)
        size = os.path.getsize(dst)
        if size <= limit:
            print(f"[OK] {dst}: {cur.width}x{cur.height} (scale {scale}), {size // 1024} KB (source "
                  f"{im.width}x{im.height}, {os.path.getsize(src) // 1024} KB)")
            return
    print(f"[FAILED] {dst} still {size // 1024} KB")


if __name__ == "__main__":
    main()
