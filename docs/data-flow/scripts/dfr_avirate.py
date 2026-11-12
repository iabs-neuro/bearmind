"""Frame rate written in the AVI header of the first chunk of each LT1 session (header only), codec and size."""
import json, sys
import av
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind.paths import SeriesPaths
from bearmind.registry import read_all
paths = SeriesPaths(r"H:\Projects\LT1\CalciumData")
recs, _ = read_all(paths)
out = {}
for r in recs:
    p = r.parts[0]
    with av.open(str(paths.root / p.folder / p.chunks[0])) as c:
        s = c.streams.video[0]
        out[r.sid] = {"avg": float(s.average_rate) if s.average_rate else None, "guessed": float(s.guessed_rate) if s.guessed_rate else None,
                      "base": str(s.base_rate), "codec": s.codec_context.name, "pix": s.codec_context.pix_fmt}
json.dump(out, open("dfr_avirate.json", "w"), indent=0)
import collections
print(collections.Counter((v["avg"], v["codec"], v["pix"]) for v in out.values()))
print(collections.Counter((round(r.fps, 2), out[r.sid]["avg"]) for r in recs))
