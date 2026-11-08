"""Monitor's cells for the check root without a window (monitor.series_view, the data the Monitor tab draws)."""
import sys
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind import monitor
from bearmind.paths import SeriesPaths
v = monitor.series_view(SeriesPaths(r"H:\Projects\_bearmind_test\phase5_short"))
for row in v.rows:
    print(row.sid, {s: c.text for s, c in row.cells.items()})
print("waiting counts:", getattr(v, "waiting", None) or [a for a in vars(v).items() if "manual" in str(a[1])][:3])
