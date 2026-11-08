"""While the step-2 tab holds NOF_H03_3D: the index page, and a second tab asking for NOF_H03_3D (one IOLoop per
connection). usage: python t7_probe.py URL"""
import asyncio, re, sys, time, urllib.request
from bokeh.client import pull_session
from bokeh.models import Div
from bokeh.util.token import get_session_id
from tornado.ioloop import IOLoop
URL = sys.argv[1]

def page(path, wait):
    t = time.perf_counter()
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        body = r.read().decode("utf-8", "replace"); code = r.status
    sid = get_session_id(re.search(r'"token"\s*:\s*"([^"]+)"', body).group(1))
    io = IOLoop(make_current=False); asyncio.set_event_loop(io.asyncio_loop)
    s = pull_session(session_id=sid, url=URL + "/", io_loop=io)
    try:
        time.sleep(wait)
        s.close()
        s = pull_session(session_id=sid, url=URL + "/", io_loop=io)  # the server's document after the wait
        doc = s.document
        texts = [d.text for r in doc.roots for d in r.references() if isinstance(d, Div)]
        print(f"[probe] GET {path} -> {code} in {time.perf_counter() - t:.2f}s; title {doc.title!r}; divs {texts!r}"
              .encode("ascii", "backslashreplace").decode(), flush=True)
    finally:
        s.close(); io.close(all_fds=True)

page("/", 0.5)
page("/?sid=NOF_H03_3D", 4.0)
