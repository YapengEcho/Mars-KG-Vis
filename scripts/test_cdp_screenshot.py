"""Tests for the headless screenshot helper.

The 404 case is the point of the file: the browser screenshots an error page just as
happily as a real one, so ``capture`` has to refuse it before writing a file.
"""

from __future__ import annotations

import http.server
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import cdp_screenshot as cdp  # noqa: E402

GOOD = {
    "title": "Mars Atlas",
    "ready": "complete",
    "app": 3,
    "text": "探索火星 滑坡目录 图鉴 12 个案例",
}


class PageProblemTest(unittest.TestCase):
    def test_good_page_passes(self) -> None:
        self.assertIsNone(cdp.page_problem(GOOD, 200, "http://x/#/atlas"))

    def test_404_status_is_rejected(self) -> None:
        message = cdp.page_problem(GOOD, 404, "http://x/atlas")
        self.assertIsNotNone(message)
        self.assertIn("HTTP 404", message)

    def test_missing_app_root_is_rejected(self) -> None:
        state = dict(GOOD, app=-1)
        self.assertIn("#app", cdp.page_problem(state, 200, "http://x/") or "")

    def test_empty_app_root_is_rejected(self) -> None:
        state = dict(GOOD, app=0, text="")
        self.assertIsNotNone(cdp.page_problem(state, 200, "http://x/"))

    def test_error_page_text_is_rejected(self) -> None:
        state = dict(GOOD, text="404 Not Found")
        self.assertIsNotNone(cdp.page_problem(state, 200, "http://x/"))

    def test_unfinished_load_is_rejected(self) -> None:
        state = dict(GOOD, ready="loading")
        self.assertIsNotNone(cdp.page_problem(state, 200, "http://x/"))


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        bodies = {
            "/missing": (404, "<h1>404 Not Found</h1>"),
            "/blank": (200, '<!doctype html><html><body><div id="app"></div></body></html>'),
            "/ok": (
                200,
                '<!doctype html><html><body><div id="app"><p>Mars Atlas 图鉴案例页面</p></div></body></html>',
            ),
        }
        status, body = bodies.get(self.path, (404, "nope"))
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # keep the test output clean
        pass


class CaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cdp.find_browser()
        except SystemExit:
            raise unittest.SkipTest("没有 Chrome/Edge，跳过真实截图用例")
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def _capture(self, path: str, port: int) -> dict:
        out = os.path.join(self.tmp.name, f"shot{path.replace('/', '-')}.png")
        return cdp.capture(
            f"http://127.0.0.1:{self.port}{path}",
            out,
            width=430,
            height=932,
            dpr=1,
            wait=1.0,
            port=port,
        )

    def test_404_page_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._capture("/missing", 9401)
        self.assertIn("HTTP 404", str(ctx.exception))
        self.assertFalse(Path(self.tmp.name, "shot-missing.png").exists(), "404 页面不该留下截图")

    def test_page_without_content_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._capture("/blank", 9402)
        message = str(ctx.exception)
        self.assertTrue("#app" in message or "空的" in message, message)

    def test_real_page_is_written(self) -> None:
        info = self._capture("/ok", 9403)
        self.assertTrue(Path(info["file"]).exists())
        self.assertGreater(info["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
