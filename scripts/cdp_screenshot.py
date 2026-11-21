"""Headless Chrome screenshot helper for the mobile demo pages.

Windows PowerShell has no working Playwright/Puppeteer install here, so the
figures are captured over the Chrome DevTools Protocol with the standard
library only. The page is laid out at a phone viewport, then the viewport is
grown to the document height before the shot so the fixed bottom navigation
lands at the bottom of the frame instead of across the middle of the figure.

Usage:
    python scripts/cdp_screenshot.py --url http://127.0.0.1:5174/#/atlas \\
        --out output/week5/shot-atlas.png --width 430 --height 932 --dpr 2

Exit codes: 0 written, 2 the browser did not come up.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


class WebSocket:
    """Tiny RFC 6455 client: enough for request/response CDP traffic."""

    def __init__(self, url: str) -> None:
        if not url.startswith("ws://"):
            raise ValueError(f"unexpected websocket url: {url}")
        hostport, path = url[len("ws://") :].split("/", 1)
        path = "/" + path
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port or 80)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET {path} HTTP/1.1\r\nHost: {hostport}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("websocket handshake failed")
            head += chunk
        self.buf = head.split(b"\r\n\r\n", 1)[1]

    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", length)
        header += os.urandom(4)
        self.sock.sendall(bytes(header) + _mask(data, header[-4:]))

    def _read_exact(self, size: int) -> bytes:
        while len(self.buf) < size:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:size], self.buf[size:]
        return out

    def recv(self) -> dict:
        while True:
            first = self._read_exact(2)
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if first[1] & 0x80 else b""
            payload = self._read_exact(length)
            if mask:
                payload = _mask(payload, mask)
            if opcode == 0x8:
                raise RuntimeError("websocket closed by peer")
            if opcode in (0x1, 0x0):
                return json.loads(payload.decode())


def _mask(data: bytes, mask: bytes) -> bytes:
    return bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))


def find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("找不到 Chrome/Edge，截图需要其中一个浏览器")


def connect(port: int, timeout: float = 30.0) -> WebSocket:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as fp:
                targets = json.load(fp)
            pages = [item for item in targets if item.get("type") == "page"]
            if pages:
                return WebSocket(pages[0]["webSocketDebuggerUrl"])
        except (urllib.error.URLError, OSError, KeyError) as exc:  # browser still starting
            last = exc
        time.sleep(0.4)
    raise SystemExit(f"浏览器调试端口没有启动：{last}")


def capture(url: str, out: str, *, width: int, height: int, dpr: float, wait: float, port: int) -> dict:
    browser = find_browser()
    profile = os.path.join(os.environ.get("TEMP", "."), f"dsh-cdp-{port}")
    proc = subprocess.Popen(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws = connect(port)
        counter = [0]

        def call(method: str, params: dict | None = None) -> dict:
            counter[0] += 1
            ws.send({"id": counter[0], "method": method, "params": params or {}})
            while True:
                message = ws.recv()
                if message.get("id") == counter[0]:
                    if "error" in message:
                        raise RuntimeError(f"{method}: {message['error']}")
                    return message

        call("Page.enable")
        call(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": dpr, "mobile": True},
        )
        call("Page.navigate", {"url": url})
        time.sleep(wait)
        health = call(
            "Runtime.evaluate",
            {
                "expression": (
                    "JSON.stringify({title: document.title,"
                    " app: document.querySelector('#app') ? document.querySelector('#app').childElementCount : -1,"
                    " text: (document.body ? document.body.innerText : '').slice(0, 120)})"
                ),
                "returnByValue": True,
            },
        )
        state = json.loads(health["result"]["result"]["value"])
        text = state.get("text", "")
        if "无法访问此网站" in text or "ERR_CONNECTION" in text or state.get("app") == 0:
            raise RuntimeError(f"页面没有渲染出来（{url}）：确认 dev server 在这个地址上，且首页能在浏览器打开")
        measured = call(
            "Runtime.evaluate",
            {
                "expression": "Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)",
                "returnByValue": True,
            },
        )
        full_height = int(measured["result"]["result"]["value"])
        call(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": full_height, "deviceScaleFactor": dpr, "mobile": True},
        )
        time.sleep(0.6)
        shot = call("Page.captureScreenshot", {"format": "png"})
        data = base64.b64decode(shot["result"]["data"])
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "wb") as fp:
            fp.write(data)
        return {"url": url, "file": out, "size": [width, full_height], "bytes": len(data)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="用无头浏览器截取移动端页面")
    parser.add_argument("--url", required=True, help="页面地址，例如 http://127.0.0.1:5174/#/atlas")
    parser.add_argument("--out", required=True, help="输出 PNG 路径")
    parser.add_argument("--width", type=int, default=430, help="手机逻辑宽度")
    parser.add_argument("--height", type=int, default=932, help="初始视口高度")
    parser.add_argument("--dpr", type=float, default=2.0, help="设备像素比")
    parser.add_argument("--wait", type=float, default=5.0, help="导航后等待渲染的秒数")
    parser.add_argument("--port", type=int, default=9333, help="调试端口")
    args = parser.parse_args()

    try:
        info = capture(
            args.url,
            args.out,
            width=args.width,
            height=args.height,
            dpr=args.dpr,
            wait=args.wait,
            port=args.port,
        )
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(f"截图完成：{info['file']}（{info['size'][0]}×{info['size'][1]}，{info['bytes']:,} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
