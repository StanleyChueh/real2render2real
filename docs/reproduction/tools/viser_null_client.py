"""Minimal headless viser client.

scripts/run.py blocks until a viser client has connected (the server registers a client after its
first ViewerCameraMessage). A browser does this but, without a GPU, software-renders the viser scene
(~10 CPU cores in our measurements), competing with the simulator. This client sends one camera
message and then reads and discards everything the server streams. It does not influence the saved
data: with calibrated camera poses loaded, the client camera is never used for rendering.

usage: python docs/reproduction/tools/viser_null_client.py PORT
"""
import asyncio
import sys
import time

import msgspec.msgpack
import viser
import websockets


async def main(port: int) -> None:
    camera = {
        "type": "ViewerCameraMessage",
        "wxyz": (1.0, 0.0, 0.0, 0.0), "position": (1.5, 0.0, 1.0), "fov": 1.0, "near": 0.01, "far": 100.0,
        "image_height": 720, "image_width": 1280, "look_at": (0.0, 0.0, 0.0), "up_direction": (0.0, 0.0, 1.0),
    }
    while True:
        try:
            async with websockets.connect(
                f"ws://localhost:{port}", subprotocols=[f"viser-v{viser.__version__}"], max_size=None
            ) as ws:
                await ws.send(msgspec.msgpack.encode(camera))
                print(f"connected {time.time():.3f}", flush=True)
                n = 0
                async for _ in ws:
                    n += 1
        except (OSError, websockets.exceptions.WebSocketException) as e:
            print(f"reconnect: {e!r}", flush=True)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1])))
