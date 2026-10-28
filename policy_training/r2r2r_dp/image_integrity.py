"""Versioned image-integrity manifest for the camera frames TinyDP reads (Codex final audit F4).

Contract:
  * build (once, read-only on the source): every frame of every manifest episode is fully decoded with
    PIL from the same bytes that are hashed; it must be an RGB JPEG of the declared resolution, and the
    frame set must be exactly 0000..T-1.jpg. Per camera, the identity is
        sha256( "<name>\\0<sha256(frame bytes)>\\n" for each frame in name order ).
  * verify (every launch): the camera digests of every episode in every split root are recomputed from
    the current frame bytes and must equal the manifest. Byte equality implies the decode/resolution
    certificate still applies, so frames are not re-decoded at launch.
The integrity file is bound to one episode manifest by sha256 and is pinned by sha256 in the protocol.
Production images and the episode manifest are never written.
"""
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple

IMAGE_INTEGRITY_VERSION = "r2r2r-dp-images-v1"
CAMERAS = ("camera_0", "camera_1")


class ImageIntegrityError(ValueError):
    pass


def frame_names(T: int) -> List[str]:
    return [f"{t:04d}.jpg" for t in range(T)]


def _digest(lines: Sequence[Tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for name, sha in lines:
        h.update(f"{name}\0{sha}\n".encode())
    return h.hexdigest()


def _listing(rgb_dir: str, T: int) -> List[str]:
    names = sorted(os.listdir(rgb_dir)) if os.path.isdir(rgb_dir) else []
    if names != frame_names(T):
        raise ImageIntegrityError(f"{rgb_dir}: does not hold exactly frames 0000..{T - 1:04d}.jpg")
    return names


def camera_digest(rgb_dir: str, T: int) -> str:
    """Identity of one camera's frames from their current bytes (no decoding)."""
    lines = []
    for name in _listing(rgb_dir, T):
        with open(os.path.join(rgb_dir, name), "rb") as f:
            lines.append((name, hashlib.sha256(f.read()).hexdigest()))
    return _digest(lines)


def certify_camera(rgb_dir: str, T: int, size: Tuple[int, int]) -> str:
    """Fully decode every frame (from the hashed bytes) and check format/mode/size; return the digest."""
    from PIL import Image, ImageFile
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ImageIntegrityError("PIL ImageFile.LOAD_TRUNCATED_IMAGES must be False")
    lines = []
    for name in _listing(rgb_dir, T):
        with open(os.path.join(rgb_dir, name), "rb") as f:
            data = f.read()
        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt, mode, got = im.format, im.mode, im.size
                im.load()
        except Exception as e:
            raise ImageIntegrityError(f"{rgb_dir}/{name}: not decodable: {e}") from e
        if fmt != "JPEG" or mode != "RGB" or tuple(got) != tuple(size):
            raise ImageIntegrityError(f"{rgb_dir}/{name}: {fmt} {mode} {got}, expected JPEG RGB {tuple(size)}")
        lines.append((name, hashlib.sha256(data).hexdigest()))
    return _digest(lines)


def _certify_episode(args):
    uid, ep_dir, T, size = args
    return uid, {cam: certify_camera(os.path.join(ep_dir, cam, "rgb"), T, size) for cam in CAMERAS}


def build_image_integrity(manifest: Dict, manifest_sha256: str, size: Tuple[int, int], workers: int = 8) -> Dict:
    """Certify every manifest episode's frames under manifest['source_root'] (read-only)."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(e["uid"], os.path.join(manifest["source_root"], e["rel_path"]), e["num_steps"], tuple(size))
            for e in manifest["episodes"]]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_certify_episode, jobs, chunksize=4))
    else:
        results = [_certify_episode(j) for j in jobs]
    with open(__file__, "rb") as f:
        code = hashlib.sha256(f.read()).hexdigest()
    return {"version": IMAGE_INTEGRITY_VERSION, "manifest_sha256": manifest_sha256,
            "image_format": "JPEG", "image_mode": "RGB", "image_size": list(size), "cameras": list(CAMERAS),
            "digest": "sha256 over '<frame name>\\0<sha256(frame bytes)>\\n' in name order, per camera",
            "code_sha256": code, "episodes": dict(sorted(results))}


def _reject_constant(c):
    raise ImageIntegrityError(f"non-standard JSON constant {c}")


def load_image_integrity(path: str, manifest: Dict, manifest_sha256: str) -> Dict:
    with open(path) as f:
        ii = json.load(f, parse_constant=_reject_constant)
    want_keys = {"version", "manifest_sha256", "image_format", "image_mode", "image_size", "cameras", "digest",
                 "code_sha256", "episodes"}
    if not isinstance(ii, dict) or set(ii) != want_keys:
        raise ImageIntegrityError(f"{path}: keys must be {sorted(want_keys)}")
    if ii["version"] != IMAGE_INTEGRITY_VERSION:
        raise ImageIntegrityError(f"{path}: version {ii['version']!r} != {IMAGE_INTEGRITY_VERSION}")
    if ii["manifest_sha256"] != manifest_sha256:
        raise ImageIntegrityError(f"{path}: certifies manifest {ii['manifest_sha256']}, not {manifest_sha256}")
    if ii["cameras"] != list(CAMERAS) or ii["image_format"] != "JPEG" or ii["image_mode"] != "RGB":
        raise ImageIntegrityError(f"{path}: unsupported cameras/format/mode")
    uids = {e["uid"] for e in manifest["episodes"]}
    eps = ii["episodes"]
    if not isinstance(eps, dict) or set(eps) != uids:
        raise ImageIntegrityError(f"{path}: episode set differs from the manifest")
    for uid, cams in eps.items():
        if not isinstance(cams, dict) or set(cams) != set(CAMERAS):
            raise ImageIntegrityError(f"{path}: {uid}: camera digests incomplete")
    return ii


def verify_episodes(episodes: Sequence[Tuple[str, str, int]], ii: Dict, workers: int = 8) -> List[str]:
    """episodes: (uid, view episode dir, T). Returns problems (empty if every camera digest matches)."""
    def one(item):
        uid, ep, T = item
        out = []
        for cam in CAMERAS:
            try:
                got = camera_digest(os.path.join(ep, cam, "rgb"), T)
            except (ImageIntegrityError, OSError) as e:
                out.append(f"{ep}: {cam}: {e}")
                continue
            if got != ii["episodes"][uid][cam]:
                out.append(f"{ep}: {cam} frame bytes differ from the image-integrity manifest")
        return out
    with ThreadPoolExecutor(max(1, workers)) as ex:  # hashing and file reads release the GIL
        return [p for ps in ex.map(one, episodes) for p in ps]


def dump_image_integrity(ii: Dict, path: str) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    with open(path, "w") as f:
        json.dump(ii, f, indent=1, allow_nan=False)
        f.write("\n")
