"""Read / selectively extract members of a remote ZIP via HTTP Range requests.

usage:
  python -I remotezip.py list URL > listing.tsv
  python -I remotezip.py extract URL OUT_DIR PREFIX [PREFIX ...]
"""
import io
import os
import sys
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.block = block
        self.cache_start = -1
        self.cache = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as e:  # retry transient errors
                print(f"retry {attempt}: {e}", file=sys.stderr)
        raise RuntimeError("range fetch failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cs, ce = self.cache_start, self.cache_start + len(self.cache)
        if cs <= self.pos and self.pos + n <= ce:
            out = self.cache[self.pos - cs:self.pos - cs + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n)
        else:
            end = min(self.size, self.pos + max(n, self.block))
            self.cache = self._fetch(self.pos, end)
            self.cache_start = self.pos
            out = self.cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    cmd, url = sys.argv[1], sys.argv[2]
    f = HttpRangeFile(url)
    zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    if cmd == "list":
        for i in zf.infolist():
            print(f"{i.file_size}\t{i.compress_size}\t{i.filename}")
    elif cmd == "extract":
        out_dir, prefixes = sys.argv[3], sys.argv[4:]
        members = [i for i in zf.infolist() if any(i.filename.startswith(p) for p in prefixes)]
        total = sum(i.compress_size for i in members)
        print(f"extracting {len(members)} members, {total / 1e9:.2f} GB compressed", file=sys.stderr)
        for i in members:
            dest = os.path.realpath(os.path.join(out_dir, i.filename))
            if not dest.startswith(os.path.realpath(out_dir) + os.sep):
                raise RuntimeError(f"unsafe path {i.filename}")
            zf.extract(i, out_dir)
            print(i.filename, file=sys.stderr)


if __name__ == "__main__":
    main()
