"""Consumed checkpoint pages leave the loading process's RSS, and the weights stay readable.

The safetensors loader hands out views into a private mmap of each shard; the pages a rank
touches stay resident until loading ends unless they are released (DeepSeek-V4.1's host-memory
peak). These cases fail if the release is removed or if it releases a shard still in use.
"""

import mmap
import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.layers import engram
from sglang.srt.models.deepseek_common.checkpoint_page_release import (
    release_consumed_checkpoint_pages,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SHARD_BYTES = 8 << 20
COLS = 4096


def _pattern(seed):
    return (
        (torch.arange(SHARD_BYTES, dtype=torch.int64) * 7 + seed)
        .remainder(251)
        .to(torch.uint8)
    )


def _rss_kib(tensor):
    """Resident KiB of the mapping that holds ``tensor``, from /proc/self/smaps."""
    ptr = tensor.data_ptr()
    inside = False
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            head = line.split()
            if head and "-" in head[0] and len(head) >= 5:
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                inside = lo <= ptr < hi
            elif inside and line.startswith("Rss:"):
                return int(line.split()[1])
    raise AssertionError("tensor is not inside a mapping")


class _Shards:
    """Private, writable-but-unwritten mmaps of ``*.safetensors`` files, as the loader maps them."""

    def __init__(self):
        self._dir = tempfile.TemporaryDirectory()
        self._maps = []

    def map(self, name, seed):
        path = os.path.join(self._dir.name, name)
        with open(path, "wb") as f:
            f.write(_pattern(seed).numpy().tobytes())
        fd = os.open(path, os.O_RDONLY)
        try:
            m = mmap.mmap(
                fd, 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
        finally:
            os.close(fd)
        self._maps.append(m)
        tensor = torch.frombuffer(m, dtype=torch.uint8)
        tensor.sum()  # fault every page in
        return tensor

    def close(self):
        self._maps.clear()
        self._dir.cleanup()


class TestCheckpointPageRelease(CustomTestCase):
    def setUp(self):
        self.shards = _Shards()

    def tearDown(self):
        self.shards.close()

    def assert_released(self, tensor):
        self.assertLess(_rss_kib(tensor), SHARD_BYTES // 1024 // 10)

    def test_stream_releases_each_shard_only_after_moving_past_it(self):
        a = self.shards.map("model-00001-of-00002.safetensors", seed=1)
        b = self.shards.map("model-00002-of-00002.safetensors", seed=2)
        plain = torch.arange(16)
        half = SHARD_BYTES // 2
        weights = [("a.0", a[:half]), ("a.1", a[half:]), ("b.0", b), ("plain", plain)]
        stream = release_consumed_checkpoint_pages(iter(weights))

        seen = []
        for name, tensor in stream:
            seen.append((name, tensor))
            if name == "a.1":
                # Still inside shard a: none of its pages may be released yet.
                self.assertGreater(_rss_kib(a), SHARD_BYTES // 1024 // 2)
            if name == "b.0":
                self.assert_released(a)
                self.assertGreater(_rss_kib(b), SHARD_BYTES // 1024 // 2)
        self.assert_released(b)

        self.assertEqual([n for n, _ in seen], [n for n, _ in weights])
        for (_, got), (_, sent) in zip(seen, weights):
            self.assertIs(got, sent)
        # Released pages are clean file pages: reading them again refaults the original bytes.
        torch.testing.assert_close(a, _pattern(1))
        torch.testing.assert_close(b, _pattern(2))
        torch.testing.assert_close(plain, torch.arange(16))

    def test_host_table_copy_is_exact_and_releases_the_source(self):
        src = self.shards.map("engram.safetensors", seed=3).view(-1, COLS)
        dst = torch.empty_like(src)
        # Several chunks, so chunk boundaries and the last partial chunk are exercised.
        with mock.patch.object(engram, "_HOST_LOAD_CHUNK_BYTES", (1 << 20) + 3 * COLS):
            engram._copy_releasing_checkpoint_pages(dst, src)
        expected = _pattern(3).view(-1, COLS)
        torch.testing.assert_close(dst, expected)
        self.assert_released(src)
        torch.testing.assert_close(src, expected)

    def test_host_table_copy_falls_back_to_a_plain_copy(self):
        src = self.shards.map("engram.safetensors", seed=4).view(-1, COLS)
        expected = _pattern(4).view(-1, COLS)
        # Non-contiguous source: copied without any release.
        strided = src[:, ::2]
        dst = torch.empty_like(strided)
        engram._copy_releasing_checkpoint_pages(dst, strided)
        torch.testing.assert_close(dst, expected[:, ::2])
        self.assertGreater(_rss_kib(src), SHARD_BYTES // 1024 // 2)
        # A source outside any .safetensors mapping is copied as is.
        heap = expected.clone()
        dst = torch.empty_like(heap)
        engram._copy_releasing_checkpoint_pages(dst, heap)
        torch.testing.assert_close(dst, expected)


if __name__ == "__main__":
    unittest.main()
