"""Drop a consumed checkpoint shard's mapped pages from the loading process's RSS.

The safetensors loader hands out zero-copy views into a private mmap of each shard, and
every page a rank touches while copying its slices to the GPU stays resident until all of
that shard's tensors are freed; with several shards in flight per rank this is the
service's host-memory peak (all ranks read every shard). Once the weight stream moves on
to the next shard file, MADV_DONTNEED drops the previous file's pages. They are clean file
pages (nothing writes the CPU views), so a later read of one of its tensors (an async
copy still in flight, a tensor cached for fusion) refaults it from the file.
"""

import bisect
import ctypes
import logging
from typing import Iterable, Iterator, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_MADV_DONTNEED = 4
# Arbitrary bound on /proc/self/maps rescans for tensors outside any known shard mapping
# (e.g. ones a streaming transform allocated); past it, lookups stop rescanning.
_MAX_RESCANS = 256

_Region = Tuple[int, int, str]


class _ShardMappings:
    def __init__(self) -> None:
        self._starts: List[int] = []
        self._regions: List[_Region] = []
        self._rescans = 0
        self._libc = ctypes.CDLL(None, use_errno=True)
        self.released_bytes = 0

    def _rescan(self) -> None:
        regions = []
        with open("/proc/self/maps") as maps:
            for line in maps:
                parts = line.split()
                if len(parts) < 6 or not parts[5].endswith(".safetensors"):
                    continue
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                regions.append((lo, hi, parts[5]))
        regions.sort()
        self._regions = regions
        self._starts = [r[0] for r in regions]
        self._rescans += 1

    def _find(self, ptr: int) -> Optional[_Region]:
        i = bisect.bisect_right(self._starts, ptr) - 1
        if i >= 0 and ptr < self._regions[i][1]:
            return self._regions[i]
        return None

    def region_of(self, tensor) -> Optional[_Region]:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.numel() == 0
        ):
            return None
        ptr = tensor.data_ptr()
        region = self._find(ptr)
        if region is None and self._rescans < _MAX_RESCANS:
            self._rescan()
            region = self._find(ptr)
        return region

    def release(self, region: _Region) -> None:
        lo, hi, _ = region
        rc = self._libc.madvise(
            ctypes.c_void_p(lo), ctypes.c_size_t(hi - lo), _MADV_DONTNEED
        )
        if rc == 0:
            self.released_bytes += hi - lo


def release_consumed_checkpoint_pages(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Pass ``weights`` through, dropping each shard file's pages once the stream
    has moved past it; tensors outside a .safetensors mapping pass untouched."""
    mappings = _ShardMappings()
    current: Optional[_Region] = None
    for name, tensor in weights:
        region = mappings.region_of(tensor)
        if region is not None and region != current:
            if current is not None:
                mappings.release(current)
            current = region
        yield name, tensor
    if current is not None:
        mappings.release(current)
    logger.info(
        "Released %.1f GiB of consumed checkpoint mappings during weight loading.",
        mappings.released_bytes / (1 << 30),
    )
