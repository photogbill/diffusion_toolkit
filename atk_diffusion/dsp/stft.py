# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The detector's front end: the profile's spectrogram, cut into tiles in dB
above the measured noise floor (DETECTION_DESIGN §2; decisions D5, D7).

WHAT. `spectrogram(x, fs, geom)` is the profile's STFT (window, FFT size and
hop from `StftGeometry`, bins shifted so they run low -> high frequency,
power in dB). `TileBuilder` turns a stream of IQ blocks into `Tile`s — fixed
windows of time x the whole span, time max-pooled to `geom.tile_rows` rows,
in dB ABOVE the floor (`dsp.floor.NoiseFloor`), with the absolute level kept
beside and never fed to a model. `tiles(...)` is the one-shot form.

WHY THIS SHAPE.
* *"STFT parameters are part of the profile"* (§2): a model never meets a
  spectrogram of a geometry it was not trained on, so the geometry comes
  from the profile and nothing here takes an FFT size as a casual argument.
* *"Inputs are dB above the measured noise floor"* (§2, D5): gain, LNA state
  and receiver sensitivity are removed before anything learned sees a pixel —
  the largest single domain-gap reducer there is. `abs_db` keeps the
  absolute level for measurement only.
* *Max-pooling in time* keeps a burst shorter than a row at its peak level
  instead of averaging it into the noise; `mean_above` (the mean-pooled
  power, linear, in floor units) is kept beside for honest SNR numbers.
* *Tiles overlap* by `geom.tile_overlap` so a burst on a boundary is whole in
  one of them. Every tile also carries an OWNERSHIP window (`own_t0`,
  `own_t1`): consecutive ownership windows tile the stream exactly, so the
  pipeline keeps each burst once and stitches long signals across tiles
  without gaps or doubles.
* *Streaming is chunk-invariant.* Frames sit on one global grid (frame k is
  samples [k·hop, k·hop+fft) of the stream), tiles on a grid of whole rows,
  and the floor is updated once per tile from that tile's own frames — so one
  big block or a thousand small ones give identical tiles (the test feeds
  both and demands equality).

GEOMETRY, MEASURED NOT ROUNDED AWAY (D7 is pending measurement). A row is a
WHOLE number of frames (`pool`), so every row covers exactly the same time and
pixel <-> (t, f) conversion is exact in both directions — the dataset builder,
the 2D proposer and ATK's waterfall all agree on where a box is. The price:
the tile span is `tile_rows x pool x hop / fs`, the nearest such span to
`geom.tile_seconds`, not exactly it. For the RTL-SDR at 2.4 MS/s with a
1024-point FFT and 512 rows that is 5 frames a row and 1.092 s a tile (asked
1.0 s); `tile_layout(...).words()` says it for any profile.

UNITS. Power per bin is |FFT(x·w)|² / (Σw)²: a tone of amplitude A centred on
a bin reads A² (its power in dBFS, since full scale is 1.0 — dsp.iq); white
noise of variance σ² reads σ²·Σw²/(Σw)² per bin. Pixel boxes are continuous
coordinates in which pixel i spans [i, i+1): (row0, bin0, row1, bin1) with
row1/bin1 exclusive, the torchvision convention the 2D proposer is trained in
(ARCHITECTURE §5). Times are stream seconds; frequencies are absolute Hz.

LIMITS. A frame is fft_size/fs long (0.43 ms at 2.4 MS/s with 1024 points);
nothing shorter is resolved in time, and with hop < fft_size a box can start
up to one frame before the burst it contains (boxes are conservative). A
partial frame at the very end of a stream (< fft_size samples) is dropped.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from atk_diffusion.profiles import StftGeometry

_TINY = 1e-30

#: Frames transformed per FFT call. Bounds memory when a whole capture is
#: pushed in one block (a minute at 2.4 MS/s is 140,000 frames).
FFT_BATCH = 2048

#: Frames used to estimate or track the floor per tile (strided subset). The
#: per-bin quantile of 512 exponential samples is good to ~0.35 dB, and the
#: tracking averages many tiles, so more would cost time and buy nothing.
FLOOR_FRAMES = 512

_WINDOW_ALIASES = {"rect": "boxcar", "rectangular": "boxcar", "none": "boxcar",
                   "hanning": "hann"}


@lru_cache(maxsize=32)
def _window(name: str, n: int) -> np.ndarray:
    from scipy.signal import get_window
    key = _WINDOW_ALIASES.get(name.strip().lower(), name.strip().lower())
    try:
        w = get_window(key, int(n), fftbins=True)
    except ValueError:
        raise ValueError(f"the STFT window {name!r} is not one scipy knows "
                         "(hann, hamming, blackman, boxcar, …); fix the "
                         "profile's stft.window") from None
    w = np.asarray(w, dtype=np.float32)
    w.setflags(write=False)
    return w


def window_array(name: str, n: int) -> np.ndarray:
    """The periodic (DFT-even) window the profile names, float32, read-only."""
    return _window(str(name), int(n))


def frame_power(frames: np.ndarray, window: np.ndarray) -> np.ndarray:
    """|FFT(frame·w)|² / (Σw)² per frame, bins shifted low -> high, float32.

    `frames` is (n, fft) complex; the transform is one batched pocketfft call
    with one worker (bitwise identical to transforming row by row — the
    chunk-invariance test depends on it, and it is measured, not assumed)."""
    import scipy.fft as sf
    fr = np.asarray(frames)
    if fr.dtype != np.complex64:
        fr = fr.astype(np.complex64)
    y = sf.fft(fr * window, axis=-1, workers=1)
    p = y.real * y.real + y.imag * y.imag
    p /= np.float32(float(np.sum(window, dtype=np.float64)) ** 2)
    return np.fft.fftshift(p, axes=-1).astype(np.float32, copy=False)


def _frames_view(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    if x.size < n:
        return np.zeros((0, n), dtype=np.complex64)
    return np.lib.stride_tricks.sliding_window_view(x, n)[::hop]


def to_db(p) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(p, _TINY))).astype(np.float32)


def spectrogram(x, fs: float, geom: StftGeometry, t_start: float = 0.0):
    """The profile's spectrogram of `x`: (S_db float32 [frames, bins],
    frame_times float64 [frames]). Bins run low -> high frequency (bin
    `fft_size // 2` is the tuner centre); frame time is the stream time of
    the frame's first sample. Only whole frames are returned."""
    x = np.asarray(x, dtype=np.complex64).ravel()
    n, hop = int(geom.fft_size), int(geom.hop)
    w = window_array(geom.window, n)
    view = _frames_view(x, n, hop)
    out = np.empty((view.shape[0], n), dtype=np.float32)
    for i in range(0, view.shape[0], FFT_BATCH):
        out[i:i + FFT_BATCH] = to_db(frame_power(view[i:i + FFT_BATCH], w))
    times = float(t_start) + np.arange(view.shape[0], dtype=np.float64) * hop / float(fs)
    return out, times


def frame_freqs(fs: float, fft_size: int, center_hz: float = 0.0) -> np.ndarray:
    """Absolute centre frequency of every (shifted) bin."""
    n = int(fft_size)
    return float(center_hz) + (np.arange(n) - n // 2) * (float(fs) / n)


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TileLayout:
    """The exact tile geometry a profile's StftGeometry produces at `fs`."""
    fs: float
    fft_size: int
    hop: int
    window: str
    rows: int
    pool: int              # frames max-pooled into one row
    frames: int            # frames per tile = rows * pool
    overlap_rows: int      # rows shared with the next tile
    step_rows: int         # rows between consecutive tile starts
    bin_hz: float
    row_period: float      # seconds per row = pool * hop / fs
    seconds: float         # nominal tile span = rows * row_period
    requested_seconds: float

    @property
    def step_frames(self) -> int:
        return self.step_rows * self.pool

    @property
    def half_overlap_rows(self) -> int:
        return self.overlap_rows // 2

    @property
    def overlap_seconds(self) -> float:
        return self.overlap_rows * self.row_period

    @property
    def bins(self) -> int:
        return self.fft_size

    def words(self) -> str:
        return (f"{self.fft_size}-point {self.window} FFT, hop {self.hop}: "
                f"{self.bin_hz / 1e3:.3g} kHz bins; {self.pool} frame"
                f"{'s' if self.pool != 1 else ''} per row; {self.rows} rows = "
                f"{self.seconds:.4g} s per tile (the profile asks "
                f"{self.requested_seconds:g} s; a row is a whole number of "
                f"frames), {self.overlap_rows} rows ({self.overlap_seconds:.3g} s) "
                "shared with the next tile")


def tile_layout(fs: float, geom: StftGeometry) -> TileLayout:
    """Pure arithmetic: the tile geometry of `geom` at `fs` (see the module
    docstring for why a row is a whole number of frames)."""
    fs = float(fs)
    n, hop, rows = int(geom.fft_size), int(geom.hop), int(geom.tile_rows)
    if fs <= 0:
        raise ValueError("a sample rate must be positive")
    if n < 8:
        raise ValueError(f"an FFT of {n} points is too small for a detector")
    if hop < 1:
        raise ValueError("the STFT hop must be at least one sample")
    if rows < 1:
        raise ValueError("a tile needs at least one row")
    if not 0.0 <= float(geom.tile_overlap) < 1.0:
        raise ValueError(f"tile_overlap must be in [0, 1), got {geom.tile_overlap!r}")
    if float(geom.tile_seconds) <= 0:
        raise ValueError("tile_seconds must be positive")
    window_array(geom.window, n)                  # validates the window name
    nominal = float(geom.tile_seconds) * fs / hop
    pool = max(1, int(round(nominal / rows)))
    overlap = int(round(rows * float(geom.tile_overlap)))
    overlap = min(max(overlap, 0), rows - 1)
    rp = pool * hop / fs
    return TileLayout(fs=fs, fft_size=n, hop=hop, window=str(geom.window),
                      rows=rows, pool=pool, frames=rows * pool,
                      overlap_rows=overlap, step_rows=rows - overlap,
                      bin_hz=fs / n, row_period=rp, seconds=rows * rp,
                      requested_seconds=float(geom.tile_seconds))


# ---------------------------------------------------------------------------
# The tile
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    """One tile: a fixed window of time x the whole span.

    spec        float32 [rows, bins] — dB ABOVE the floor, max-pooled in time;
                the only array a model is ever fed
    abs_db      float32 [rows, bins] — absolute dB per bin (dBFS scaling, same
                pooling); for measurement beside the tile, never a model input
    mean_above  float32 [rows, bins] — mean-pooled power in floor units
                (1.0 = the floor); for SNR estimates that max-pooling would
                inflate
    t0, t1      stream seconds of row 0's start and the last row's end
    f0, f1      absolute Hz of the outer bin edges (f1 - f0 = fs)
    rows_valid  rows holding data (the final tile of a stream is padded with
                the floor beyond them)
    own_t0/1    the ownership window (module docstring)
    floor_ok    False when the floor moved by more than the margin since the
                last tile (a gain change) — floor-referenced tests stand down
    """
    spec: np.ndarray
    abs_db: np.ndarray
    mean_above: np.ndarray
    t0: float
    t1: float
    f0: float
    f1: float
    bin_hz: float
    row_period: float
    center_hz: float
    fs: float
    layout: TileLayout
    index: int = 0
    rows_valid: int = 0
    own_t0: float = 0.0
    own_t1: float = 0.0
    first: bool = True
    final: bool = False
    floor_ok: bool = True
    floor_offset_db: float = 0.0
    floor_source: str = ""
    profile: str = ""
    epoch: float | None = None

    # -- shape ---------------------------------------------------------------
    @property
    def rows(self) -> int:
        return int(self.spec.shape[0])

    @property
    def bins(self) -> int:
        return int(self.spec.shape[1])

    @property
    def duration(self) -> float:
        return float(self.t1 - self.t0)

    @property
    def freqs(self) -> np.ndarray:
        """Absolute centre frequency of every bin."""
        return self.f0 + (np.arange(self.bins) + 0.5) * self.bin_hz

    @property
    def data_t1(self) -> float:
        """End of the last row that holds data."""
        return self.t0 + self.rows_valid * self.row_period

    # -- pixel <-> (t, f) ------------------------------------------------------
    def row_to_t(self, row: float) -> float:
        return self.t0 + float(row) * self.row_period

    def t_to_row(self, t: float) -> float:
        return (float(t) - self.t0) / self.row_period

    def bin_to_f(self, b: float) -> float:
        return self.f0 + float(b) * self.bin_hz

    def f_to_bin(self, f: float) -> float:
        return (float(f) - self.f0) / self.bin_hz

    def pixels_to_tf(self, row0: float, bin0: float, row1: float,
                     bin1: float) -> tuple[float, float, float, float]:
        """A pixel box (continuous; row1/bin1 exclusive) -> (t0, t1, f_lo,
        f_hi) in stream seconds and absolute Hz."""
        r0, r1 = sorted((float(row0), float(row1)))
        b0, b1 = sorted((float(bin0), float(bin1)))
        return (self.row_to_t(r0), self.row_to_t(r1),
                self.bin_to_f(b0), self.bin_to_f(b1))

    def tf_to_pixels(self, t0: float, t1: float, f_lo: float, f_hi: float,
                     clip: bool = True) -> tuple[float, float, float, float]:
        """(t0, t1, f_lo, f_hi) -> a continuous pixel box (row0, bin0, row1,
        bin1), clipped to the tile unless `clip=False`. The exact inverse of
        `pixels_to_tf` inside the tile."""
        r0, r1 = sorted((self.t_to_row(t0), self.t_to_row(t1)))
        b0, b1 = sorted((self.f_to_bin(f_lo), self.f_to_bin(f_hi)))
        if clip:
            r0, r1 = min(max(r0, 0.0), self.rows), min(max(r1, 0.0), self.rows)
            b0, b1 = min(max(b0, 0.0), self.bins), min(max(b1, 0.0), self.bins)
        return r0, b0, r1, b1

    def cell_slices(self, row0: float, bin0: float, row1: float,
                    bin1: float) -> tuple[slice, slice]:
        """The whole pixels a continuous box touches, clipped to valid rows."""
        r0 = max(0, int(math.floor(row0)))
        r1 = min(self.rows_valid, int(math.ceil(row1)))
        b0 = max(0, int(math.floor(bin0)))
        b1 = min(self.bins, int(math.ceil(bin1)))
        return slice(r0, max(r0, r1)), slice(b0, max(b0, b1))

    def box_snr_db(self, row0: float, bin0: float, row1: float, bin1: float,
                   mask: np.ndarray | None = None) -> float | None:
        """Mean in-box SNR in dB above the floor, from the MEAN-pooled power
        (max-pooling would inflate it): 10·log10(mean(P/floor) − 1) over the
        box's cells (or the cells `mask` marks, same shape as the box). None
        when the box holds no data; floored at −20 dB."""
        rs, bs = self.cell_slices(row0, bin0, row1, bin1)
        cells = self.mean_above[rs, bs]
        if cells.size == 0:
            return None
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape == cells.shape and m.any():
                cells = cells[m]
        s = float(np.mean(cells, dtype=np.float64)) - 1.0
        return float(10.0 * math.log10(max(s, 0.01)))

    # -- construction for tests and synthetic data -----------------------------
    @classmethod
    def from_spec(cls, spec, fs: float, center_hz: float, geom: StftGeometry,
                  t0: float = 0.0, profile: str = "", epoch: float | None = None,
                  floor_db: float | np.ndarray = 0.0) -> "Tile":
        """A tile from a given dB-above-floor array (synthetic datasets, tests,
        a denoiser's output). Its shape must be the geometry's [rows, bins]."""
        lay = tile_layout(fs, geom)
        s = np.asarray(spec, dtype=np.float32)
        if s.shape != (lay.rows, lay.bins):
            raise ValueError(f"a tile for this geometry is {lay.rows} x "
                             f"{lay.bins}; got {s.shape[0]} x {s.shape[1]}"
                             if s.ndim == 2 else "a tile is a 2D array")
        f0 = float(center_hz) - (lay.bins // 2 + 0.5) * lay.bin_hz
        fl = np.asarray(floor_db, dtype=np.float32)
        return cls(spec=s, abs_db=(s + fl).astype(np.float32),
                   mean_above=np.power(10.0, s / 10.0).astype(np.float32),
                   t0=float(t0), t1=float(t0) + lay.seconds, f0=f0,
                   f1=f0 + lay.bins * lay.bin_hz, bin_hz=lay.bin_hz,
                   row_period=lay.row_period, center_hz=float(center_hz),
                   fs=float(fs), layout=lay, rows_valid=lay.rows,
                   own_t0=float(t0), own_t1=float(t0) + lay.seconds,
                   profile=str(profile), epoch=epoch)

    def with_spec(self, spec) -> "Tile":
        """A copy whose spec is replaced (the low-SNR path's denoised tile).
        The absolute level and the mean-pooled power stay the RAW ones: a
        measurement is never made on a reconstruction."""
        s = np.asarray(spec, dtype=np.float32)
        if s.shape != self.spec.shape:
            raise ValueError(f"the replacement spectrogram is {s.shape}; the "
                             f"tile is {self.spec.shape}")
        t = Tile(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        t.spec = s
        return t


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
class _FrameStore:
    """Frames (linear power) by global index, kept as chunks so appending a
    few frames never copies the whole buffer."""

    def __init__(self, bins: int):
        self.bins = bins
        self.start = 0          # global index of the first kept frame
        self.count = 0          # frames kept
        self._chunks: deque = deque()

    @property
    def end(self) -> int:
        return self.start + self.count

    def append(self, arr: np.ndarray) -> None:
        if arr.shape[0]:
            self._chunks.append(arr)
            self.count += arr.shape[0]

    def get(self, k0: int, k1: int) -> np.ndarray:
        """Frames [k0, k1) (must be kept), as one contiguous array."""
        if k0 < self.start or k1 > self.end or k1 < k0:
            raise IndexError(f"frames {k0}..{k1} not held ({self.start}..{self.end})")
        parts, pos = [], self.start
        for ch in self._chunks:
            n = ch.shape[0]
            lo, hi = max(k0, pos), min(k1, pos + n)
            if lo < hi:
                parts.append(ch[lo - pos:hi - pos])
            pos += n
            if pos >= k1:
                break
        if not parts:
            return np.zeros((0, self.bins), dtype=np.float32)
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def drop_before(self, k: int) -> None:
        while self._chunks and self.start + self._chunks[0].shape[0] <= k:
            n = self._chunks.popleft().shape[0]
            self.start += n
            self.count -= n
        if self._chunks and self.start < k:
            cut = k - self.start
            self._chunks[0] = self._chunks[0][cut:].copy()
            self.start += cut
            self.count -= cut


class TileBuilder:
    """Blocks of IQ in, complete tiles out (the module docstring has the
    rules). One builder per (tuner centre, contiguous stream): a retune or a
    gap in the samples is a new builder — no tile, and so no box, spans one.

        b = TileBuilder(fs, center_hz, profile.stft, floor=NoiseFloor.from_profile(p))
        for block in blocks:
            for tile in b.push(block): ...
        for tile in b.flush(): ...          # the end of the stream

    `floor`: a NoiseFloor (measured terminated, from the profile, or carried
    over from before a gap) or None — then it is estimated robustly from the
    first tile's own frames. Either way it is tracked slowly, once per tile,
    from that tile's new frames, and a tile is made with the floor as it was
    BEFORE its own frames moved it (causal; a tile never normalises itself).
    """

    def __init__(self, fs: float, center_hz: float, geom: StftGeometry,
                 floor=None, t_start: float = 0.0, profile: str = "",
                 epoch: float | None = None, floor_margin_db: float = 1.0):
        self.layout = tile_layout(fs, geom)
        self.geom = geom
        self.fs = float(fs)
        self.center_hz = float(center_hz)
        self.t_start = float(t_start)
        self.profile = str(profile)
        self.epoch = epoch
        self.floor = floor
        self.floor_margin_db = float(floor_margin_db)
        self.floor_info: dict = {}
        self.samples_in = 0
        self.tiles_out = 0
        self.closed = False
        lay = self.layout
        self._win = window_array(lay.window, lay.fft_size)
        self._pending = np.zeros(0, dtype=np.complex64)
        self._pending_start = 0          # global sample index of _pending[0]
        self._n_frames = 0               # frames computed (global count)
        self._store = _FrameStore(lay.bins)
        self._next_tile = 0
        self._floor_upto = 0             # frames already given to the floor

    # -- public --------------------------------------------------------------
    @property
    def frames_done(self) -> int:
        return self._n_frames

    @property
    def stream_t(self) -> float:
        """Stream time just after the last sample pushed."""
        return self.t_start + self.samples_in / self.fs

    @property
    def next_tile_t0(self) -> float:
        """Stream time at which the next tile will start: nothing before it
        is needed again (the pipeline prunes its IQ history to here)."""
        lay = self.layout
        return self.t_start + self._next_tile * lay.step_frames * lay.hop / self.fs

    def push(self, block) -> list[Tile]:
        """Add samples; return every tile they complete."""
        return list(self.push_iter(block))

    def push_iter(self, block):
        if self.closed:
            raise RuntimeError("this tile builder was flushed (its stream "
                               "ended); start a new one for new samples")
        x = np.asarray(block, dtype=np.complex64).ravel()
        if x.size == 0:
            return
        self.samples_in += x.size
        self._pending = x if self._pending.size == 0 else \
            np.concatenate([self._pending, x])
        lay = self.layout
        n, hop = lay.fft_size, lay.hop
        total = self._pending_start + self._pending.size
        if total < n:
            return
        last = (total - n) // hop                      # last complete frame
        while self._n_frames <= last:
            k0 = self._n_frames
            k1 = min(last + 1, k0 + FFT_BATCH)
            off = k0 * hop - self._pending_start
            seg = self._pending[off:off + (k1 - k0 - 1) * hop + n]
            self._store.append(frame_power(_frames_view(seg, n, hop), self._win))
            self._n_frames = k1
            yield from self._emit_ready()
        drop = self._n_frames * hop - self._pending_start
        if drop > 0:
            self._pending = self._pending[drop:].copy()
            self._pending_start += drop

    def flush(self) -> list[Tile]:
        """End of stream: the last, partial tile (padded with the floor past
        its data), if any data is not yet owned by an emitted tile."""
        if self.closed:
            return []
        self.closed = True
        lay = self.layout
        s = self._next_tile * lay.step_frames
        # frames up to here are owned by the last emitted tile
        owned_to = s + lay.half_overlap_rows * lay.pool if self.tiles_out else 0
        if self._n_frames <= max(owned_to, s):
            return []
        valid = self._n_frames - s
        if self.floor is None and valid < 4:
            return []           # a few hundred microseconds: nothing to measure against
        return [self._make_tile(s, valid, final=True)]

    # -- internals -------------------------------------------------------------
    def _emit_ready(self):
        lay = self.layout
        while True:
            s = self._next_tile * lay.step_frames
            if self._n_frames < s + lay.frames:
                return
            tile = self._make_tile(s, lay.frames, final=False)
            self._next_tile += 1
            self._store.drop_before(self._next_tile * lay.step_frames)
            yield tile

    def _floor_subset(self, frames: np.ndarray) -> np.ndarray:
        stride = max(1, frames.shape[0] // FLOOR_FRAMES)
        return frames[::stride]

    def _make_tile(self, s: int, valid: int, final: bool) -> Tile:
        from atk_diffusion.dsp.floor import NoiseFloor
        lay = self.layout
        fr = self._store.get(s, s + valid)
        offset, ok = 0.0, True
        if self.floor is None:
            # nothing measured: estimate robustly from this tile's own frames
            self.floor = NoiseFloor.estimate(self._floor_subset(fr))
            self.floor_info = {"estimated_from_frames": int(fr.shape[0])}
            self._floor_upto = s + valid
        elif getattr(self.floor, "needs_alignment", False):
            # a measured SHAPE (terminated capture): level it to today's gain once
            moved = float(self.floor.align(self._floor_subset(fr)))
            self.floor_info = {"aligned_db": moved}
            self._floor_upto = s + valid
        pre_db = self.floor.floor_db.astype(np.float32).copy()
        new0 = max(s, self._floor_upto)
        if s + valid - new0 >= 4:
            # use-then-update: this tile is normalised by the floor as it was
            # before its own frames moved it
            new = fr[new0 - s:]
            info = self.floor.update(self._floor_subset(new),
                                     duration_s=new.shape[0] * lay.hop / self.fs)
            self.floor_info = info
            offset = float(info.get("offset_db", 0.0))
            ok = abs(offset) <= self.floor_margin_db
            self._floor_upto = s + valid
        pre_lin = np.power(10.0, pre_db / 10.0).astype(np.float32)
        if valid < lay.frames:
            pad = np.broadcast_to(pre_lin, (lay.frames - valid, lay.bins))
            fr = np.concatenate([fr, pad], axis=0)
        cube = fr.reshape(lay.rows, lay.pool, lay.bins)
        mx = cube.max(axis=1)
        mean = cube.mean(axis=1, dtype=np.float32) if lay.pool > 1 else cube[:, 0, :]
        abs_db = to_db(mx)
        spec = (abs_db - pre_db).astype(np.float32)
        mean_above = (mean / pre_lin).astype(np.float32)
        t0 = self.t_start + s * lay.hop / self.fs
        rp = lay.row_period
        rows_valid = int(math.ceil(valid / lay.pool))
        if rows_valid < lay.rows:
            # rows past the data are the floor exactly (not floor to rounding)
            spec[rows_valid:] = 0.0
            abs_db[rows_valid:] = pre_db
            mean_above[rows_valid:] = 1.0
        first = self.tiles_out == 0
        own_t0 = t0 if first else t0 + lay.half_overlap_rows * rp
        own_t1 = (t0 + rows_valid * rp) if final else \
            t0 + (lay.step_rows + lay.half_overlap_rows) * rp
        f0 = self.center_hz - (lay.bins // 2 + 0.5) * lay.bin_hz
        tile = Tile(spec=spec, abs_db=abs_db, mean_above=mean_above,
                    t0=t0, t1=t0 + lay.rows * rp, f0=f0,
                    f1=f0 + lay.bins * lay.bin_hz, bin_hz=lay.bin_hz,
                    row_period=rp, center_hz=self.center_hz, fs=self.fs,
                    layout=lay, index=self.tiles_out, rows_valid=rows_valid,
                    own_t0=own_t0, own_t1=max(own_t0, own_t1), first=first,
                    final=final, floor_ok=ok, floor_offset_db=offset,
                    floor_source=getattr(self.floor, "source", ""),
                    profile=self.profile, epoch=self.epoch)
        self.tiles_out += 1
        return tile


def tiles(x, fs: float, center_hz: float, geom: StftGeometry, floor=None,
          t_start: float = 0.0, profile: str = "", epoch: float | None = None,
          final: bool = True):
    """Every tile of `x` (one-shot form of TileBuilder): an iterator of Tile.
    `final=False` leaves out the end-of-stream partial tile."""
    b = TileBuilder(fs, center_hz, geom, floor=floor, t_start=t_start,
                    profile=profile, epoch=epoch)
    yield from b.push_iter(x)
    if final:
        yield from b.flush()
