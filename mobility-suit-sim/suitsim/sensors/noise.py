"""Generic sensor-imperfection utilities: a fixed-step delay line and Gaussian noise.

Both are deliberately tiny and allocation-free in the hot path (``record`` runs at the
physics rate, 1 kHz).
"""
from __future__ import annotations

import numpy as np


def delay_steps(delay_s: float, dt: float) -> int:
    """Delay in seconds -> whole physics steps (rounded, never negative)."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    return max(0, int(round(float(delay_s) / float(dt))))


class DelayLine:
    """Ring buffer that returns the sample pushed exactly ``n_steps`` pushes ago.

    ``push(x)`` stores a copy of ``x`` (any fixed-shape array / scalar).  ``get()`` returns
    the value pushed ``n_steps`` pushes before the most recent one (``n_steps == 0`` -> the
    most recent value).  Before enough history exists the oldest available sample is
    returned (the buffer is pre-filled by :meth:`fill`), so a reading is always defined.
    """

    def __init__(self, n_steps: int, shape: tuple[int, ...] | int = (), dtype=float):
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        self.n_steps = int(n_steps)
        self._len = self.n_steps + 1
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self._buf = np.zeros((self._len, *shape), dtype=dtype)
        self._t = np.zeros(self._len)
        self._head = -1           # index of most recent sample, -1 = empty

    @property
    def shape(self) -> tuple[int, ...]:
        return self._buf.shape[1:]

    def fill(self, x, t: float = 0.0) -> None:
        """Reset the history so every slot holds ``x`` (time stamp ``t``)."""
        self._buf[...] = x
        self._t[:] = t
        self._head = self._len - 1

    def push(self, x, t: float = 0.0) -> None:
        if self._head < 0:
            self.fill(x, t)
            return
        self._head = (self._head + 1) % self._len
        self._buf[self._head] = x
        self._t[self._head] = t

    def _idx(self) -> int:
        if self._head < 0:
            raise RuntimeError("DelayLine.get() before any sample was pushed")
        # slot after head is the oldest = n_steps pushes ago
        return (self._head + 1) % self._len

    def get(self) -> np.ndarray:
        """Delayed sample (a view into the buffer; copy it if you keep it)."""
        return self._buf[self._idx()]

    def get_time(self) -> float:
        """Time stamp of the delayed sample."""
        return float(self._t[self._idx()])

    def latest(self) -> np.ndarray:
        return self._buf[self._head]


class GaussianNoise:
    """Additive zero-mean Gaussian noise.

    ``std`` is an absolute standard deviation; with ``relative=True`` it is a fraction of
    the signal magnitude (``std_i = std * |x_i|``).  ``scale`` multiplies the std (domain
    randomisation of sensor quality).  Noise is drawn from the generator passed to
    :meth:`apply`, so seeded runs are reproducible.
    """

    def __init__(self, std: float, relative: bool = False, scale: float = 1.0):
        self.std = float(std)
        self.relative = bool(relative)
        self.scale = float(scale)

    @property
    def effective_std(self) -> float:
        return self.std * self.scale

    def apply(self, x, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        s = self.effective_std
        if s == 0.0:
            return x.copy()
        n = rng.standard_normal(x.shape)
        if self.relative:
            return x + n * (s * np.abs(x))
        return x + n * s
