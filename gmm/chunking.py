"""Re-iterable row-chunk sources for memory-bounded (chunked) EM.

Streaming EM sweeps the whole dataset once per iteration, so a source must yield
the *same* rows on every pass. Two input shapes are supported:

* an in-memory array / tensor -> sliced into ``chunk_size`` row-blocks. Only one
  block is moved to the compute device at a time, so the ``(chunk, k, d)``
  E-step transient never scales with the full sample count.
* a zero-argument callable returning a *fresh* iterator of row-blocks -> for
  out-of-core data that never lives in memory as one array. The callable is
  re-invoked once per sweep, so it must be replayable.

Both paths bottom out in :class:`ChunkedData`, which the EM driver iterates.
Initialization (k-means++) is inherently multi-pass and forms its own
``(n, k, d)`` / ``(n, d, d)`` intermediates, so :func:`subsample_for_init`
draws a bounded uniform sample in a single pass to seed it instead.
"""

from __future__ import annotations

import numpy as np


class ChunkedData:
    """A replayable source of row-chunks over the joint ``[source | target]`` data.

    Parameters
    ----------
    data : array-like or callable
        Either an array/tensor of shape ``(n_samples, 2 * feature_dim)`` (sliced
        into ``chunk_size`` blocks) or a zero-arg callable returning a fresh
        iterator of such blocks.
    chunk_size : int or None
        Rows per block when ``data`` is an array. ``None`` yields the whole array
        as a single block (i.e. unchanged, full-batch behaviour). Ignored when
        ``data`` is a callable (the callable decides its own block sizes).
    """

    def __init__(self, data, chunk_size=None):
        self.is_callable = callable(data)
        self.data = data
        self.chunk_size = chunk_size
        # Known upfront only for in-memory arrays; counted on first sweep otherwise.
        self.n_samples = None if self.is_callable else int(data.shape[0])

    def __iter__(self):
        if self.is_callable:
            yield from self.data()
        elif self.chunk_size is None:
            yield self.data
        else:
            n = self.data.shape[0]
            for start in range(0, n, self.chunk_size):
                yield self.data[start:start + self.chunk_size]


def as_chunked(data, chunk_size=None):
    """Coerce ``data`` into a :class:`ChunkedData` (pass-through if already one)."""
    if isinstance(data, ChunkedData):
        return data
    return ChunkedData(data, chunk_size)


def subsample_for_init(source, backend, cap, seed=None):
    """Uniformly sample up to ``cap`` rows across ``source`` in a single pass.

    Uses the "smallest random key" reservoir variant: every row gets an i.i.d.
    key in ``[0, 1)`` and the ``cap`` rows with the smallest keys are kept. That
    is exactly uniform without a prior row count, and each chunk is reduced to at
    most ``cap`` rows immediately, so peak memory stays bounded by ``cap``.

    Returns a backend-native array of shape ``(min(cap, total), 2 * feature_dim)``
    ready to hand to :meth:`CovarianceModel.initialize`.
    """
    rng = np.random.default_rng(seed)
    buf = None          # host copy of the current best <= cap rows
    buf_keys = None
    for chunk in source:
        rows = backend.to_numpy(backend.asarray(chunk))
        keys = rng.random(rows.shape[0])
        if buf is None:
            buf, buf_keys = rows, keys
        else:
            buf = np.concatenate([buf, rows], axis=0)
            buf_keys = np.concatenate([buf_keys, keys], axis=0)
        if buf.shape[0] > cap:
            keep = np.argpartition(buf_keys, cap)[:cap]
            buf, buf_keys = buf[keep], buf_keys[keep]
    if buf is None:
        raise ValueError("Cannot initialize from an empty data source")
    return backend.asarray(buf)
