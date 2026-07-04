"""MNIST loading helpers."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]

_MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = {
    "train-images": "train-images-idx3-ubyte.gz",
    "train-labels": "train-labels-idx1-ubyte.gz",
    "test-images": "t10k-images-idx3-ubyte.gz",
    "test-labels": "t10k-labels-idx1-ubyte.gz",
}


@dataclass(frozen=True)
class MnistSplits:
    """Canonical MNIST arrays split into deterministic train/validation/test subsets."""

    x_train: Array
    y_train: Labels
    x_val: Array
    y_val: Labels
    x_test: Array
    y_test: Labels


def _download_file(url: str, path: Path) -> None:
    """Download a single IDX gzip file with a browser-like user agent."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, path.open("wb") as handle:
        handle.write(response.read())


def _read_idx_gzip(path: Path) -> bytes:
    """Read and decompress an IDX gzip archive to raw big-endian bytes."""
    with gzip.open(path, "rb") as handle:
        return handle.read()


def _load_images(path: Path) -> Array:
    """Parse IDX image bytes into `[N, 784]` float64 rows normalized by 255."""
    raw = _read_idx_gzip(path)
    magic, num_images, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"unexpected image magic number {magic} in {path}")
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    expected = num_images * rows * cols
    if pixels.size != expected:
        raise ValueError(f"expected {expected} pixels, found {pixels.size} in {path}")
    images = pixels.reshape(num_images, rows * cols).astype(np.float64)
    return images / 255.0


def _load_labels(path: Path) -> Labels:
    """Parse IDX label bytes into a one-dimensional int64 class-index array."""
    raw = _read_idx_gzip(path)
    magic, num_labels = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"unexpected label magic number {magic} in {path}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if labels.size != num_labels:
        raise ValueError(f"expected {num_labels} labels, found {labels.size} in {path}")
    return labels.astype(np.int64)


def load_mnist(data_dir: str | Path = "data") -> MnistSplits:
    """Load MNIST as float64 features and int64 labels with the canonical 54k/6k split.

    The four gzip IDX files are cached in `data_dir`, the 60k training set is permuted
    with seed 42, and the first 54k rows become training data while the last 6k rows
    become validation data. Pixel intensities are flattened row-major and normalized by
    dividing by 255.0.
    """
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: base_dir / filename for name, filename in _MNIST_FILES.items()}
    for filename in _MNIST_FILES.values():
        _download_file(_MNIST_BASE_URL + filename, base_dir / filename)

    x_train_all = _load_images(paths["train-images"])
    y_train_all = _load_labels(paths["train-labels"])
    x_test = _load_images(paths["test-images"])
    y_test = _load_labels(paths["test-labels"])

    if x_train_all.shape != (60000, 784) or y_train_all.shape != (60000,):
        raise ValueError("training MNIST data has unexpected shape")
    if x_test.shape != (10000, 784) or y_test.shape != (10000,):
        raise ValueError("test MNIST data has unexpected shape")

    rng = np.random.default_rng(42)
    perm = rng.permutation(60000)
    train_idx = perm[:54000]
    val_idx = perm[54000:]

    x_train = x_train_all[train_idx]
    y_train = y_train_all[train_idx]
    x_val = x_train_all[val_idx]
    y_val = y_train_all[val_idx]

    return MnistSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
