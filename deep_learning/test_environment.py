"""Verify that the notebook environment can run TensorFlow computations."""

import sys
from importlib.metadata import PackageNotFoundError, version


def main() -> None:
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Python 3.10 through 3.13 is required.")

    try:
        import matplotlib
        import numpy as np
        import tensorflow as tf
        version("ipykernel")
        version("jupyterlab")
    except (ImportError, PackageNotFoundError):
        raise SystemExit(
            "Install the project dependencies with: python -m pip install -r requirements.txt"
        ) from None

    result = tf.reduce_sum(tf.constant(np.array([1.0, 2.0], dtype=np.float32)))
    if float(result.numpy()) != 3.0:
        raise RuntimeError("TensorFlow computation failed.")

    print(f"Python {sys.version.split()[0]}")
    print(f"TensorFlow {tf.__version__}")
    print(f"NumPy {np.__version__}")
    print(f"Matplotlib {matplotlib.__version__}")
    print("Environment check passed.")


if __name__ == "__main__":
    main()
