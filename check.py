"""Run the BLAS calls that crash scipy's Windows CI (scipy/scipy#26265).

Each call runs in its own subprocess, so a crash is reported instead of
killing the script. Exits 1 if any call crashed.

    python check.py <label>
"""
import ctypes
import json
import os
import re
import subprocess
import sys

SETUP = """
import numpy as np
rng = np.random.default_rng(1234)
"""

# first tests to crash in the "fast, py3.12/npAny, spin" job
CASES = {
    "cluster.vq.kmeans (TestKMeans::test_large_features)": """
from scipy.cluster.vq import kmeans
d, n = 300, 100
m1, m2 = rng.standard_normal(d), rng.standard_normal(d)
x = 10000 * rng.standard_normal((n, d)) - 20000 * m1
y = 10000 * rng.standard_normal((n, d)) + 20000 * m2
kmeans(np.concatenate([x, y]), 2, rng=1)
""",
    "linalg.qr (TestQR::test_random_tall)": """
from scipy.linalg import qr
for _ in range(5):
    a = rng.random((200, 100))
    q, r = qr(a)
    q.T @ q
    q @ r
""",
    "linalg.lstsq (TestLstsq::test_random_exact, gelsd)": """
from scipy.linalg import lstsq
for _ in range(5):
    lstsq(rng.random((200, 200)), rng.random(200), lapack_driver="gelsd")
""",
    "numpy matmul (200x300 @ 300x2)": """
obs = rng.standard_normal((200, 300))
code_book = rng.standard_normal((2, 300))
obs @ code_book.T
""",
}


def openblas_config():
    import numpy  # noqa: F401  (loads openblas.dll and its dependencies)
    bin_dir = os.path.join(sys.prefix, "Library", "bin")
    try:
        os.add_dll_directory(bin_dir)
        lib = ctypes.CDLL(os.path.join(bin_dir, "openblas.dll"))
        lib.openblas_get_config.restype = ctypes.c_char_p
        return lib.openblas_get_config().decode()
    except (OSError, AttributeError) as e:
        return f"unknown ({e})"


def cpu():
    import cpuinfo
    info = cpuinfo.get_cpu_info()
    return {"brand": info.get("brand_raw"), "vendor": info.get("vendor_id_raw"),
            "family": info.get("family"), "model": info.get("model"),
            "avx512f": "avx512f" in info.get("flags", [])}


def main():
    label = sys.argv[1]
    result = {"label": label, "openblas": openblas_config(), "cpu": cpu(),
              "cases": {}}
    print(f"CPU: {result['cpu']['brand']} (AVX-512: {result['cpu']['avx512f']})")
    print(f"OpenBLAS: {result['openblas']}\n", flush=True)
    for name, code in CASES.items():
        p = subprocess.run([sys.executable, "-X", "faulthandler", "-c", SETUP + code],
                           capture_output=True, text=True)
        status = "ok" if p.returncode == 0 else f"CRASH 0x{p.returncode & 0xFFFFFFFF:08X}"
        result["cases"][name] = status
        print(f"{'PASS' if status == 'ok' else 'FAIL'}  {name}  [{status}]", flush=True)
        if p.returncode:
            print("      " + p.stderr.strip().splitlines()[0], flush=True)
    result["crashed"] = any(s != "ok" for s in result["cases"].values())
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", label).strip("-")
    with open(f"result-{slug}.json", "w") as f:
        json.dump(result, f, indent=1)
    sys.exit(1 if result["crashed"] else 0)


if __name__ == "__main__":
    main()
