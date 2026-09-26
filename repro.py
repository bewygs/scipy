"""Reproducer for scipy/scipy#26265: access violations in BLAS on Windows CI.

Each case runs in its own subprocess so that a crash is recorded instead of
killing the run. Every case is repeated under several OpenBLAS settings
(thread count, forced kernel via OPENBLAS_CORETYPE).
"""
import argparse
import json
import os
import platform
import subprocess
import sys

SETUP = """
import numpy as np
rng = np.random.default_rng(1234)
"""

INFO = """
import json
import numpy, scipy, scipy.linalg
from threadpoolctl import threadpool_info
print("INFO_JSON:" + json.dumps(threadpool_info()))
"""

CASES = {
    # shape of the dgemm("T", "N", ncodes, nobs, nfeat) call in
    # scipy/cluster/vq/_vq.pyx for TestKMeans::test_large_features
    "np_matmul_vq_shape": """
code_book = rng.standard_normal((2, 300))
obs = rng.standard_normal((200, 300)) * 1e4
for _ in range(200):
    code_book @ obs.T
    obs @ code_book.T
c, o = code_book.astype(np.float32), obs.astype(np.float32)
for _ in range(200):
    c @ o.T
""",
    # TestQR::test_random_tall crashed on `q.T @ q`
    "np_matmul_qr_shape": """
a = rng.random((200, 100))
q = rng.random((200, 200))
for _ in range(50):
    q.T @ q
    a.T @ a
""",
    "np_linalg": """
for _ in range(10):
    a = rng.random((200, 100))
    np.linalg.qr(a)
    np.linalg.svd(a)
    np.linalg.lstsq(rng.random((200, 200)), rng.random(200), rcond=None)
""",
    "scipy_dgemm_vq": """
from scipy.linalg.blas import dgemm, sgemm
code_book = rng.standard_normal((2, 300))
obs = rng.standard_normal((200, 300)) * 1e4
c, o = code_book.astype(np.float32), obs.astype(np.float32)
for _ in range(200):
    # F-contiguous views + trans_a=1 -> dgemm("T", "N", 2, 200, 300)
    dgemm(1.0, code_book.T, obs.T, trans_a=1)
    sgemm(1.0, c.T, o.T, trans_a=1)
""",
    "scipy_kmeans_large_features": """
from scipy.cluster.vq import kmeans
for seed in range(5):
    r = np.random.default_rng(seed)
    d, n = 300, 100
    m1 = r.standard_normal(d)
    m2 = r.standard_normal(d)
    x = 10000 * r.standard_normal((n, d)) - 20000 * m1
    y = 10000 * r.standard_normal((n, d)) + 20000 * m2
    kmeans(np.concatenate([x, y]), 2, rng=1)
""",
    "scipy_qr_random_tall": """
from scipy.linalg import qr
for _ in range(5):
    a = rng.random((200, 100))
    q, r = qr(a)
    q.T @ q
    q @ r
""",
    "scipy_lstsq_gelsd": """
from scipy.linalg import lstsq
for _ in range(5):
    lstsq(rng.random((200, 200)), rng.random(200), lapack_driver="gelsd")
""",
}

VARIANTS = {
    "default": {"OPENBLAS_NUM_THREADS": "1"},
    "allthreads": {},
    "Haswell": {"OPENBLAS_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Haswell"},
    "SkylakeX": {"OPENBLAS_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "SkylakeX"},
    "Cooperlake": {"OPENBLAS_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Cooperlake"},
    "SapphireRapids": {"OPENBLAS_NUM_THREADS": "1",
                       "OPENBLAS_CORETYPE": "SapphireRapids"},
    "Zen": {"OPENBLAS_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Zen"},
}

CPU_FLAGS = ("avx2", "fma", "avx512f", "avx512dq", "avx512cd", "avx512bw",
             "avx512vl", "avx512_bf16", "avx512vnni", "avx512_fp16",
             "avx512ifma", "avx512vbmi", "amx_tile", "amx_bf16", "amx_int8",
             "avxvnni")


def run(code, variant_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENBLAS")}
    env.update(variant_env)
    try:
        p = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code],
                           env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "rc": None, "tail": ""}
    rc = p.returncode
    status = "ok" if rc == 0 else f"CRASH 0x{rc & 0xFFFFFFFF:08X}"
    tail = "\n".join((p.stderr or "").strip().splitlines()[:20])
    out = {"status": status, "rc": rc, "tail": tail if rc else ""}
    for line in p.stdout.splitlines():
        if line.startswith("INFO_JSON:"):
            out["info"] = json.loads(line[len("INFO_JSON:"):])
    return out


def cpu_info():
    import cpuinfo
    info = cpuinfo.get_cpu_info()
    flags = set(info.get("flags", []))
    return {
        "brand": info.get("brand_raw"),
        "vendor": info.get("vendor_id_raw"),
        "family": info.get("family"),
        "model": info.get("model"),
        "stepping": info.get("stepping"),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "flags": {f: f in flags for f in CPU_FLAGS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy
    import scipy
    result = {"cpu": cpu_info(), "numpy": numpy.__version__,
              "scipy": scipy.__version__, "variants": {}}
    print(json.dumps(result["cpu"], indent=1), flush=True)

    for vname, venv in VARIANTS.items():
        vres = {"info": run(INFO, venv), "cases": {}}
        blas = [(d.get("internal_api"), d.get("version"), d.get("architecture"),
                 d.get("filepath")) for d in vres["info"].get("info", [])]
        print(f"\n== {vname} {venv} -> {vres['info']['status']} {blas}",
              flush=True)
        for cname, code in CASES.items():
            r = run(SETUP + code, venv)
            vres["cases"][cname] = r
            print(f"  {cname:30s} {r['status']}", flush=True)
            if r["tail"]:
                print("    " + r["tail"].replace("\n", "\n    "), flush=True)
        result["variants"][vname] = vres

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    cpu = result["cpu"]
    default = result["variants"]["default"]
    arch = sorted({d.get("architecture") for d in default["info"].get("info", [])
                   if d.get("internal_api") == "openblas"} - {None})
    crashed = [c for c, r in default["cases"].items() if r["status"] != "ok"]
    summary = (f"| {cpu['brand']} | fam {cpu['family']} model {cpu['model']} "
               f"| avx512f={cpu['flags']['avx512f']} | {arch} "
               f"| {', '.join(crashed) or 'no crash'} |")
    print("\nSUMMARY " + summary)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("| CPU | id | AVX-512 | OpenBLAS core | crashes (default) |\n"
                    "|---|---|---|---|---|\n" + summary + "\n")
    sys.exit(1 if crashed else 0)


if __name__ == "__main__":
    main()
