"""Round 2 for scipy/scipy#26265.

  probe   print CPU info; write `target=true` to $GITHUB_OUTPUT on AMD CPUs
          exposing AVX-512 (the ones that crashed in round 1)
  run     in the current pixi env: which OpenBLAS core is picked (with and
          without OPENBLAS_CORETYPE=Haswell) and whether the round-1 cases
          crash; optionally sweep dgemm shapes
"""
import argparse
import json
import os
import random
import subprocess
import sys

from repro import CASES, SETUP, cpu_info

CORE = r'''
import ctypes, glob, json, os, sys, sysconfig
import numpy, scipy.linalg
cands = []
lib_bin = os.path.join(sys.prefix, "Library", "bin")
for pat in ("libblas.dll", "libcblas.dll", "liblapack.dll", "openblas.dll",
            "libopenblas*.dll", "mkl_rt*.dll"):
    cands += glob.glob(os.path.join(lib_bin, pat))
sp = sysconfig.get_paths()["purelib"]
for d in ("numpy.libs", "scipy.libs"):
    cands += glob.glob(os.path.join(sp, d, "*openblas*.dll"))
out = {}
for path in cands:
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        out[os.path.basename(path)] = f"load error: {e}"
        continue
    res = {}
    for pre in ("openblas_", "scipy_openblas_"):
        for suf in ("", "64_"):
            for what in ("get_corename", "get_config"):
                f = getattr(lib, pre + what + suf, None)
                if f is not None:
                    f.restype = ctypes.c_char_p
                    res[what] = f().decode()
    out[os.path.relpath(path, sys.prefix)] = res or "no openblas symbols"
print("CORE_JSON:" + json.dumps(out))
'''

SWEEP = r'''
import json, sys
import numpy as np
from scipy.linalg.blas import dgemm
shapes = json.load(open(sys.argv[1]))
start, stop = int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(0)
F = np.asfortranarray
for i in range(start, stop):
    t, m, n, k = shapes[i]
    print(f"T {i}", flush=True)
    a = F(rng.standard_normal((k, m) if t[0] == "T" else (m, k)))
    b = F(rng.standard_normal((n, k) if t[1] == "T" else (k, n)))
    for _ in range(3):
        dgemm(1.0, a, b, trans_a=int(t[0] == "T"), trans_b=int(t[1] == "T"))
print("DONE", flush=True)
'''

VARIANTS = {
    "default": {"OPENBLAS_NUM_THREADS": "1"},
    # workaround from OpenBLAS#6021: the Zen 4/5 P/Q override needs l2 == 1 MiB
    "L2_2048": {"OPENBLAS_NUM_THREADS": "1", "OPENBLAS_L2_SIZE": "2048"},
    "L2_2048_allthreads": {"OPENBLAS_L2_SIZE": "2048"},
}


def child(args, variant_env, timeout=600):
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENBLAS")}
    env.update(variant_env)
    try:
        p = subprocess.run([sys.executable, "-X", "faulthandler", *args],
                           env=env, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "", ""
    return p.returncode, p.stdout, p.stderr


def status(rc):
    if rc is None:
        return "TIMEOUT"
    return "ok" if rc == 0 else f"CRASH 0x{rc & 0xFFFFFFFF:08X}"


def core_probe(variant_env):
    rc, out, err = child(["-c", CORE], {**variant_env, "OPENBLAS_VERBOSE": "2"})
    res = {"status": status(rc),
           "verbose": [l for l in err.splitlines() if "Core" in l][:10]}
    for line in out.splitlines():
        if line.startswith("CORE_JSON:"):
            res["libs"] = json.loads(line[len("CORE_JSON:"):])
    return res


def sweep(variant_env, max_crashes=120):
    shapes = [(t, m, n, k) for t in ("NN", "TN", "NT", "TT")
              for m in (1, 2, 3, 4, 8, 16, 64, 200)
              for n in (1, 2, 3, 4, 8, 16, 64, 200)
              for k in (1, 2, 8, 64, 300)]
    random.Random(0).shuffle(shapes)
    path = os.path.abspath("sweep_shapes.json")
    with open(path, "w") as f:
        json.dump(shapes, f)
    crashes, start = [], 0
    while start < len(shapes) and len(crashes) < max_crashes:
        rc, out, _ = child(["-c", SWEEP, path, str(start), str(len(shapes))],
                           variant_env)
        tried = [int(l.split()[1]) for l in out.splitlines() if l.startswith("T ")]
        if rc == 0:
            start = len(shapes)
            break
        if not tried:  # crashed before the first call
            return {"error": status(rc), "crashes": crashes}
        crashes.append([*shapes[tried[-1]], status(rc)])
        start = tried[-1] + 1
    # re-run the first crashing shapes alone, in fresh processes
    shape_idx = {tuple(s): i for i, s in enumerate(shapes)}
    retest = []
    for c in crashes[:15]:
        i = shape_idx[tuple(c[:4])]
        rc, _, _ = child(["-c", SWEEP, path, str(i), str(i + 1)], variant_env)
        retest.append([*c[:4], status(rc)])
    return {"n_shapes": len(shapes), "n_done": start, "crashes": crashes,
            "retest_alone": retest}


def cmd_probe(args):
    cpu = cpu_info()
    print(json.dumps(cpu, indent=1))
    target = cpu["vendor"] == "AuthenticAMD" and cpu["flags"]["avx512f"]
    print(f"target={str(target).lower()}")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"target={str(target).lower()}\n")


def cmd_run(args):
    import numpy
    result = {"label": args.label, "cpu": cpu_info(),
              "numpy": numpy.__version__, "variants": {}}
    try:
        import scipy
        result["scipy"] = scipy.__version__
    except ImportError:
        pass
    for vname, venv in VARIANTS.items():
        vres = {"core": core_probe(venv), "cases": {}}
        print(f"\n== [{args.label}] {vname}: {json.dumps(vres['core'])}",
              flush=True)
        for cname, code in CASES.items():
            rc, _, err = child(["-c", SETUP + code], venv)
            vres["cases"][cname] = status(rc)
            print(f"  {cname:30s} {status(rc)}", flush=True)
        if args.sweep:
            vres["sweep"] = sweep(venv)
            s = vres["sweep"]
            print(f"  sweep: {len(s['crashes'])} crashes in {s.get('n_done')}"
                  f"/{s.get('n_shapes')} shapes", flush=True)
            for c in s["crashes"][:40]:
                print("   ", c, flush=True)
            print("  retest alone:", s.get("retest_alone"), flush=True)
        result["variants"][vname] = vres
    with open(f"result-{args.label}-{args.copy}.json", "w") as f:
        json.dump(result, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("probe")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("run")
    p.add_argument("--label", required=True)
    p.add_argument("--copy", required=True)
    p.add_argument("--sweep", action="store_true")
    p.set_defaults(func=cmd_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
