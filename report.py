"""Summarise check.py results from all runners into one Markdown table."""
import collections
import glob
import json
import os
import sys

VERSIONS = {
    "0.3.30": "0.3.30",
    "0.3.33": "0.3.33",
    "0.3.34-lock": "0.3.34 (scipy's pixi.lock)",
    "0.3.34-latest": "0.3.34 (latest conda-forge build)",
}


def cpu_name(c):
    if c["vendor"] == "AuthenticAMD":
        gen = {(25, 1): "Zen 3", (25, 17): "Zen 4"}.get((c["family"], c["model"]))
        gen = gen or ("Zen 5" if c["family"] == 26 else None)
        return f"{c['brand']} ({gen})" if gen else c["brand"]
    return c["brand"]


runners = collections.defaultdict(dict)
for path in glob.glob(os.path.join(sys.argv[1], "*", "result-*.json")):
    r = json.load(open(path))
    runners[os.path.basename(os.path.dirname(path))][r["label"]] = r

rows = collections.Counter()
for results in runners.values():
    c = next(iter(results.values()))["cpu"]
    cells = tuple("–" if v not in results else ("❌" if results[v]["crashed"] else "✅")
                  for v in VERSIONS)
    rows[(cpu_name(c), "yes" if c["avx512f"] else "no", cells)] += 1

lines = [
    "## conda-forge OpenBLAS on `windows-2025` runners (scipy/scipy#26265)",
    "",
    "Same runner, same numpy 2.3.5 / scipy 1.18.1, only `libopenblas` changes. "
    "✅ all BLAS calls passed, ❌ at least one crashed (access violation).",
    "",
    "| Runner CPU | AVX-512 | Runners | " + " | ".join(VERSIONS.values()) + " |",
    "|---|---|---|" + "---|" * len(VERSIONS),
]
for (name, avx, cells), n in sorted(rows.items(), key=lambda kv: ("❌" not in kv[0][2], kv[0][0])):
    lines.append(f"| {name} | {avx} | {n} | " + " | ".join(cells) + " |")
out = "\n".join(lines) + "\n"
print(out)
if "GITHUB_STEP_SUMMARY" in os.environ:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(out)
