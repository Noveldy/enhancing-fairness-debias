"""Air Quality FSDG rerun for the journal revision, and the code figures Fig. 6, Fig. 15 and Fig. 16.

One command, from the repository root, in the locked environment (requirements-lock.txt, Python 3.12.3):

  Docker image built by `make docker-build` (the repository is mounted read-only; only this folder is writable):
    docker run --rm -v "$PWD:/repo:ro" \
        -v "$PWD/results/revision_airquality_fsdg:/repo/results/revision_airquality_fsdg" \
        -w /repo debias-revision /app/venv/bin/python results/revision_airquality_fsdg/run_airquality_fsdg.py
  Linux, with venv/ from `make env`:
    venv/bin/python results/revision_airquality_fsdg/run_airquality_fsdg.py

Why: the submitted Air Quality FSDG analysis (Section 3.3.4, the Air Quality FSDG table, the value-function figure)
was fitted on the raw file, with the -200 missing-value codes used as values and NMHC(GT) included. The rest of
the revision sets -200 to missing and drops NMHC(GT). This script repeats the analysis on the corrected data and
checks the diagnosis against the old stored value functions.

Steps (existing repository code is reused; nothing outside this folder is written):
 1. Data: experiments.ts_protocol.load_series (the time-series experiment's loader: data/loaders.load_air_quality,
    -200 -> NaN, NMHC(GT) dropped, 12 sensor columns, 9,357 hourly rows). The gaps are filled with the proposed
    time-series cleaning (experiments.ts_cleaning, method "proposed", parameters from configs/timeseries.yaml).
 2. Value functions: per column, intercept (mean of the cleaned series) + the 5 largest positive-frequency FFT
    components + Gaussian noise with SD = residual SD. Number of components, intercept and noise rule follow the
    original construction code (legacy/debias_ops/1st_Journal_2025/descriptive analysis.py,
    generate_fourier_lambdas_with_noise(top_n=5)). Phase: see DEVIATIONS in RESULTS.md.
 3. Each spec is validated (service.expressions.validate_spec) and stored (service.value_store.save_spec) in a
    LocalStore rooted at this folder (records in value_functions/), signed with a fixed demonstration HMAC key.
    The API's own loader (apis.refresh_value_functions) reads them back with integrity checks and registers the
    interpreted factories; TimeSeriesSyntheticGenerator draws the series (start, length and frequency of the real
    data), seeds 0-4. Nothing is passed to eval or exec.
 4. Fidelity against the real observed values only (imputed cells excluded): KS and Wasserstein per column,
    mean +- SD over seeds; real and synthetic min/max; share of negative synthetic values in the concentration
    columns.
 5. Figures (300 dpi, monospace, white): fig16 (stored records of CO(GT) and T), fig6 (stored record of the
    built-in flat_random function), fig15 (age and marital-status entries of the Adult schema as derived by the
    revised evaluation, experiments.synthetic.MarginalSampler, from the 45,222 complete records).
 6. RESULTS.md, the tables, a run record (versions, checksums, times) and run_log.txt are written here.
"""
import ast
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_ROOT = OUT.parents[1]
# Import from the repository root, not from this folder: its value_functions/ directory would otherwise be
# found by the generator's `from value_functions import ...`.
sys.path[0] = str(REPO_ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import fft as sfft  # noqa: E402
from scipy.stats import ks_2samp, wasserstein_distance  # noqa: E402

import apis  # noqa: E402  (refresh_value_functions: the API's path from the store to the generator's registry)
import bias_mitigation.synthetic_data.value_functions as vf  # noqa: E402
from bias_mitigation.synthetic_data.synthetic_data_generator import TimeSeriesSyntheticGenerator  # noqa: E402
from data import loaders  # noqa: E402
from experiments.synthetic import MarginalSampler  # noqa: E402
from experiments.ts_cleaning import fill_training_period  # noqa: E402
from experiments.ts_protocol import load_series  # noqa: E402
from service import value_store  # noqa: E402
from service.config import Settings  # noqa: E402
from service.expressions import make_factory, spec_from_ui_template, validate_spec  # noqa: E402
from service.limits import check_request  # noqa: E402
from service.storage import LocalStore  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TOP_N = 5                                    # legacy: generate_fourier_lambdas_with_noise(data, top_n=5)
TS_CONFIG = REPO_ROOT / "configs" / "timeseries.yaml"
LEGACY_VF = REPO_ROOT / "legacy" / "original" / "bias_mitigation" / "synthetic_data" / "value_functions.py"
LEGACY_RULE = REPO_ROOT / "legacy" / "debias_ops" / "1st_Journal_2025" / "descriptive analysis.py"
DESCRIPTION_CSV = REPO_ROOT / "results" / "summary" / "synthetic_data_description.csv"
# Fixed and published on purpose, so the stored records are reproducible byte for byte. In a deployment the
# key is a server secret (DEBIAS_STORE_HMAC_KEY); without a key the toolkit stores "hmac_sha256": null.
DEMO_HMAC_KEY = "revision-airquality-fsdg-demonstration-key"
CONCENTRATION = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]   # reference-analyser concentrations
UNITS = {"CO(GT)": "mg/m³", "PT08.S1(CO)": "sensor response", "C6H6(GT)": "µg/m³",
         "PT08.S2(NMHC)": "sensor response", "NOx(GT)": "ppb", "PT08.S3(NOx)": "sensor response",
         "NO2(GT)": "µg/m³", "PT08.S4(NO2)": "sensor response", "PT08.S5(O3)": "sensor response",
         "T": "°C", "RH": "%", "AH": "absolute humidity, unit not stated by UCI"}
FIG_FONT_PT = 7.0
DPI = 300

log = logging.getLogger("airquality_fsdg")
STAGES = {}


def setup_logging():
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(OUT / "run_log.txt", mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


class stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        log.info("== %s", self.name)

    def __exit__(self, *exc):
        STAGES[self.name] = round(time.perf_counter() - self.t0, 3)
        log.info("   %s: %.2f s", self.name, STAGES[self.name])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(col):
    """'PT08.S1(CO)' -> 'PT08_S1_CO' (the naming of the original auto_* functions)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_")


def wrap_phase(p):
    return (np.asarray(p) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------------------- Fourier model
def top_components(signal, top_n=TOP_N):
    """The legacy selection rule: the top_n positive-frequency FFT bins by magnitude, largest first."""
    n = len(signal)
    yf = sfft.fft(signal)
    xf = sfft.fftfreq(n, d=1)                # cycles per record (per hour)
    pos = np.flatnonzero(xf > 0)
    top = pos[np.argsort(np.abs(yf[pos]))[-top_n:][::-1]]
    return yf, top, 2.0 / n * np.abs(yf[top]), xf[top], np.angle(yf[top])


def fourier_values(i, a0, amps, freqs, phases):
    """Same operations, in the same order, as the declarative expression."""
    v = a0
    for a, f, p in zip(amps, freqs, phases):
        v = v + a * np.sin(2 * np.pi * f * i + p)
    return v


def expression(top_n=TOP_N):
    terms = [f"a{k} * np.sin(2 * np.pi * f{k} * i + p{k})" for k in range(1, top_n + 1)]
    return " + ".join(["a0", *terms, "np.random.normal(0, s)"])


def build_spec(col, a0, amps, freqs, phases, s):
    params = {"a0": float(a0)}
    for k, (a, f, p) in enumerate(zip(amps, freqs, phases), start=1):
        params.update({f"a{k}": float(a), f"f{k}": float(f), f"p{k}": float(p)})
    params["s"] = float(s)
    doc = (f"Fourier approximation of the Air Quality series {col} (cleaned; i = hours since 2004-03-10 18:00): "
           f"intercept, {len(amps)} largest FFT components and Gaussian noise with the residual SD.")
    return {"name": f"aq_{safe_name(col)}", "doc": doc, "params": params, "expression": expression(len(amps))}


# ---------------------------------------------------------------------------------------------- legacy check
def parse_legacy_functions(path):
    """{name: {intercept, terms[(amp, freq, phase) as 6-decimal strings], noise}} from the old value_functions.py."""
    src = path.read_text(encoding="utf-8")
    out = {}
    for name, expr in re.findall(r'@register_value_func\("(auto_\w+)"\)\s*\ndef \w+\(params=None\):\s*\n'
                                 r'\s*"""[\s\S]*?"""\s*\n\s*return lambda i: (.+)\n', src):
        terms = re.findall(r"(-?\d+\.\d+) \* np\.sin\(2 \* np\.pi \* (-?\d+\.\d+) \* i \+ (-?\d+\.\d+)\)", expr)
        noise = re.search(r"np\.random\.normal\(0, ([\d.]+)\)", expr)
        out[name] = {"intercept": re.match(r"\s*(-?\d+\.\d+)", expr).group(1), "terms": terms,
                     "noise": noise.group(1) if noise else None}
    return out


def legacy_check():
    """Re-apply the legacy rule (written out here: the legacy module cannot be imported, it needs gplearn and runs
    the household-power analysis at import) to the RAW file, -200 kept, NMHC(GT) included, and compare with the
    stored auto_* functions."""
    raw = loaders.read_raw("air_quality")          # checksum-verified raw file
    stored = parse_legacy_functions(LEGACY_VF)
    rows = []
    for col in [c for c in raw.columns if c not in ("Date", "Time")]:
        name = "auto_" + safe_name(col)
        sig = raw[col].astype(float).to_numpy()
        n = len(sig)
        _, _, amps, freqs, ph = top_components(sig)
        dc = sig.mean()
        # the legacy expression string: 6 decimals, np.sin with the FFT (cosine) phase
        s6 = lambda x: f"{x:.6f}"  # noqa: E731
        t = np.arange(n)
        base = float(s6(dc)) + sum(float(s6(a)) * np.sin(2 * np.pi * float(s6(f)) * t + float(s6(p)))
                                   for a, f, p in zip(amps, freqs, ph))
        st = stored.get(name)
        obs = sig[sig != loaders.AIR_QUALITY_MISSING_CODE]
        rows.append({
            "column": col, "legacy_function": name, "n_rows": n,
            "n_minus200": int((sig == loaders.AIR_QUALITY_MISSING_CODE).sum()),
            "raw_mean_incl_minus200": dc, "observed_mean": obs.mean(),
            "legacy_intercept_stored": float(st["intercept"]) if st else np.nan,
            "intercept_matches_6dp": bool(st) and s6(dc) == st["intercept"],
            "components_match_6dp": bool(st) and [tuple(map(s6, x)) for x in zip(amps, freqs, ph)] == st["terms"],
            "legacy_noise_sd_stored": float(st["noise"]) if st and st["noise"] else np.nan,
            "legacy_rule_residual_sd": float(np.std(sig - base)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- generation
def generate(schema, seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        out = TimeSeriesSyntheticGenerator(schema).generate(n_rows=1)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    return out


def closed_form(specs, columns, n, seed):
    """The series the expressions define, drawn with numpy directly: used to check the generator's output."""
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        i = np.arange(n)
        out = {}
        for col in columns:
            p = specs[col]["params"]
            k = range(1, TOP_N + 1)
            det = fourier_values(i, p["a0"], [p[f"a{j}"] for j in k], [p[f"f{j}"] for j in k],
                                 [p[f"p{j}"] for j in k])
            out[col] = det + np.array([np.random.normal(0, p["s"]) for _ in range(n)])
    finally:
        np.random.set_state(state)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- figures
def render(lines, path):
    """lines: [(text, bold)]; monospace text on white, 300 dpi, sized to the text."""
    matplotlib.rcParams["font.monospace"] = ["DejaVu Sans Mono"]
    char_w, line_h, margin = 0.6021 * FIG_FONT_PT / 72, 1.28 * FIG_FONT_PT / 72, 0.12
    width = max(len(t) for t, _ in lines) * char_w + 2 * margin
    height = len(lines) * line_h + 2 * margin
    fig = plt.figure(figsize=(width, height), dpi=DPI, facecolor="white")
    for k, (text, bold) in enumerate(lines):
        y = 1 - (margin + (k + 0.78) * line_h) / height
        fig.text(margin / width, y, text, family="monospace", fontsize=FIG_FONT_PT,
                 weight="bold" if bold else "normal", ha="left", va="baseline", color="black")
    fig.savefig(path, dpi=DPI, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def fmt4(x):
    return f"{x:.4f}"


def fmt_freq(x):
    return f"{x:.4g}"    # 4 significant digits: 4 decimals would leave 1 digit for the low frequencies


def wrapped_string(key, text, indent, width, sep=None):
    """'"key": "text"' over several lines (display only); sep=' + ' breaks between terms of an expression."""
    head = f'{" " * indent}"{key}": "'
    pad = " " * len(head)
    if sep:
        parts = text.split(sep)
        lines, cur = [], parts[0]
        for p in parts[1:]:
            if len(head) + len(cur) + len(sep) + len(p) + 2 > width:
                lines.append(cur)
                cur = "+ " + p
            else:
                cur += sep + p
        lines.append(cur)
    else:
        lines = textwrap.wrap(text, width=width - len(head) - 2)
    out = [head + lines[0]] + [pad + ln for ln in lines[1:]]
    out[-1] += '"'
    return out


def display_record(env, width=80):
    """A stored record, as in the file but with coefficients rounded and long strings wrapped."""
    spec, p = env["spec"], env["spec"]["params"]
    lines = ["{", f' "format": {env["format"]},', f' "kind": "{env["kind"]}",', ' "spec": {',
             f'  "name": "{spec["name"]}",']
    doc = wrapped_string("doc", spec["doc"], 2, width)
    doc[-1] += ","
    lines += doc + ['  "params": {', f'   "a0": {fmt4(p["a0"])},']
    for k in range(1, TOP_N + 1):
        lines.append(f'   "a{k}": {fmt4(p[f"a{k}"])}, "f{k}": {fmt_freq(p[f"f{k}"])}, "p{k}": {fmt4(p[f"p{k}"])},')
    lines += [f'   "s": {fmt4(p["s"])}', "  },"]
    lines += wrapped_string("expression", spec["expression"], 2, width, sep=" + ")
    lines += [" },", f' "sha256": "{env["sha256"]}",',
              f' "hmac_sha256": {json.dumps(env["hmac_sha256"])}', "}"]
    return lines


# ---------------------------------------------------------------------------------------------- environment
def lock_check():
    pins, bad = {}, []
    for line in (REPO_ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, ver = line.split("==", 1)
            pins[name] = ver
            try:
                have = metadata.version(name)
            except metadata.PackageNotFoundError:
                have = None
            if have != ver:
                bad.append({"package": name, "locked": ver, "installed": have})
    return pins, bad


def fmt_sig(x, sd=None, sig=3):
    """x (and sd) with about `sig` significant digits on x."""
    if x == 0 or not np.isfinite(x):
        d = 3
    else:
        d = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{d}f}" if sd is None else f"{x:.{d}f} ± {sd:.{d}f}"


def decimals_of(values):
    """Decimals the real values are recorded with (at most 4), used to print min/max."""
    s = pd.Series(values).map(lambda v: f"{v:.6f}".rstrip("0").partition(".")[2])
    return int(min(4, s.str.len().max()))


# ---------------------------------------------------------------------------------------------- main
def main():
    started = datetime.now(timezone.utc)
    t_all = time.perf_counter()
    setup_logging()
    log.info("Air Quality FSDG rerun; output folder %s", OUT)
    pins, lock_mismatch = lock_check()
    log.info("Python %s; %d locked packages, %d mismatches", platform.python_version(), len(pins), len(lock_mismatch))

    # ------------------------------------------------------------------ 1. data and cleaning
    with stage("load"):
        df = load_series()                       # the time-series experiment's loader
        cols = list(df.columns)
        raw_sha = sha256(loaders.raw_path("air_quality"))
        assert raw_sha == loaders.RAW_FILES["air_quality"]["sha256"]
        assert "NMHC(GT)" not in cols and len(cols) == 12, cols
        assert len(df) == 9357 and df.index.freqstr == "h" and df.index.is_monotonic_increasing
        assert not (df == loaders.AIR_QUALITY_MISSING_CODE).any().any(), "a -200 code is left"
        observed = df.notna()
        log.info("%d rows x %d columns, %s to %s; missing cells %d", len(df), len(cols), df.index[0], df.index[-1],
                 int((~observed).sum().sum()))

    with stage("clean"):
        ts_cfg = yaml.safe_load(TS_CONFIG.read_text(encoding="utf-8"))
        exp = ts_cfg["timeseries_experiments"][0]
        method_params = exp["method_params"]["proposed"]
        filled, clean_seconds, filler = fill_training_period(df, "proposed", method_params)
        assert not filled.isna().any().any()
        assert not (filled == loaders.AIR_QUALITY_MISSING_CODE).any().any(), "a -200 code is left after cleaning"
        assert "NMHC(GT)" not in filled.columns
        assert (filled[observed] == df[observed]).sum().sum() == observed.sum().sum(), "observed values changed"
        filler.donor_log_.to_csv(OUT / "cleaning_donor_log.csv", index=False)
        # A gap only partly filled in pass 1 is logged again in pass 2 for its remaining cells.
        routes = filler.donor_log_.groupby(["passes", "route"])["gap_length"].agg(["count", "sum"]).reset_index()
        log.info("proposed cleaning (%s): %.2f s; routes:\n%s", method_params, clean_seconds, routes.to_string(index=False))
        log.info("checks passed: no -200 value, no NMHC(GT) column, no missing value, observed values unchanged")

    # ------------------------------------------------------------------ 2. value functions
    with stage("construct value functions"):
        n = len(filled)
        i = np.arange(n)
        specs, summary = {}, []
        for col in cols:
            sig = filled[col].to_numpy(dtype=float)
            yf, top, amps, freqs, ph_fft = top_components(sig)
            # FFT term k is (2/n)|Y_k| cos(2 pi f_k i + angle Y_k) = (2/n)|Y_k| sin(2 pi f_k i + angle Y_k + pi/2)
            phases = wrap_phase(ph_fft + np.pi / 2)
            a0 = sig.mean()
            clean = fourier_values(i, a0, amps, freqs, phases)
            z = np.zeros(n, dtype=complex)
            z[0] = yf[0]
            z[top], z[n - top] = yf[top], yf[n - top]
            recon_err = float(np.max(np.abs(sfft.ifft(z).real - clean)))
            assert recon_err < 1e-9 * max(1.0, np.abs(sig).max()), (col, recon_err)
            s = float(np.std(sig - clean))                 # legacy: noise_std = np.std(residuals)
            s_old_phase = float(np.std(sig - fourier_values(i, a0, amps, freqs, ph_fft)))
            spec = validate_spec(build_spec(col, a0, amps, freqs, phases, s))
            specs[col] = spec
            obs = df[col].dropna()
            row = {"column": col, "function": spec["name"], "n_observed": int(observed[col].sum()),
                   "n_imputed": int((~observed[col]).sum()), "observed_mean": obs.mean(), "intercept_a0": a0,
                   "residual_sd_s": s, "r2_cleaned_series": 1 - s ** 2 / np.var(sig),
                   "residual_sd_with_legacy_phase": s_old_phase, "ifft_check_max_abs_error": recon_err}
            for k in range(TOP_N):
                row.update({f"a{k + 1}": amps[k], f"f{k + 1}": freqs[k], f"period_h{k + 1}": 1 / freqs[k],
                            f"p{k + 1}": phases[k]})
            summary.append(row)
            log.info("%-14s a0=%10.4f (observed mean %10.4f)  s=%9.4f  R2=%.3f  periods(h)=%s", col, a0, obs.mean(),
                     s, row["r2_cleaned_series"], ", ".join(f"{1 / f:.1f}" for f in freqs))
        vf_summary = pd.DataFrame(summary)
        vf_summary.to_csv(OUT / "value_function_summary.csv", index=False)

    # ------------------------------------------------------------------ 3. store, load back, generate
    with stage("store and load"):
        shutil.rmtree(OUT / "value_functions", ignore_errors=True)    # this script's own records from a rerun
        store = LocalStore(OUT)
        for col in cols:
            value_store.save_spec(store, specs[col], DEMO_HMAC_KEY, reserved=apis.BUILTIN_VALUE_FUNCS)
        stored_keys = store.list("value_functions/")
        assert len(stored_keys) == 12, stored_keys
        rejected = apis.refresh_value_functions(store, Settings(hmac_key=DEMO_HMAC_KEY))
        assert rejected == [], rejected
        registry = vf.VALUE_FUNC_FACTORY_REGISTRY
        for col in cols:
            fac = registry[specs[col]["name"]]
            assert getattr(fac, "_spec", None) == specs[col], "registry entry is not the interpreted stored spec"
        log.info("stored %d records (sha256 + HMAC); refresh_value_functions accepted all, rejected %d",
                 len(stored_keys), len(rejected))

    schema = {"record_id": {"type": "id", "prefix": "AQ"}}
    for col in cols:
        schema[col] = {"type": "timeseries", "entity_col": "record_id", "start": str(df.index[0]),
                       "periods": n, "freq": df.index.freqstr, "value_func": {"name": specs[col]["name"], "params": {}}}
    rows_requested = check_request(schema, 1, Settings().max_rows)
    (OUT / "airquality_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

    real = {col: df[col].dropna().to_numpy() for col in cols}
    per_seed, gen_times, max_check = [], {}, 0.0
    for seed in SEEDS:
        with stage(f"generate seed {seed}"):
            t0 = time.perf_counter()
            out = generate(schema, seed)
            gen_times[seed] = round(time.perf_counter() - t0, 2)
            synth = out.set_index("timestamp")[cols].apply(pd.to_numeric).astype(float)
            assert len(synth) == n and (synth.index == df.index).all() and not synth.isna().any().any()
            ref = closed_form(specs, cols, n, seed)
            err = float(np.max(np.abs(synth.to_numpy() - ref.to_numpy()) / (1 + np.abs(ref.to_numpy()))))
            max_check = max(max_check, err)
            assert err < 1e-9, f"generator output differs from the expressions (seed {seed}): {err}"
            for col in cols:
                x = synth[col].to_numpy()
                per_seed.append({"seed": seed, "column": col, "ks": ks_2samp(real[col], x).statistic,
                                 "wasserstein": wasserstein_distance(real[col], x), "synth_min": x.min(),
                                 "synth_max": x.max(), "synth_mean": x.mean(), "synth_sd": x.std(ddof=1),
                                 "share_negative": float((x < 0).mean())})
            log.info("seed %d: %d x %d generated in %.1f s; matches the expressions (max rel. diff %.1e)",
                     seed, *synth.shape, gen_times[seed], err)
    per_seed = pd.DataFrame(per_seed)
    per_seed.to_csv(OUT / "metrics_per_seed.csv", index=False)

    # ------------------------------------------------------------------ 4. table
    with stage("metrics and tables"):
        agg = per_seed.groupby("column", sort=False).agg(
            ks_mean=("ks", "mean"), ks_sd=("ks", "std"), wasserstein_mean=("wasserstein", "mean"),
            wasserstein_sd=("wasserstein", "std"), synth_min=("synth_min", "min"), synth_max=("synth_max", "max"),
            negative_mean=("share_negative", "mean"), negative_sd=("share_negative", "std")).reindex(cols)
        table = []
        for col in cols:
            a, r = agg.loc[col], real[col]
            d = decimals_of(r)
            table.append({
                "Feature": col, "KS (mean ± SD)": f"{a.ks_mean:.3f} ± {a.ks_sd:.3f}",
                "Wasserstein (mean ± SD)": fmt_sig(a.wasserstein_mean, a.wasserstein_sd),
                "real min-max": f"{r.min():.{d}f} – {r.max():.{d}f}",
                "synthetic min-max": f"{a.synth_min:.{d}f} – {a.synth_max:.{d}f}",
                "unit": UNITS[col], "n_real_observed": len(r), "n_synthetic_per_seed": n,
                "ks_mean": a.ks_mean, "ks_sd": a.ks_sd, "wasserstein_mean": a.wasserstein_mean,
                "wasserstein_sd": a.wasserstein_sd, "real_min": r.min(), "real_max": r.max(),
                "synth_min": a.synth_min, "synth_max": a.synth_max})
        table = pd.DataFrame(table)
        table.to_csv(OUT / "table25_airquality_fsdg.csv", index=False, encoding="utf-8")
        per_seed_mean_ks = per_seed.groupby("seed")["ks"].mean()
        mean_ks, mean_ks_sd = per_seed_mean_ks.mean(), per_seed_mean_ks.std()
        neg = pd.DataFrame({"column": CONCENTRATION,
                            "real_share_negative": [float((real[c] < 0).mean()) for c in CONCENTRATION],
                            "synthetic_share_negative_mean": agg.loc[CONCENTRATION, "negative_mean"].to_numpy(),
                            "synthetic_share_negative_sd": agg.loc[CONCENTRATION, "negative_sd"].to_numpy()})
        neg.to_csv(OUT / "negative_share_concentrations.csv", index=False)
        md_cols = ["Feature", "KS (mean ± SD)", "Wasserstein (mean ± SD)", "real min-max", "synthetic min-max"]
        md = ["| " + " | ".join(md_cols) + " |", "|" + "---|" * len(md_cols)]
        for _, r in table.iterrows():
            md.append("| " + " | ".join([f"{r['Feature']} ({r['unit']})"] + [str(r[c]) for c in md_cols[1:]]) + " |")
        md.append(f"| Mean over the 12 columns | {mean_ks:.3f} ± {mean_ks_sd:.3f} | – | – | – |")
        table_md = "\n".join(md)
        note = (f"KS and Wasserstein distance between each synthetic column (9,357 values per seed) and the real "
                f"observed values of that column (imputed cells excluded); mean ± SD over seeds {SEEDS}. Wasserstein "
                f"distances are in the column's unit and are not averaged across columns. Synthetic min-max over all "
                f"5 seeds.")
        (OUT / "table25_airquality_fsdg.md").write_text(
            "# Air Quality, FSDG Fourier value functions vs real observed data\n\n" + table_md + "\n\n" + note + "\n",
            encoding="utf-8")
        log.info("\n%s", table_md)
        log.info("mean KS over the 12 columns: %.4f ± %.4f (SD over seeds)", mean_ks, mean_ks_sd)
        for _, r in neg.iterrows():
            log.info("negative synthetic values, %s: %.2f%% ± %.2f%% (real: %.2f%%)", r.column,
                     100 * r.synthetic_share_negative_mean, 100 * r.synthetic_share_negative_sd,
                     100 * r.real_share_negative)

    # ------------------------------------------------------------------ 5. diagnosis of the old analysis
    with stage("legacy check"):
        legacy = legacy_check()
        legacy.to_csv(OUT / "legacy_check.csv", index=False)
        log.info("\n%s", legacy[["column", "n_minus200", "raw_mean_incl_minus200", "observed_mean",
                                 "legacy_intercept_stored", "intercept_matches_6dp", "components_match_6dp",
                                 "legacy_noise_sd_stored", "legacy_rule_residual_sd"]].to_string(index=False))

    # ------------------------------------------------------------------ 6. figures
    with stage("figures"):
        # Fig. 16: stored records of CO(GT) and T
        envs = {c: json.loads(store.get(f"value_functions/{specs[c]['name']}.json")) for c in ("CO(GT)", "T")}
        f16 = []
        for label, col in (("(a)", "CO(GT)"), ("(b)", "T")):
            if f16:
                f16.append(("", False))
            f16.append((f"{label} {col}: value_functions/{specs[col]['name']}.json", True))
            f16 += [(ln, False) for ln in display_record(envs[col])]
        render(f16, OUT / "fig16_value_functions_CO_T.png")
        (OUT / "fig16_value_functions_CO_T.txt").write_text("\n".join(t for t, _ in f16) + "\n", encoding="utf-8")

        # Fig. 6: the built-in flat_random function as the toolkit stores it
        src = Path(vf.__file__).read_text(encoding="utf-8")
        fn = next(x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == "flat_random_factory")
        code = "\n".join(src.splitlines()[fn.decorator_list[0].lineno - 1: fn.end_lineno])
        flat_spec = spec_from_ui_template(code)       # read without running it, as the API does with UI code
        for k in (0, 7, 1234):
            np.random.seed(k)
            a = apis.BUILTIN_VALUE_FUNCS["flat_random"]({})(k)
            np.random.seed(k)
            assert np.isclose(a, make_factory(flat_spec)({})(k), rtol=1e-12, atol=0)
        with tempfile.TemporaryDirectory(dir=OUT, prefix="tmp_store_") as tmp:
            tstore = LocalStore(tmp)
            value_store.save_spec(tstore, flat_spec, DEMO_HMAC_KEY)
            good, bad = value_store.load_specs(tstore, DEMO_HMAC_KEY)
            assert list(good) == ["flat_random"] and not bad
            flat_bytes = tstore.get("value_functions/flat_random.json")
        (OUT / "fig6_flat_random.json").write_bytes(flat_bytes)
        render([(ln, False) for ln in flat_bytes.decode("utf-8").splitlines()], OUT / "fig6_flat_random.png")

        # Fig. 15: Adult schema entries, derived as in the revised evaluation (Phase 5 MarginalSampler)
        adult = loaders.load_adult("baseline")          # 45,222 complete records
        schema_adult = MarginalSampler(target=loaders.INFO["adult"]["target"]).fit(adult).schema_
        age, ms = schema_adult["age"], schema_adult["marital-status"]
        desc = pd.read_csv(DESCRIPTION_CSV, float_precision="round_trip")
        d = desc[desc.experiment == "adult_synthetic"]
        dage = d[d.column == "age"].iloc[0]
        assert len(adult) == 45222 and np.allclose([age["mean"], age["std"], age["min"], age["max"]],
                                                   [dage["mean"], dage["sd"], dage["min"], dage["max"]], rtol=0, atol=1e-12)
        dms = d[d.column == "marital-status"].set_index("category")["share"]
        assert list(dms.index) == ms["categories"] and np.allclose(dms.to_numpy(), ms["weights"], rtol=0, atol=1e-12)
        full = {"age": age, "marital-status": ms}
        (OUT / "fig15_adult_schema.json").write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
        k = 3
        cats = ", ".join(json.dumps(c) for c in ms["categories"][:k])
        wts = ", ".join(f"{w:.3f}" for w in ms["weights"][:k])
        f15 = ["{", '  "age": {', f'    "type": "{age["type"]}",', f'    "distribution": "{age["distribution"]}",',
               f'    "mean": {age["mean"]:.2f},', f'    "std": {age["std"]:.2f},', f'    "min": {age["min"]:.0f},',
               f'    "max": {age["max"]:.0f},', f'    "clip": {json.dumps(age["clip"])}', "  },",
               '  "marital-status": {', f'    "type": "{ms["type"]}",', f'    "categories": [{cats}, ...],',
               f'    "weights": [{wts}, ...]', "  }", "}"]
        shown = json.loads("\n".join(f15).replace(", ...]", "]"))   # valid JSON once the '...' are removed
        assert shown["age"]["mean"] == 38.55 and shown["age"]["std"] == 13.22
        assert shown["age"]["min"] == 17 and shown["age"]["max"] == 90
        assert shown["marital-status"]["categories"] == ["Married-civ-spouse", "Never-married", "Divorced"]
        assert shown["marital-status"]["weights"] == [0.466, 0.323, 0.139]
        render([(ln, False) for ln in f15], OUT / "fig15_adult_schema.png")
        (OUT / "fig15_adult_schema.txt").write_text("\n".join(f15) + "\n", encoding="utf-8")
        log.info("figures written: fig16_value_functions_CO_T.png, fig6_flat_random.png, fig15_adult_schema.png")

    # ------------------------------------------------------------------ 7. sanity checks and records
    vsum = vf_summary.set_index("column")
    sanity = []
    for col, approx in (("CO(GT)", 2), ("T", 18)):
        r = vsum.loc[col]
        ok = abs(r.intercept_a0 - r.observed_mean) < 0.05 * abs(r.observed_mean)
        sanity.append((col, approx, r.intercept_a0, r.observed_mean, ok))
        log.info("SANITY %s intercept a0 = %.4f; observed mean %.4f (expected about %s): %s", col, r.intercept_a0,
                 r.observed_mean, approx, "ok" if ok else "NOT OK")
    wall = time.perf_counter() - t_all
    log.info("TOTAL run time %.1f s", wall)

    write_results(df, cols, observed, method_params, clean_seconds, routes, vf_summary, table_md, mean_ks, mean_ks_sd,
                  neg, agg, legacy, sanity, gen_times, max_check, rows_requested, pins, lock_mismatch, wall, envs)

    outputs = {p.relative_to(OUT).as_posix(): sha256(p) for p in sorted(OUT.rglob("*"))
               if p.is_file() and p.name not in ("run_record.json", "run_log.txt") and "__pycache__" not in p.parts}
    record = {
        "script": "results/revision_airquality_fsdg/run_airquality_fsdg.py", "script_sha256": sha256(__file__),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_clock_seconds": round(wall, 1), "stage_seconds": STAGES, "generation_seconds_per_seed": gen_times,
        "seeds": SEEDS, "top_n": TOP_N, "cleaning": {"method": "proposed", "params": method_params,
                                                     "source": "configs/timeseries.yaml", "seconds": round(clean_seconds, 3)},
        "hmac_key": {"value": DEMO_HMAC_KEY, "note": "fixed demonstration key, not a secret"},
        "datasets": {"air_quality.csv": raw_sha, "adult.csv": sha256(loaders.raw_path("adult"))},
        "inputs": {p: sha256(REPO_ROOT / p) for p in ["configs/timeseries.yaml", "requirements-lock.txt",
                                                      "legacy/original/bias_mitigation/synthetic_data/value_functions.py",
                                                      "results/summary/synthetic_data_description.csv"]},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cpu_count": os.cpu_count(), "locked_packages": len(pins),
                        "lock_mismatches": lock_mismatch,
                        "key_packages": {k: metadata.version(k) for k in ["numpy", "pandas", "scipy", "matplotlib",
                                                                         "PyYAML", "Flask", "rstr", "plotly"]},
                        "docker_image": os.environ.get("DEBIAS_IMAGE_ID"),
                        "image_git_commit": os.environ.get("DEBIAS_GIT_COMMIT"),
                        "host_git_commit": os.environ.get("DEBIAS_HOST_GIT_COMMIT"),
                        "host_git_status_porcelain": os.environ.get("DEBIAS_HOST_GIT_STATUS")},
        "outputs_sha256": outputs,
    }
    (OUT / "run_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("run record written")


def write_results(df, cols, observed, method_params, clean_seconds, routes, vfs, table_md, mean_ks, mean_ks_sd, neg,
                  agg, legacy, sanity, gen_times, max_check, rows_requested, pins, lock_mismatch, wall, envs):
    v = vfs.set_index("column")
    lg = legacy.set_index("column")
    n_noise = int((vfs["residual_sd_s"] > 0).sum())
    miss = (~observed).sum()
    fit_rows = ["| Column | observed | imputed | intercept a0 | observed mean | residual SD s | R² (5 terms) | "
                "periods of the 5 terms (h) |", "|---|---|---|---|---|---|---|---|"]
    for col in cols:
        r = v.loc[col]
        periods = ", ".join(f"{r[f'period_h{k}']:.1f}" for k in range(1, TOP_N + 1))
        fit_rows.append(f"| {col} | {r.n_observed} | {r.n_imputed} | {fmt_sig(r.intercept_a0, sig=4)} | "
                        f"{fmt_sig(r.observed_mean, sig=4)} | {fmt_sig(r.residual_sd_s, sig=4)} | "
                        f"{r.r2_cleaned_series:.3f} | {periods} |")
    neg_rows = ["| Column | synthetic values < 0 (mean ± SD over seeds) | real observed values < 0 |", "|---|---|---|"]
    neg_rows += [f"| {r.column} | {100 * r.synthetic_share_negative_mean:.1f}% ± {100 * r.synthetic_share_negative_sd:.1f}% "
                 f"| {100 * r.real_share_negative:.1f}% |" for _, r in neg.iterrows()]
    leg_rows = ["| Column | -200 cells | raw mean (with -200) | old stored intercept | match (6 dp) | "
                "old stored noise SD | legacy-rule residual SD on raw data | observed mean |",
                "|---|---|---|---|---|---|---|---|"]
    for col, r in lg.iterrows():
        noise = "none" if np.isnan(r.legacy_noise_sd_stored) else f"{r.legacy_noise_sd_stored:g}"
        leg_rows.append(f"| {col} | {r.n_minus200} | {r.raw_mean_incl_minus200:.6f} | {r.legacy_intercept_stored:.6f} "
                        f"| {'yes' if r.intercept_matches_6dp and r.components_match_6dp else 'NO'} | {noise} | "
                        f"{r.legacy_rule_residual_sd:.1f} | {r.observed_mean:.4f} |")
    all_match = bool((legacy.intercept_matches_6dp & legacy.components_match_6dp).all())
    noise_stored = legacy.dropna(subset=["legacy_noise_sd_stored"])
    floor_match = (np.floor(noise_stored.legacy_rule_residual_sd) == noise_stored.legacy_noise_sd_stored)
    noise_txt = (f"are the legacy rule's residual SDs on the raw data rounded down to whole numbers ({int(floor_match.sum())} "
                 f"of {len(noise_stored)} equal the residual SD's integer part)" if floor_match.all() else
                 f"are close to the legacy rule's residual SDs on the raw data (integer part equal in "
                 f"{int(floor_match.sum())} of {len(noise_stored)})")
    no_noise = ", ".join(lg.index[lg.legacy_noise_sd_stored.isna()])
    other = [c for c in cols if c not in CONCENTRATION and c != "T"]    # cannot be negative either
    other_txt = ", ".join(f"{c} {100 * agg.loc[c, 'negative_mean']:.2f}%" for c in other if agg.loc[c, "negative_mean"] > 0)
    co, t = lg.loc["CO(GT)"], lg.loc["T"]
    sanity_lines = [f"- {c}: intercept a0 = {a:.4f}; observed mean = {m:.4f} (expected about {e}) → "
                    f"{'ok' if ok else 'NOT OK'}" for c, e, a, m, ok in sanity]
    routes_txt = "; ".join(f"pass {r['passes']}, {r['route']}: {r['count']} gaps ({r['sum']} cells)"
                           for _, r in routes.iterrows())
    fallback = int(routes.loc[~routes.route.str.startswith("donor window"), "sum"].sum())
    fallback_txt = (f"All {int(miss.sum()):,} missing cells were filled from donor windows (no linear fallback)."
                    if fallback == 0 else f"{fallback:,} cells were filled by another route (see the donor log).")
    stage_txt = ", ".join(f"{k} {s:.1f} s" for k, s in STAGES.items())
    co_env = envs["CO(GT)"]
    text = f"""# Air Quality FSDG rerun (revision)

Generated by `run_airquality_fsdg.py` (do not edit by hand). Run on {datetime.now(timezone.utc):%Y-%m-%d} UTC;
log in `run_log.txt`, versions, checksums and times in `run_record.json`.

## What was run and how

Command (repository root; the repository mounted read-only, only this folder writable):

```
docker run --rm -v "$PWD:/repo:ro" -v "$PWD/results/revision_airquality_fsdg:/repo/results/revision_airquality_fsdg" \\
    -w /repo debias-revision /app/venv/bin/python results/revision_airquality_fsdg/run_airquality_fsdg.py
```

1. **Data.** `experiments.ts_protocol.load_series()`, the loader of the time-series experiment
   (`data/loaders.py:load_air_quality`: -200 → missing, NMHC(GT) dropped). {len(df):,} hourly rows,
   {df.index[0]} to {df.index[-1]}, {len(cols)} columns; {int(miss.sum()):,} missing cells
   ({", ".join(f"{c} {int(miss[c])}" for c in cols)}). Checked: no -200 value and no NMHC(GT) column before
   and after cleaning, no missing value after cleaning, observed values unchanged.
2. **Cleaning.** The proposed time-series cleaning (`experiments/ts_cleaning.py`, `fill_training_period(...,
   "proposed", {method_params})`, the settings of `configs/timeseries.yaml`), fitted on the whole series:
   {clean_seconds:.2f} s; {routes_txt}. A gap only partly filled in pass 1 (its donor window had missing
   cells) is logged again in pass 2 for its remaining cells, so the cell counts overlap. {fallback_txt}
   Donor log: `cleaning_donor_log.csv`.
3. **Value functions.** For each column: intercept a0 = mean of the cleaned series; the {TOP_N} positive-frequency
   FFT components with the largest magnitude (amplitude 2|Y_k|/n, frequency in cycles per hour = per record
   index i, phase); Gaussian noise with SD s = SD of the residual (cleaned series minus the {TOP_N}-term model).
   Expression (numbers in `params`): `{expression()}`.
   Each spec was checked with `service.expressions.validate_spec` and stored with `service.value_store.save_spec`
   in a `LocalStore` rooted at this folder (`value_functions/*.json`, SHA-256 and HMAC-SHA256 with a fixed
   demonstration key). `apis.refresh_value_functions` (the API's loader) read all 12 back, verified the
   integrity fields and registered the interpreted factories (rejected: none).
4. **Generation.** `TimeSeriesSyntheticGenerator` with the schema `airquality_schema.json` (one entity, 12
   time-series columns, start {df.index[0]}, {len(df):,} periods, frequency "h"; {rows_requested:,} generated
   values, within the API's `check_request` limit). Seeds {SEEDS} (`random` and `numpy` seeded per call,
   restored afterwards). Generation time per seed: {", ".join(f"{s}: {x:.1f} s" for s, x in gen_times.items())}.
   Check: every generated value equals the closed-form expression computed with numpy and the same seed
   (largest relative difference {max_check:.1e}), so the series are exactly what the stored expressions define.
5. **Metrics.** For every seed and column: `scipy.stats.ks_2samp` statistic and `scipy.stats.wasserstein_distance`
   between the 9,357 synthetic values and the real **observed** values (imputed cells excluded). Mean ± SD
   (ddof = 1) over the 5 seeds. Per seed: `metrics_per_seed.csv`.

## Environment

- Docker image `debias-revision`, built on this machine with `make docker-build`'s command (`git archive HEAD |
  docker build`) from `requirements-lock.txt`: Python {platform.python_version()}; all {len(pins)} pinned packages
  installed at the locked version ({"no mismatches" if not lock_mismatch else f"MISMATCHES: {lock_mismatch}"}).
  numpy {metadata.version("numpy")}, pandas {metadata.version("pandas")}, scipy {metadata.version("scipy")},
  matplotlib {metadata.version("matplotlib")}.
- Machine: {platform.platform()}, {os.cpu_count()} logical CPUs (Docker Desktop on Windows 11). The original
  revision runs used an Ubuntu machine (`results/hardware.json`); this run used no timing-sensitive comparison.
- Dataset checksums verified by the loaders (sha256 in `run_record.json`).

## Run time

Total {wall:.1f} s ({stage_txt}).

## Numbers for the paper

- **Value functions:** 12 (one per column; NMHC(GT) excluded). **With noise:** {n_noise} of 12 (all; the original
  construction code makes a noisy version of every column, with SD = residual SD).
- **Mean KS over the 12 columns:** {mean_ks:.3f} ± {mean_ks_sd:.3f} (mean over columns per seed; mean ± SD over
  the 5 seeds). Wasserstein distances are not averaged across columns (different units).

Per column (`table25_airquality_fsdg.csv` / `.md`):

{table_md}

Negative synthetic values in the concentration columns (`negative_share_concentrations.csv`):

{chr(10).join(neg_rows)}

The other columns that cannot be negative also receive negative synthetic values (mean share over seeds):
{other_txt or "none"}. Synthetic RH ranges beyond 0–100% (table above). The Gaussian noise and the sinusoids are
unbounded, and the value functions have no clipping.

Fit of the {TOP_N}-term models on the cleaned series (`value_function_summary.csv`; R² = 1 − s²/variance):

{chr(10).join(fit_rows)}

Sanity checks:

{chr(10).join(sanity_lines)}

## Diagnosis of the old analysis (for the response letter)

The old script and its outputs are in the repository: `legacy/debias_ops/1st_Journal_2025/descriptive analysis.py`
(construction), `legacy/debias_ops/1st_Journal_2025/timeseries_preprocessing.py` (its input file was the raw
`air_quality.csv` with only Date and Time merged) and the stored functions in
`legacy/original/bias_mitigation/synthetic_data/value_functions.py` (`auto_*`). Re-applying the old rule to the
raw file (-200 kept, NMHC(GT) included) reproduces {"every" if all_match else "NOT every"} stored intercept and
all {TOP_N} components of all 13 old functions to the 6 printed decimals (`legacy_check.csv`):

{chr(10).join(leg_rows)}

Suggested sentence: *The original Air Quality value functions were fitted to the raw file, with the -200
missing-value codes treated as measurements and NMHC(GT) included: each of the 13 stored intercepts equals the raw
column mean including the -200 codes (CO(GT) {co.raw_mean_incl_minus200:.2f}, T {t.raw_mean_incl_minus200:.2f}),
whereas the observed means are {co.observed_mean:.2f} mg/m³ and {t.observed_mean:.2f} °C.*

The old noise SDs (where present) {noise_txt}; the old functions for {no_noise} had no noise term. They are large
because the -200 blocks dominate the variance and because of the phase error described below.

## Deviations from the instructions, and notes

1. **Phase convention (deviation from the old code, not from the instructions).** The old code wrote each
   component as `amp * np.sin(2*pi*f*t + phase)` with the FFT's phase, which belongs to a cosine; every component
   was therefore shifted by a quarter period and was uncorrelated with the component it should reproduce. Here the
   phase is `angle(Y_k) + pi/2`, so `a*sin(...)` equals the FFT component; checked against the inverse FFT of the
   selected bins (largest error {vfs.ifft_check_max_abs_error.max():.1e}). With the old convention the residual SDs
   would be {", ".join(f"{c} {v.loc[c].residual_sd_with_legacy_phase:.3g}" for c in ("CO(GT)", "T"))} instead of
   {", ".join(f"{c} {v.loc[c].residual_sd_s:.3g}" for c in ("CO(GT)", "T"))}.
2. **Noise in every function.** The old construction code builds a noisy version of every column (SD = residual
   SD). The old stored file had noise in only 9 of its 13 functions, with the residual SD rounded down to an
   integer; no code in the repository makes that selection, so it is not repeated. All 12 functions here include
   noise with the unrounded residual SD.
3. **Cleaning fitted on the whole series.** In the forecasting experiment the proposed cleaning is fitted on each
   training period; here there is no held-out period, so it is fitted on, and fills, the whole series. Its
   settings are those of `configs/timeseries.yaml`. The FFT and the residual SD use the cleaned series (imputed
   cells included); the fidelity metrics use observed cells only.
4. **Fig. 16 rounding.** Amplitudes, phases, intercept and s are shown with 4 decimals. Frequencies are shown with
   4 significant digits (e.g. `{fmt_freq(co_env["spec"]["params"]["f1"])}`): with 4 decimals the low frequencies
   would keep a single digit. Line breaks inside the `doc` and `expression` strings are for display only. The
   sha256/HMAC shown are those of the full-precision stored records in `value_functions/`.
5. **HMAC key.** Records are signed with the fixed demonstration key `{DEMO_HMAC_KEY}`, so they reproduce byte for
   byte; in a deployment the key is a server secret (`DEBIAS_STORE_HMAC_KEY`). Without a key the toolkit writes
   `"hmac_sha256": null`.
6. **Fig. 6 (flat_random).** `flat_random` is a built-in function (`bias_mitigation/synthetic_data/value_functions.py`).
   Its source was converted with the toolkit's own `spec_from_ui_template` (the API's path for UI code, without
   running it), checked to give the same values as the built-in under the same seed, and stored with
   `value_store.save_spec` in a temporary store. Through the API, built-in names are reserved, so a user could
   store the same function only under another name; the stored format is identical. Its params are stored as
   written in the source (`0` and `10`).
7. **Fig. 15 (Adult schema).** Derived by `experiments.synthetic.MarginalSampler` (the revised evaluation's FSDG
   schema) from the 45,222 complete Adult records; checked against `results/summary/synthetic_data_description.csv`
   (Tables 17–18). The figure rounds mean/std to 2 and weights to 3 decimals and writes min/max as integers (the
   schema stores 17.0 and 90.0); `fig15_adult_schema.json` holds the full-precision entries (all 7 categories).
   The figure text is valid JSON once the `, ...` markers are removed (checked by the script).
8. **Generator call.** The schema is generated with `TimeSeriesSyntheticGenerator` directly. The API's
   `/generate-synthetic-data-api` route only picks that generator when a column is named `timeseries_activity`;
   otherwise it uses the tabular generator, which skips time-series columns. The old schema also started one hour
   early (2004-03-10 17:00); here the start is the real data's first timestamp.
9. **Table number.** The files are named `table25_*` as requested. In `docs/paper_reported_values.csv`
   (paper_v1.6), Table 25 is the AIF360/Fairlearn comparison; the numbering presumably follows the revised draft.
10. **Environment.** The locked `venv/` does not exist on this Windows machine, so the run used the Docker image
    built from the lock file (author's choice, 2026-10-27).
"""
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
