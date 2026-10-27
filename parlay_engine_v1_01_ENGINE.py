#!/usr/bin/env python3
"""parlay_engine_v1 - 01_ENGINE.py - the complete walk-forward football engine.

A five-file QWEN Projects kit. This single file is the
whole engine: 21 pre-trained market models + the live 1X2 ensemble soft layer
+ the soft-FACTOR layer (a 31-module LLM-compiled Adjustment Register:
fail-closed verification, measured caps, cross-market coherence) +
Asian-handicap per-line settlement + the embedded CSV -> 135-feature
pipeline + TreeSHAP XAI summaries + a human-readable report written next to
the JSON output. Probability and confidence only - no EV, no staking.

  markets   1X2 (champion CatBoost + 5 soft-layer variants) - O/U 2.5, 1.5,
            3.5 - team totals over 0.5 / 1.5 per side - BTTS - double chance
            1X / X2 / 12 - draw-no-bet (both directions) - win-to-nil home /
            away - corners O/U 9.5 - cards O/U 3.5 - HT 1X2 - HT/FT (9) -
            correct score (top-20 + other) - Asian handicap margin model
  soft      (1) the 1X2 ensemble: vote-GBM / 5-3-2 blend / tuned blend
            (walk-forward OOF weights) / stack->LR meta / isotonic x2
            (2) the soft-FACTOR layer: a 31-module Adjustment Register with
            fail-closed source verification, a measured RULE 10 cap ladder,
            RULE 20 priced-in detection, RULE 22 cross-market coherence,
            RULE 23 model-overlap discounts and RULE 24 syndication dedup,
            applied with full-delta delivery + overflow guards
  picks     a FINAL PICK on EVERY modeled market (21 of them) on the
            calibrated + soft-adjusted probabilities. Each pick carries its
            own headline gate (0.60 for the decision markets; the
            walk-forward base-rate gate for the long-shot multi-class
            markets - HT/FT 0.25, correct score 0.12) and a HIGH / MEDIUM /
            LOW confidence rating next to its probability. No odds-based
            value, no staking.
  intuition an AI-INTUITION block per match: the audited register narrative
            digest (drivers, priced-in, rejections) + a reserved pick slot
            the running AI completes with its own reasoning-based lean -
            deliberately NOT derived from the model probabilities
  xai       every pick carries an exact TreeSHAP decomposition (top model
            drivers + soft-factor delta decomposition + walk-forward
            evidence) in a plain-language summary
  inputs    --features-template writes a single-match fixture CSV template;
            --features-csv (+ optional --history-csv) prices fixtures
            without any master training CSVs

Everything was validated under a strict walk-forward protocol (8 annual
expanding-window folds, test 2019-2026, 96,222 matches, seed 42). The
numbers embedded below are out-of-sample. Models are fitted on every match
available; use them for fixtures AFTER the training cutoff.

KIT (5 files, each <= 20 MB, PLAIN FILES ONLY - no zip archives, QWEN
Projects would not accept them):
  00_README.md                     operating guide + environment setup +
                                   soft-factor module catalog + register spec
  01_ENGINE.py                     this file (also: --verify-env)
  02_MODELS_CHAMPION_SOFTLAYER.py  self-extracting bundle: 1X2 champion +
                                   the complete soft layer + manifests
  03_MODELS_19_MARKETS.py          self-extracting bundle: the 18 CatBoost
                                   market models + the HT/FT LightGBM text
  04_MODELS_CORRECT_SCORE.py       self-extracting bundle: the correct-score
                                   model (its own 20 MB file)

The three bundles are ordinary Python files (base85-wrapped xz payloads,
stdlib-only): the first run unpacks models_v1/ - byte-identical,
sha256-verified, idempotent. Run a bundle directly or let this engine do
it (it auto-extracts whatever *MODELS*.py bundles sit next to it).

QUICK START
  python 01_ENGINE.py --verify-env    # environment check (+ requirements.txt)
  python 01_ENGINE.py --sanity        # staged self-check of the whole kit
  python 01_ENGINE.py --demo          # synthetic match on every market, with
                                      #   a sample register -> soft factors,
                                      #   final picks + XAI + report text
  python 01_ENGINE.py --selftest      # adversarial unit checks

  # SINGLE MATCH (no master CSVs needed):
  python 01_ENGINE.py --features-template          # writes features_template.csv
  #   ... fill one row per fixture (empty cells are OK - missing values are
  #   tolerated; fill odds + elo + form at minimum), then:
  python 01_ENGINE.py --features-csv features_template.csv --out report.json
  #   optionally add --history-csv recent_matches.csv (last ~10 matches per
  #   team, same columns + FTR + goals) to activate the rolling-history
  #   features (form10 / venue / h2h / xg5).

  # price a master CSV the historical way: put master CSVs in ./upload/
  python 01_ENGINE.py --after 2026-10-16 --min-conf 0.60 --out report.json

  # verify a report is a genuine, unmodified ENGINE.py output (anti-fake):
  python 01_ENGINE.py --check report.json

  # with an LLM-compiled Adjustment Register (see 00_README PART S;
  # --register-template writes a schema-correct skeleton):
  python 01_ENGINE.py --features-csv features_template.csv \
      --soft-register register.json --odds-as-of 2026-10-27T15:00:00Z \
      --out report.json

The engine NEVER silently substitutes a failed model: every market returns
either its model probabilities or an explicit FAIL entry with the reason.
Dependencies: numpy, pandas, catboost, lightgbm, xgboost (+ scikit-learn
1.7.2 for the soft layer's sklearn artifacts - see --verify-env).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
SEED = 42
CLASSES = ["A", "D", "H"]           # class order everywhere in this engine

# --- headline confidence gates (walk-forward evidence, all out-of-sample) ---
# label a pick STRONG when its probability >= the market's headline gate;
# below it the pick stays fully computable and is labeled ADVISORY. Every
# market id (including dnb) carries its own key here.
GATES = {
    "x12": 0.60, "ou25": 0.60, "btts": 0.60, "ou15": 0.60, "ou35": 0.60,
    "tt_h05": 0.60, "tt_h15": 0.60, "tt_a05": 0.60, "tt_a15": 0.60,
    "dc1x": 0.60, "dcx2": 0.60, "dc12": 0.60,
    "dnb": 0.60, "wtn_h": 0.60, "wtn_a": 0.60,
    "corners": 0.60, "cards": 0.60, "ht13": 0.60, "htft": 0.25, "cs": 0.12,
    "ah": 0.60,
}

SOFT_CANDS = ["book", "lr_b", "lgbm_b", "xgb_b", "cat_b", "cat_tuned"]
STACK_BASES = ["lr_b", "lgbm_b", "xgb_b", "cat_b"]   # 4 bases x 3 classes = 12 cols

# --- final picks ------------------------------------------------------------
# EVERY modeled market gets a FINAL PICK: probability + its own headline gate
# + a HIGH/MEDIUM/LOW confidence rating (odds are shown as reference when
# available, never used to gate anything). The gate is per-market by design:
# 0.60 for the decision markets, and the walk-forward base-rate gate for the
# long-shot multi-class markets (htft 0.25, cs 0.12 - a 0.60 gate there could
# never be met and would be a dishonest bar). Display order below.
PICK_MARKETS = (
    "x12", "ah", "ou25", "btts", "ou15", "ou35",
    "tt_h05", "tt_h15", "tt_a05", "tt_a15",
    "dc1x", "dcx2", "dc12", "dnb", "wtn_h", "wtn_a",
    "corners", "cards", "ht13", "htft", "cs",
)
# markets whose only real bookmaker side is YES/WIN (the "no" side is not a
# standalone market): the pick always sits on the yes/win side and is shown
# honestly even when its probability is below 0.50 (LOW / ADVISORY).
YES_ONLY_MARKETS = {"dc1x", "dcx2", "dc12", "wtn_h", "wtn_a"}
# long-shot multi-class markets: model-only basis (the soft-factor catalog
# carries no targets for them - shown honestly on the pick)
MODEL_ONLY_MARKETS = {"htft", "cs"}
CONF_MEDIUM = 0.50     # MEDIUM from 50% up; HIGH from the market's own
                       #   headline gate up; LOW below 50%
CACHE_VERSION = 2      # bump when the payload schema changes (odds surfacing)

# human-readable pick labels per market: (negative_side, positive_side) for
# binary markets; the class strings for 3-way / multi-class are mapped below
_BINARY_PICK_LABELS = {
    "ou15": ("UNDER 1.5 GOALS", "OVER 1.5 GOALS"),
    "ou25": ("UNDER 2.5 GOALS", "OVER 2.5 GOALS"),
    "ou35": ("UNDER 3.5 GOALS", "OVER 3.5 GOALS"),
    "tt_h05": ("HOME TEAM UNDER 0.5", "HOME TEAM OVER 0.5"),
    "tt_h15": ("HOME TEAM UNDER 1.5", "HOME TEAM OVER 1.5"),
    "tt_a05": ("AWAY TEAM UNDER 0.5", "AWAY TEAM OVER 0.5"),
    "tt_a15": ("AWAY TEAM UNDER 1.5", "AWAY TEAM OVER 1.5"),
    "btts": ("BTTS NO", "BTTS YES"),
    "corners": ("UNDER 9.5 CORNERS", "OVER 9.5 CORNERS"),
    "cards": ("UNDER 3.5 CARDS", "OVER 3.5 CARDS"),
    "dc1x": ("DC 1X - NO", "DC 1X (HOME OR DRAW)"),
    "dcx2": ("DC X2 - NO", "DC X2 (DRAW OR AWAY)"),
    "dc12": ("DC 12 - NO", "DC 12 (HOME OR AWAY)"),
    "wtn_h": ("HOME WIN TO NIL - NO", "HOME WIN TO NIL"),
    "wtn_a": ("AWAY WIN TO NIL - NO", "AWAY WIN TO NIL"),
}
_HTFT_PICK_LABELS = {
    "HH": "HT/FT HOME-HOME", "HD": "HT/FT HOME-DRAW", "HA": "HT/FT HOME-AWAY",
    "DH": "HT/FT DRAW-HOME", "DD": "HT/FT DRAW-DRAW", "DA": "HT/FT DRAW-AWAY",
    "AH": "HT/FT AWAY-HOME", "AD": "HT/FT AWAY-DRAW", "AA": "HT/FT AWAY-AWAY",
}
_3W_PICK_LABELS = {"H": "HOME", "D": "DRAW", "A": "AWAY"}


def pick_label_for(mid, classes, ci):
    """Human-readable pick label for market `mid` picking classes[ci]."""
    cls = str(classes[ci])
    if mid in _BINARY_PICK_LABELS:
        return _BINARY_PICK_LABELS[mid][ci]
    if mid == "x12":
        return _3W_PICK_LABELS.get(cls, cls)
    if mid == "ht13":
        return "HT " + _3W_PICK_LABELS.get(cls, cls)
    if mid == "dnb":
        return "DNB " + ("HOME" if cls == "home" else "AWAY")
    if mid == "htft":
        return _HTFT_PICK_LABELS.get(cls, f"HT/FT {cls}")
    if mid == "cs":
        return f"CORRECT SCORE {cls}"
    return cls

# probability source for FINAL picks, in fallback order (best-calibrated
# first; all document their own OOS log loss in kit_evidence.json)
FINAL_X12_PRIORITY = ("blend_wf_iso", "cat_b_iso", "blend_wf", "cat_b")

NUM = lambda s: pd.to_numeric(s, errors="coerce")     # noqa: E731

# ------------------------------------------------------------------ kit paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models_v1"
UPLOAD_DIR = ROOT / "upload"
CACHE_DIR = ROOT / ".cache"
STATE_FILE = MODELS_DIR / ".extracted.json"


def find_bundles() -> list[Path]:
    """The self-extracting model bundles shipped as plain .py files."""
    me = Path(__file__).name
    return [p for p in sorted(ROOT.glob("*MODELS*.py")) if p.name != me]


_BUNDLE_CACHE: dict = {}


def _bundle_module(path: Path):
    """Import a bundle .py (defines FILES + extract) without extracting."""
    key = str(path)
    if key not in _BUNDLE_CACHE:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BUNDLE_CACHE[key] = mod
    return _BUNDLE_CACHE[key]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(bundles: list[Path]) -> None:
    """models_v1/metadata.json - the merged per-artifact manifest."""
    files = {}
    for b in bundles:
        try:
            for rel, rec in getattr(_bundle_module(b), "FILES", {}).items():
                files[rel] = {"bytes": rec[1], "sha256": rec[2]}
        except Exception:                                     # noqa: BLE001
            continue
    (MODELS_DIR / "metadata.json").write_text(json.dumps(
        {"kit": "parlay_engine_v1", "engine_version": "1.2.0",
         "models_version": "models_v1", "n_bundles": len(bundles),
         "note": ("artifacts ship inside self-extracting .py bundles "
                  "(base85 xz, per-artifact sha256)"),
         "files": files}, indent=1))


def ensure_extracted(verbose: bool = True) -> bool:
    """Run the self-extracting bundles into models_v1/ (idempotent).

    Each bundle is a plain .py file carrying base85-wrapped xz payloads
    with per-artifact sha256; extraction is byte-identical and re-runs
    only when a bundle changes or its files go missing.
    """
    bundles = find_bundles()
    if not bundles:
        return False
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:                                     # noqa: BLE001
            state = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    have = {str(p.relative_to(MODELS_DIR))
            for p in MODELS_DIR.rglob("*") if p.is_file()}
    for b in bundles:
        try:
            mod = _bundle_module(b)
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print(f"WARNING: bundle {b.name} failed to import: {e}")
            continue
        rels = set(getattr(mod, "FILES", {}))
        sha = _sha256(b)
        if state.get(b.name) == sha and rels <= have:
            continue
        if verbose:
            print(f"extracting {b.name} -> models_v1/ ...")
        mod.extract(str(MODELS_DIR), quiet=not verbose)
        state[b.name] = sha
        have |= rels
    STATE_FILE.write_text(json.dumps(state, indent=1))
    _write_manifest(bundles)
    return True


# ------------------------------------------------------------- shared helpers
def norm(p):
    p = np.asarray(p, dtype=float)
    return p / p.sum(axis=-1, keepdims=True)


def logloss(p, y):
    eps = 1e-15
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y)
    if p.ndim == 1:
        return float(-np.mean(np.log(p[y.astype(int)])))
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def load_kit_evidence() -> dict:
    p = MODELS_DIR / "kit_evidence.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ====================================================== confidence rating ====
def confidence_rating(p, mid):
    """HIGH / MEDIUM / LOW from the final probability.

    HIGH   = p at or above the market's headline gate (0.60 for every
             pick market - the walk-forward point where accuracy lifts
             sharply).
    MEDIUM = p from 0.50 up to the gate.
    LOW    = below 0.50 (the pick is still shown, honestly labeled).
    """
    p = float(p)
    g = GATES.get(mid, 0.60)
    if p >= g - 1e-12:
        return "HIGH"
    if p >= CONF_MEDIUM - 1e-12:
        return "MEDIUM"
    return "LOW"


# ============================================================================
# SOFT-FACTOR LAYER - an LLM-compiled Adjustment Register, verified, capped,
# coherence-checked and applied to the model probabilities.
#
# 31 modules: M1-M23 carry the measured families / evidence bases / roles /
# max_pp of the original catalog (including the evidence-sweep corrections:
# M8 referee FLAG_ONLY on outcome markets, M9 weather 3.0pp, M6 2.0pp,
# M7/M7B 2.0pp, M11B 2.0pp). M24-M31 extend coverage to cards, corners,
# team totals, HT markets and win-to-nil; each is restricted to explicit
# markets and carries its evidence source in the comment. Those are HAND-SET
# ceilings (labelled MIXED/SPECULATIVE where the evidence is not measured) -
# the shadow ledger (--soft-log) re-measures them from outcomes.
# Trust boundary: the LLM compiles claims + evidence; THIS code owns the
# catalog, the floors, the caps and the application. Self-reported
# evidence_basis / causal_family are ignored - the catalog wins.
# ============================================================================

ENGINE_VERSION = "parlay_engine_v1 (engine 1.3.0)"

# ---------------------------------------------------------------------------
# SEAL-V1 - the anti-fabrication seal.
#
# Every genuine run seals its outputs: run_id = sha256 over the ENGINE.py
# bytes, the bundle bytes, the input CSV bytes, the gate table, the run
# parameters AND the generated results. The seal is deterministic (same kit +
# same inputs -> same run_id) and fragile (any edit to the report's numbers
# breaks it). It is printed to stdout as a SEAL::V1 line, embedded in the
# JSON ("seal") and appended to the readable report. `--check report.json`
# re-verifies all of it against the local kit. A hand-written, edited or
# "estimated" report CANNOT carry a valid seal: if a report has no valid
# seal, it was never produced by this engine - never use it, never bet on it.
# ---------------------------------------------------------------------------
SEAL_PROTOCOL = "SEAL-V1"
_SEAL_CORE_KEYS = ("engine", "bundles", "inputs", "gates", "params",
                   "version", "outputs_sha256")


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gates_digest() -> str:
    return hashlib.sha256(
        json.dumps(GATES, sort_keys=True).encode()).hexdigest()


def _outputs_digest(matches) -> str:
    """Canonical digest of the results array (layout-independent)."""
    return hashlib.sha256(json.dumps(
        matches, sort_keys=True, default=str).encode()).hexdigest()


def make_seal(input_files, params, matches) -> dict:
    """Build the run seal: kit hashes + input hashes + results digest."""
    core = {
        "engine": _sha256_file(Path(__file__).resolve()),
        "bundles": {p.name: _sha256_file(p) for p in find_bundles()},
        "inputs": {Path(p).name: _sha256_file(p) for p in sorted(input_files)},
        "gates": _gates_digest(),
        "params": json.loads(json.dumps(params or {}, default=str)),
        "version": ENGINE_VERSION,
        "outputs_sha256": _outputs_digest(matches),
    }
    rid = hashlib.sha256(json.dumps(
        core, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {**core, "protocol": SEAL_PROTOCOL, "run_id": rid}


def verify_seal(rep) -> list:
    """Re-verify a report's seal + internal contract. Empty list = genuine."""
    problems: list = []
    seal = rep.get("seal")
    if not isinstance(seal, dict):
        return ["no seal present - this report was NOT produced by ENGINE.py"]
    if seal.get("protocol") != SEAL_PROTOCOL:
        problems.append(f"unknown seal protocol: {seal.get('protocol')!r}")
    # kit identity: the report must come from THIS folder's engine + bundles
    if seal.get("engine") != _sha256_file(Path(__file__).resolve()):
        problems.append("engine sha256 mismatch - different ENGINE.py")
    local_bundles = {p.name: _sha256_file(p) for p in find_bundles()}
    if seal.get("bundles") != local_bundles:
        problems.append("bundle sha256 mismatch - model bundles differ "
                        "from this folder")
    if seal.get("gates") != _gates_digest():
        problems.append("gate table mismatch")
    # results integrity: any post-generation edit breaks the digest
    if seal.get("outputs_sha256") != _outputs_digest(rep.get("matches")):
        problems.append("outputs digest mismatch - the report's numbers "
                        "were modified after generation")
    # run_id re-derivation from the seal's own parts
    core = {k: seal.get(k) for k in _SEAL_CORE_KEYS}
    rid = hashlib.sha256(json.dumps(
        core, sort_keys=True, default=str).encode()).hexdigest()[:16]
    if seal.get("run_id") != rid:
        problems.append("run_id mismatch - seal is forged or inconsistent")
    # output contract: probabilities valid, gates + ratings match the code
    matches = rep.get("matches") or []
    if not matches:
        problems.append("report contains no matches")
    for m in matches:
        who = f"{m.get('home', '?')} v {m.get('away', '?')}"
        for p in m.get("final_picks") or []:
            mid = p.get("market")
            g = GATES.get(mid)
            if g is not None and abs(float(p.get("gate") or 0) - g) > 1e-9:
                problems.append(f"{who} {mid}: gate {p.get('gate')} != "
                                f"engine gate {g}")
            try:
                pf = float(p.get("p_final"))
            except (TypeError, ValueError):
                problems.append(f"{who} {mid}: p_final missing/not numeric")
                continue
            if not 0.0 <= pf <= 1.0:
                problems.append(f"{who} {mid}: p_final {pf} outside [0,1]")
            if p.get("confidence") != confidence_rating(pf, mid):
                problems.append(f"{who} {mid}: confidence rating "
                                f"{p.get('confidence')!r} inconsistent "
                                f"with p_final {pf}")
    return problems


def _seal_footer(seal, json_path) -> str:
    return ("SEAL::V1::" + str(seal.get("run_id")) + "\n"
            "verify: python 01_ENGINE.py --check " + str(json_path) + "\n"
            "A report without a valid seal is NOT engine output - "
            "never use it, never bet on it.")


def check_report(path) -> int:
    """--check: verify a report.json is a genuine, unmodified engine output."""
    p = Path(path)
    if not p.exists():
        print(f"FAIL: {path} does not exist")
        return 1
    try:
        rep = json.loads(p.read_text())
    except Exception as e:                                   # noqa: BLE001
        print(f"FAIL: cannot parse {path}: {type(e).__name__}: {e}")
        return 1
    print(f"SEAL CHECK - {path}")
    print(f"  kit: {Path(__file__).resolve().parent}")
    problems = verify_seal(rep)
    if problems:
        print("VERDICT: SEAL INVALID - NOT genuine engine output:")
        for q in problems:
            print(f"   - {q}")
        print("  never use an unsealed report - re-run the engine and "
              "use its own output")
        return 1
    seal = rep.get("seal") or {}
    n = len(rep.get("matches") or [])
    print(f"VERDICT: SEAL VERIFIED - genuine, unmodified ENGINE.py output "
          f"(run_id {seal.get('run_id')}, {n} match(es))")
    return 0


# --- measured constants -----------------------------------------------------
LEAGUE_TIERS = {
    "A": ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
    "B": ["2. Bundesliga", "Championship", "Segunda Division", "Serie B",
          "Ligue 2", "Eredivisie", "Primeira Liga",
          "Belgian First Division A", "Scottish Premiership", "Super Lig",
          "MLS", "UEFA Champions League"],
    "C": ["Allsvenskan", "Eliteserien", "Ekstraklasa", "Veikkausliiga",
          "Superliga", "NB I", "Czech First League", "Eerste Divisie",
          "Liga Portugal 2", "Liga Profesional", "Premier Division",
          "Canadian Premier League", "Super League", "UEFA Europa League",
          "UEFA Conference League"],
}
DEFAULT_LEAGUE_TIER = "B"

# RULE 10 magnitude bands, MEASURED (2026-09-17) from the distribution of
# real Pinnacle morning->close line moves (tier A n=27,423, tier B n=41,095;
# full-apply cap = p60, band-2 = p90, band-3 = p99 of |move| in pp).
RULE10_BANDS = {"A": (1.5, 4.0, 7.5), "B": (2.0, 5.0, 9.0), "C": (2.5, 6.0, 10.5)}

SOURCE_TIER_WEIGHTS = {"T1": 1.00, "T1_CLUB": 0.80, "T2": 0.80, "T2_DATA": 1.00,
                       "T3": 0.85, "T4": 0.50, "T5": 0.30}
EVIDENCE_FLOOR = {"ESTABLISHED": 1, "MIXED": 2, "SPECULATIVE": 3}
_SOURCE_TIER_ALIASES = set(SOURCE_TIER_WEIGHTS)
CREDIBLE_TIERS = ("T1", "T1_CLUB", "T2", "T2_DATA", "T3")

# modules whose evidence is a statistical aggregate (T2_DATA weighs full);
# on NEWS modules a data site is still secondary news (0.80).
STATS_MODULES = {"M1", "M3B", "M4", "M4B", "M4D", "M5G", "M7", "M7B", "M7C",
                 "M11E", "M13", "M13B", "M14B", "M16", "M22",
                 "M24", "M26", "M27", "M31"}

# RULE 23 model-overlap discount. The 135-feature matrix already consumes
# several soft-layer channels (form-5/10 + goals, elo, standings, referee
# career yellows/reds/home-win%, rest/turnaround/midweek/congestion, H2H,
# rolling xG, and the market odds block). Families WITHOUT model coverage
# ship at 1.00 - lineup news, weather, motivation, tactics, psychology.
MODEL_OVERLAP_DISCOUNT = {
    "form": 0.70, "referee": 0.60, "fatigue": 0.85, "H2H": 0.80,
    "script": 0.90,      # rolling xG + goal-timing proxy partially covered
}
MODEL_OVERLAP_DEFAULT = 1.00

# RULE 22 constants + the extended pair set. Same-family, same-sign
# deltas on a correlated pair double-count one underlying signal; the
# SMALLER delta is haircut x0.50. Different families = independent evidence.
SHARED_SIGNAL_HAIRCUT = 0.50
SHARED_SIGNAL_PAIRS = (
    # measured pair-stats lift 1.50-1.62 (n=94,482)
    ("ou25:over", "btts:yes"), ("ou25:under", "btts:no"),
    # extension: nested totals + team totals (one goal-expectancy signal,
    # expressed at several lines)
    ("ou15:over", "ou25:over"), ("ou25:over", "ou35:over"),
    ("ou15:under", "ou25:under"), ("ou25:under", "ou35:under"),
    ("tt_h15:home_over", "ou25:over"), ("tt_a15:away_over", "ou25:over"),
    ("tt_h15:home_under", "ou25:under"), ("tt_a15:away_under", "ou25:under"),
    # dnb is a subset of 1X2 (draw voids); HT shares the FT signal
    ("x12:home_win", "dnb:home"), ("x12:away_win", "dnb:away"),
    ("x12:home_win", "ht13:home_win"), ("x12:draw", "ht13:draw"),
    ("x12:away_win", "ht13:away_win"),
    # win-to-nil mechanics: home win to nil == away under 0.5 (and mirror)
    ("tt_a05:away_under", "wtn_h:yes"), ("tt_h05:home_under", "wtn_a:yes"),
    # AH cover shares the 1X2 side signal
    ("x12:home_win", "ah:home_cover"), ("x12:away_win", "ah:away_cover"),
)
DC_COHERENCE_TOLERANCE_PP = 0.3

# RULE 24 syndication dedup: agency rewires across domains collapse to one
# corroboration unit when their quoted snippets are near-identical.
SYNDICATION_JACCARD = 0.80

# RULE 21: these families decay with distance to KICKOFF, not publication.
KICKOFF_SENSITIVE_FAMILIES = {"lineup", "fatigue", "motivation", "psychology",
                              "script"}

# RULE 20 priced-in discounts, MEASURED (2026-09-17): once absorbed, a move
# keeps only 2-20% of its residual (pooled ~11%).
PRICED_IN_DISCOUNT = {"YES": 0.10, "PARTIAL": 0.55, "NO": 1.00, "UNKNOWN": 1.00}

# SOFT_LAYER_CALIBRATION - the measured-magnitude table.
SOFT_LAYER_CALIBRATION = {
    "transfer_beta": {"x12": 1.0, "ou25": 1.0, "btts": 1.0, "ou15": 1.0,
                      "ou35": 1.0, "tt_h05": 1.0, "tt_h15": 1.0, "tt_a05": 1.0,
                      "tt_a15": 1.0, "dc1x": 1.0, "dcx2": 1.0, "dc12": 1.0,
                      "dnb": 1.0, "wtn_h": 1.0, "wtn_a": 1.0, "corners": 1.0,
                      "cards": 1.0, "ht13": 1.0, "ah": 1.0},
    "transfer_provenance": ("measured 2026-09-17: k* 1.03-1.20 across 1X2 "
                            "h/d/a and O/U over/under; ceiling capped at 1.0"),
    # Claim-size prior: P(|real market move| >= threshold). The 1X2 grid
    # covers x12/dnb/dc/ah/wtn/tt targets; the O/U grid covers ou/btts/
    # corners/cards (no pre-close odds measured for them - pooled
    # inheritance, documented).
    "claim_prior": {
        "1X2": {2.0: 0.299, 4.0: 0.115, 6.0: 0.040, 8.0: 0.014, 10.0: 0.004},
        "OU": {2.0: 0.471, 4.0: 0.176, 6.0: 0.053, 8.0: 0.014, 10.0: 0.004},
    },
    "family_residual_pp": {"form": 0.59, "fatigue": 0.57, "motivation": 0.51,
                           "referee": 0.73, "environment": 0.36,
                           "tactics": 0.70},
    "measured_on": "2026-09-17",
    "measured_n": {"fixtures": 127317, "move_legs_pooled": 125960,
                   "ix2_pairs": 90494, "ou_pairs": 35467, "chain_rows": 22245},
}

_MODULE_ALIAS = {  # accept alternative market-name spellings in registers
    "1x2": "x12", "o/u 2.5": "ou25", "ou 2.5": "ou25", "over/under 2.5": "ou25",
    "o/u 1.5": "ou15", "o/u 3.5": "ou35", "btts": "btts",
    "dc 1x": "dc1x", "dc x2": "dcx2", "dc 12": "dc12", "double chance 1x": "dc1x",
    "ah": "ah", "asian handicap": "ah", "ht 1x2": "ht13", "ht1x2": "ht13",
    "corners": "corners", "cards": "cards", "team total home": "tt_h15",
    "team total away": "tt_a15", "dnb": "dnb", "draw no bet": "dnb",
    "htft": "htft", "ht/ft": "htft", "cs": "cs", "correct score": "cs",
}


def norm_market(market):
    """Normalize a register market id (accepts parlay-style aliases)."""
    m = str(market or "").strip()
    return _MODULE_ALIAS.get(m.lower(), m.lower())


def claim_rarity_pct(market, delta_pp):
    """Percentile context: P(|real market move| >= |delta_pp|), or None."""
    grid = SOFT_LAYER_CALIBRATION["claim_prior"].get(
        "OU" if market.split("_")[0] in ("ou", "tt") or market in
        ("btts", "corners", "cards") else "1X2")
    if not grid or delta_pp is None or not np.isfinite(float(delta_pp)):
        return None
    d = abs(float(delta_pp))
    thrs = sorted(grid)
    if d < thrs[0]:
        return None
    if d >= thrs[-1]:
        p_hi = grid[thrs[-1]]
        return max(0.01, p_hi * (thrs[-1] / max(d, thrs[-1])) ** 3)
    for lo, hi in zip(thrs, thrs[1:]):
        if lo <= d < hi:
            t = (d - lo) / (hi - lo)
            return grid[lo] + t * (grid[hi] - grid[lo])
    return grid[thrs[-1]]


# --- module catalog (code-side source of truth) -----------------------------
MODULE_CATALOG = {
    # module: family, evidence basis, role, max_pp, [restricted markets]
    "M1":   {"family": "form",        "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 5.0},
    "M2":   {"family": "environment", "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 4.0},
    "M2B":  {"family": "environment", "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M2C":  {"family": "environment", "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 1.0},
    "M3":   {"family": "lineup",      "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 5.0},
    "M3B":  {"family": "form",        "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M3C":  {"family": "lineup",      "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M3D":  {"family": "tactics",     "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.0},
    "M4":   {"family": "tactics",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 3.0},
    "M4B":  {"family": "tactics",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 1.0},
    "M4C":  {"family": "tactics",     "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M4D":  {"family": "lineup",      "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.0},
    "M4E":  {"family": "tactics",     "basis": "MIXED",       "role": "FLAG_ONLY", "max_pp": 1.0},
    "M5":   {"family": "motivation",  "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 5.0},
    "M5B":  {"family": "fatigue",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 5.0},
    "M5C":  {"family": "psychology",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 3.0},
    "M5D":  {"family": "fatigue",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 4.0},
    "M5E":  {"family": "motivation",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 3.0},
    "M5F":  {"family": "tactics",     "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 2.0},
    "M5G":  {"family": "fatigue",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 2.0},
    # M6: SPECULATIVE with the joint-largest claimed range and no published
    # effect size (2026-10 evidence sweep) -> ceiling held at 2.0pp.
    "M6":   {"family": "psychology",  "basis": "SPECULATIVE", "role": "ADJUST",    "max_pp": 2.0},
    # M7/M7B: no H2H outcome value beyond ~5 meetings once form/
    # squad/HA are controlled (J Sports Analytics 2018) -> 2.0pp.
    "M7":   {"family": "H2H",         "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M7B":  {"family": "H2H",         "basis": "SPECULATIVE", "role": "ADJUST",    "max_pp": 2.0},
    "M7C":  {"family": "H2H",         "basis": "MIXED",       "role": "ADJUST",    "max_pp": 3.0},
    "M7D":  {"family": "H2H",         "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 2.0},
    # M8 family: the referee channel on OUTCOME markets is measured noise
    # (z = 0.9 O/U, 0.2 BTTS, 2.0 home win across 18 family-stream tests)
    # and crowd-mediated (Bilalic et al. 2021) -> FLAG_ONLY. The CARDS
    # referee channel is real and lives in M24/M29 below.
    "M8":   {"family": "referee",     "basis": "MIXED",       "role": "FLAG_ONLY", "max_pp": 1.0},
    "M8B":  {"family": "referee",     "basis": "MIXED",       "role": "FLAG_ONLY", "max_pp": 1.0},
    "M8C":  {"family": "referee",     "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 1.0},
    "M8D":  {"family": "referee",     "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 1.0},
    "M8E":  {"family": "referee",     "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 0.5},
    "M8F":  {"family": "referee",     "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 0.0},
    # M9: the Kepler 6,108-match study found rain does NOT reduce
    # goals; only wind/waterlogging/snow/serious heat move totals -> 3.0pp.
    "M9":   {"family": "environment", "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 3.0},
    "M9B":  {"family": "environment", "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.0},
    "M9C":  {"family": "environment", "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.0},
    "M9D":  {"family": "environment", "basis": "SPECULATIVE", "role": "ADJUST",    "max_pp": 3.0},
    "M9E":  {"family": "environment", "basis": "ESTABLISHED", "role": "FLAG_ONLY", "max_pp": 0.0},
    "M9F":  {"family": "environment", "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 0.0},
    "M10":  {"family": "intuition_pattern", "basis": "EXEMPT", "role": "FLAG_ONLY", "max_pp": 0.0},
    "M11":  {"family": "motivation",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 3.0},
    # M11B: new-manager bounce = regression to the mean
    # (Ter Weel, Eredivisie, 18 seasons; Soccernomics/Bridgewater) -> 2.0pp.
    "M11B": {"family": "motivation",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M11C": {"family": "motivation",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 2.0},
    "M11D": {"family": "motivation",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 2.0},
    "M11E": {"family": "tactics",     "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M11F": {"family": "motivation",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 0.0},
    "M12":  {"family": "psychology",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 2.0},
    "M12B": {"family": "psychology",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M12C": {"family": "psychology",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 1.0},
    "M12D": {"family": "psychology",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 1.0},
    "M13":  {"family": "script",      "basis": "MIXED",       "role": "ADJUST",    "max_pp": 0.5},
    "M13B": {"family": "script",      "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 2.0},
    "M14":  {"family": "psychology",  "basis": "SPECULATIVE", "role": "FLAG_ONLY", "max_pp": 1.0},
    "M14B": {"family": "lineup",      "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M15":  {"family": "lineup",      "basis": "SPECULATIVE", "role": "ADJUST",    "max_pp": 2.0},
    "M16":  {"family": "form",        "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M17":  {"family": "fatigue",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 2.0},
    "M18":  {"family": "psychology",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.5},
    "M19":  {"family": "motivation",  "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M20":  {"family": "environment", "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0},
    "M21":  {"family": "lineup",      "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.5},
    "M22":  {"family": "lineup",      "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 2.0},
    "M23":  {"family": "environment", "basis": "ESTABLISHED", "role": "FLAG_ONLY", "max_pp": 0.0},
    # ---- extension modules M24-M31 (cards, corners, team totals, HT
    # markets, win-to-nil) ----
    # Each module is market-restricted and HAND-SET (the shadow ledger
    # re-measures it). Named evidence in the comment.
    # M24: referee card profile -> the CARDS channel. Referee career yellow
    # averages are the standard cards predictor (data-guide consensus:
    # statsbench 2026, scorelab 2026) and the crowd-cards link is measured
    # (Downward 2007; Portsmouth 2022 COVID study). The cards model DOES
    # carry referee features -> RULE 23 overlap 0.60 applies.
    "M24":  {"family": "referee",     "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 3.0,
             "markets": ("cards",)},
    # M25: derby/rivalry intensity raises cards and corners (mixed
    # literature; practitioner consensus). HAND-SET 2.0pp.
    "M25":  {"family": "H2H",         "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0,
             "markets": ("cards", "corners")},
    # M26: match-state script - a chasing team piles on late corners
    # (~60% of corners arrive in the 2nd half, Opta via SharkBetting 2026)
    # and late cards. Pre-match proxy: expected match state from the price.
    "M26":  {"family": "script",      "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 2.0,
             "markets": ("corners", "cards", "ou25", "ou35", "tt_h15", "tt_a15")},
    # M27: set-piece / crossing / aerial style drives corners (statbible
    # 2026, postudium 2026). HAND-SET 1.5pp.
    "M27":  {"family": "tactics",     "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.5,
             "markets": ("corners",)},
    # M28: extreme weather/pitch on NON-goal markets (M9 owns the goals
    # markets; the collision check keeps the pair honest if both fire).
    "M28":  {"family": "environment", "basis": "MIXED",       "role": "ADJUST",    "max_pp": 2.0,
             "markets": ("corners", "cards", "ou15", "ou25", "ou35")},
    # M29: VAR-official profile -> cards rise after VAR interventions
    # (Samuel et al. 2024, IJSSC: 2x yellows/minute after an intervention,
    # d = 0.40). HAND-SET 1.5pp on cards only.
    "M29":  {"family": "referee",     "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.5,
             "markets": ("cards",)},
    # M30: dead rubber / table symmetry -> more randomness (Expecting Goals
    # 2026, via the parlay evidence sweep). Modest variance deltas.
    "M30":  {"family": "motivation",  "basis": "ESTABLISHED", "role": "ADJUST",    "max_pp": 2.0,
             "markets": ("ou25", "ou35", "corners", "cards", "ht13")},
    # M31: early-goal expectancy -> HT markets (~74.8% of matches see a
    # first-half goal; goals cluster late in each half - goalstatistics
    # 100k-match DB). HAND-SET 1.5pp.
    "M31":  {"family": "script",      "basis": "MIXED",       "role": "ADJUST",    "max_pp": 1.5,
             "markets": ("ht13", "ou15")},
}
FLAG_ONLY_MODULES = tuple(sorted(m for m, c in MODULE_CATALOG.items()
                                 if c["role"] == "FLAG_ONLY"))

# --- market <-> target mapping (delta-capable markets only) -----------------
# htft (9 classes) and cs (21 classes) are FLAG-ONLY markets: individual
# score/class deltas are meaningless at pp scale; flags still apply.
MARKET_TARGETS = {
    "x12":    ["home_win", "draw", "away_win"],
    "ht13":   ["home_win", "draw", "away_win"],
    "ou15":   ["over", "under"], "ou25": ["over", "under"],
    "ou35":   ["over", "under"],
    "tt_h05": ["home_over", "home_under"], "tt_h15": ["home_over", "home_under"],
    "tt_a05": ["away_over", "away_under"], "tt_a15": ["away_over", "away_under"],
    "btts":   ["yes", "no"],
    "dc1x":   ["win"], "dcx2": ["win"], "dc12": ["win"],
    "dnb":    ["home", "away"],
    "wtn_h":  ["yes", "no"], "wtn_a": ["yes", "no"],
    "corners": ["over", "under"], "cards": ["over", "under"],
    "ah":     ["home_cover", "away_cover"],
}
# class-order target names per market (aligned with the model classes arrays)
TARGETS_BY_CLASS = {
    "x12":    ["away_win", "draw", "home_win"],          # classes A, D, H
    "ht13":   ["away_win", "draw", "home_win"],
    "dnb":    ["away", "home"],                          # classes away, home
    "ah":     [],                                        # margin classes, special
}
_BIN = ["under", "over"]           # ou* / corners / cards (classes no, yes)
TARGETS_BY_CLASS.update({m: _BIN for m in ("ou15", "ou25", "ou35",
                                           "corners", "cards")})
TARGETS_BY_CLASS.update({m: ["home_under", "home_over"]
                         for m in ("tt_h05", "tt_h15")})
TARGETS_BY_CLASS.update({m: ["away_under", "away_over"]
                         for m in ("tt_a05", "tt_a15")})
TARGETS_BY_CLASS.update({m: ["no", "yes"] for m in ("btts", "wtn_h", "wtn_a")})
TARGETS_BY_CLASS.update({m: ["no", "win"] for m in ("dc1x", "dcx2", "dc12")})

# friendly feature labels for the XAI layer (full 135-column coverage)
FEATURE_LABELS = {
    "elo_home": "home Elo", "elo_away": "away Elo", "elo_diff": "Elo gap",
    "pos_home": "home table position", "pos_away": "away table position",
    "pos_diff": "table position gap", "ppg_home": "home points/game",
    "ppg_away": "away points/game", "ppg_diff": "points/game gap",
    "gd_home": "home goal difference", "gd_away": "away goal difference",
    "gd_diff": "goal-difference gap", "games_home": "home games played",
    "games_away": "away games played",
    "win_rate_home": "home win rate (form-5)", "win_rate_away": "away win rate (form-5)",
    "draw_rate_home": "home draw rate (form-5)", "draw_rate_away": "away draw rate (form-5)",
    "form_pts_home": "home form points (5)", "form_pts_away": "away form points (5)",
    "form_diff": "form-5 points gap", "scored_home_5": "home goals scored (5)",
    "conceded_home_5": "home goals conceded (5)", "scored_away_5": "away goals scored (5)",
    "conceded_away_5": "away goals conceded (5)", "scored_diff": "scoring gap (5)",
    "conceded_diff": "defensive gap (5)", "attack_h_vs_def_a": "home attack vs away defence (5)",
    "attack_a_vs_def_h": "away attack vs home defence (5)",
    "rest_home": "home rest days", "rest_away": "away rest days",
    "rest_diff": "rest-days gap", "short_turnaround_home": "home short turnaround",
    "short_turnaway_away": "away short turnaround", "midweek_home": "home midweek match",
    "midweek_away": "away midweek match", "matchday": "matchday number",
    "day_of_week": "day of week", "is_weekend": "weekend fixture",
    "kickoff_hour": "kickoff hour", "imp_A": "implied away prob (odds)",
    "imp_D": "implied draw prob (odds)", "imp_H": "implied home prob (odds)",
    "log_odds_A": "log odds away", "log_odds_D": "log odds draw",
    "log_odds_H": "log odds home", "overround": "bookmaker overround",
    "hist_n_home": "home history depth", "hist_n_away": "away history depth",
    "form10_w_home": "home win rate (10)", "form10_w_away": "away win rate (10)",
    "form10_pts_home": "home form points (10)", "form10_pts_away": "away form points (10)",
    "form10_diff": "form-10 points gap", "goals10_scored_home": "home goals scored (10)",
    "goals10_conceded_home": "home goals conceded (10)",
    "goals10_scored_away": "away goals scored (10)",
    "goals10_conceded_away": "away goals conceded (10)",
    "att10_h_vs_def_a": "home attack vs away defence (10)",
    "att10_a_vs_def_h": "away attack vs home defence (10)",
    "venue_ppg_home": "home venue points/game", "venue_scored_home": "home venue goals scored",
    "venue_conceded_home": "home venue goals conceded",
    "venue_ppg_away": "away venue points/game", "venue_scored_away": "away venue goals scored",
    "venue_conceded_away": "away venue goals conceded", "venue_diff": "venue form gap",
    "h2h_n": "H2H meetings", "h2h_ptshare": "H2H home point share",
    "h2h_gd": "H2H goal difference", "h2h_last": "H2H last meeting (home pts)",
    "elo_delta_home": "home Elo trend (10)", "elo_delta_away": "away Elo trend (10)",
    "congestion_21d_home": "home congestion (21d)", "congestion_21d_away": "away congestion (21d)",
    "xg_for5_home": "home xG for (5)", "xg_against5_home": "home xG against (5)",
    "xg_for5_away": "away xG for (5)", "xg_against5_away": "away xG against (5)",
    "ref_known": "referee data known", "ref_matches": "referee career matches",
    "ref_yellows": "referee career yellows/game", "ref_reds": "referee career reds/game",
    "ref_homewin": "referee home-win rate", "ref_seas_matches": "referee season matches",
    "ref_seas_yellows": "referee season yellows/game",
    "mk_ah_home": "market AH home prob", "mk_ah_known": "AH line known",
    "mk_ah_line": "AH line", "mk_ah_abs": "AH line magnitude",
    "mk_over25": "market over-2.5 prob", "mk_ou_known": "O/U odds known",
    "mk_btts": "market BTTS yes prob", "mk_btts_known": "BTTS odds known",
    "mk_bf_H": "Betfair home prob", "mk_bf_D": "Betfair draw prob",
    "mk_bf_A": "Betfair away prob", "mk_bf_known": "Betfair odds known",
    "mk_spread_H": "cross-book home spread", "mk_spread_D": "cross-book draw spread",
    "mk_spread_A": "cross-book away spread", "mk_n_books": "number of books quoting",
}


# ============================================================================
# ==================================================== feature pipeline ======
# The exact pipeline that produced every archived walk-forward number,
# merged into one pass and hardened for fixture rows that have no result yet:
#   - rows without a played result never enter any rolling history
#   - country dummies are built against the fixed training category list
#   - the final frame is reindexed to the exact 135 model feature columns
def build_features(csv_paths, use_cache=True, progress=True, frame=None):
    """Parse the master CSVs -> the 135-column B-XL feature frame + context.

    frame: an optional pre-loaded DataFrame (single-match mode) - when given,
    csv_paths is ignored and no cache is used.

    Returns the payload dict:
      X_B_XL      DataFrame, columns = the 135 model features (ordered)
      y           int8 array, A=0 / D=1 / H=2 (-1 for unplayed fixtures)
      date        datetime64 array
      odds_best   (n,3) float32 [A,D,H], NaN where unavailable
      odds_pinnacle (n,3) float32
      odds_ou25   (n,2) float32 [over, under] - first available book chain
      odds_btts   (n,2) float32 [yes, no]
      odds_ah     (n,3) float32 [line, home, away]
      has_odds, src, meta (DataFrame: date/home/away/competition/country/goals)
    """
    if frame is not None:
        df = frame.copy().reset_index(drop=True)
    else:
        csv_paths = [Path(p) for p in csv_paths]
        cache_key = None
        if use_cache and csv_paths and all(p.exists() for p in csv_paths):
            stat = [(str(p.name), p.stat().st_size, int(p.stat().st_mtime))
                    for p in sorted(csv_paths, key=str)]
            cache_key = hashlib.sha256(
                (f"v{CACHE_VERSION}|" + json.dumps(stat)).encode()).hexdigest()[:16]
            cf = CACHE_DIR / f"features_{cache_key}.pkl"
            if cf.exists():
                try:
                    with open(cf, "rb") as f:
                        payload = pickle.load(f)
                    if progress:
                        print(f"features cache hit ({cf.name}, "
                              f"{len(payload['y'])} rows)")
                    return payload
                except Exception:                              # noqa: BLE001
                    pass

        df = pd.concat([pd.read_csv(p, low_memory=False) for p in csv_paths],
                       ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    assert df["match_date"].notna().all(), "unparseable match_date rows"
    df = df.sort_values("match_date", kind="stable").reset_index(drop=True)

    played = df["FTR"].isin(["A", "D", "H"]).to_numpy()
    y = np.where(played, df["FTR"].map({"A": 0, "D": 1, "H": 2}).fillna(-1), -1
                 ).astype(np.int8)

    # ---------------- Track A features (pre-match only, no odds) -----------
    F = pd.DataFrame(index=df.index)
    F["elo_home"] = df["home_elo"]
    F["elo_away"] = df["away_elo"]
    F["elo_diff"] = df["elo_diff"]
    F["pos_home"] = df["home_position"]
    F["pos_away"] = df["away_position"]
    F["pos_diff"] = df["away_position"] - df["home_position"]
    gp_h = df["home_games_played"].replace(0, np.nan)
    gp_a = df["away_games_played"].replace(0, np.nan)
    F["ppg_home"] = df["home_points"] / gp_h
    F["ppg_away"] = df["away_points"] / gp_a
    F["ppg_diff"] = F["ppg_home"] - F["ppg_away"]
    F["gd_home"] = df["home_goal_diff"]
    F["gd_away"] = df["away_goal_diff"]
    F["gd_diff"] = F["gd_home"] - F["gd_away"]
    F["games_home"] = df["home_games_played"]
    F["games_away"] = df["away_games_played"]

    f5 = ["home_form_5_wins", "home_form_5_draws", "home_form_5_losses",
          "away_form_5_wins", "away_form_5_draws", "away_form_5_losses"]
    if df[f5].isna().any().any():
        df[f5] = df[f5].fillna(0)
    F["win_rate_home"] = df["home_form_5_wins"] / 5.0
    F["win_rate_away"] = df["away_form_5_wins"] / 5.0
    F["draw_rate_home"] = df["home_form_5_draws"] / 5.0
    F["draw_rate_away"] = df["away_form_5_draws"] / 5.0
    F["form_pts_home"] = (3 * df["home_form_5_wins"] + df["home_form_5_draws"]) / 5.0
    F["form_pts_away"] = (3 * df["away_form_5_wins"] + df["away_form_5_draws"]) / 5.0
    F["form_diff"] = F["form_pts_home"] - F["form_pts_away"]

    F["scored_home_5"] = df["home_avg_goals_scored_5"]
    F["conceded_home_5"] = df["home_avg_goals_conceded_5"]
    F["scored_away_5"] = df["away_avg_goals_scored_5"]
    F["conceded_away_5"] = df["away_avg_goals_conceded_5"]
    F["scored_diff"] = F["scored_home_5"] - F["scored_away_5"]
    F["conceded_diff"] = F["conceded_home_5"] - F["conceded_away_5"]
    F["attack_h_vs_def_a"] = F["scored_home_5"] - F["conceded_away_5"]
    F["attack_a_vs_def_h"] = F["scored_away_5"] - F["conceded_home_5"]

    F["rest_home"] = df["home_days_since_last_match"]
    F["rest_away"] = df["away_days_since_last_match"]
    F["rest_diff"] = F["rest_home"] - F["rest_away"]
    F["short_turnaround_home"] = df["home_short_turnaround"]
    F["short_turnaway_away"] = df["away_short_turnaround"]
    F["midweek_home"] = df["home_midweek_match"]
    F["midweek_away"] = df["away_midweek_match"]
    F["matchday"] = df["matchday_number"]
    F["day_of_week"] = df["match_day_of_week"]
    F["is_weekend"] = df["is_weekend"].astype(float)
    F["kickoff_hour"] = df["kickoff_hour"]

    # country one-hot: the model's c_* feature columns are the source of
    # truth; the fallback list covers the training categories
    cats = ["Argentina", "Austria", "Belgium", "China", "England", "Europe",
            "Finland", "France", "Germany", "Greece", "Ireland", "Italy",
            "Japan", "Mexico", "Netherlands", "Norway", "Other", "Portugal",
            "Romania", "Russia", "Scotland", "Spain", "Sweden", "Switzerland",
            "Turkey", "Turkiye", "USA"]
    cats = [c[2:] for c in FEATURE_COLUMNS if c.startswith("c_")] or cats
    F["country"] = pd.Categorical(df["Country"].fillna("Other").astype(str),
                                  categories=cats)
    X_A = pd.get_dummies(F, columns=["country"], prefix="c", dtype=np.float32)
    X_A = X_A.astype(np.float32)

    # ---------------- best-available odds (fallback chain) ----------------
    chain = [
        ("pinnacle", "PSH", "PSD", "PSA"),
        ("b365", "B365H", "B365D", "B365A"),
        ("max", "MaxH", "MaxD", "MaxA"),
        ("williamhill", "WHH", "WHD", "WHA"),
        ("interwetten", "IWH", "IWD", "IWA"),
        ("avg", "AvgH", "AvgD", "AvgA"),
    ]
    n = len(df)
    oH = np.full(n, np.nan); oD = np.full(n, np.nan); oA = np.full(n, np.nan)  # noqa: E702
    src = np.full(n, "", dtype=object)
    for name, h, d, a in chain:
        oh = pd.to_numeric(df[h], errors="coerce").to_numpy(dtype=float)
        od = pd.to_numeric(df[d], errors="coerce").to_numpy(dtype=float)
        oa = pd.to_numeric(df[a], errors="coerce").to_numpy(dtype=float)
        valid = (oh > 1.0) & (od > 1.0) & (oa > 1.0)
        need = np.isnan(oH) & valid
        oH[need] = oh[need]; oD[need] = od[need]; oA[need] = oa[need]  # noqa: E702
        src[need] = name
    has_odds = ~np.isnan(oH)

    odds_best = np.column_stack([oA, oD, oH]).astype(np.float32)
    inv = 1.0 / odds_best
    margin = inv.sum(axis=1, keepdims=True)
    imp = inv / margin

    pH = pd.to_numeric(df["PSH"], errors="coerce").to_numpy(dtype=float)
    pD = pd.to_numeric(df["PSD"], errors="coerce").to_numpy(dtype=float)
    pA = pd.to_numeric(df["PSA"], errors="coerce").to_numpy(dtype=float)
    pv = (pH > 1) & (pD > 1) & (pA > 1)
    odds_pinnacle = np.column_stack([pA, pD, pH]).astype(np.float32)
    odds_pinnacle[~pv] = np.nan

    X_B = X_A.copy()
    X_B["imp_A"] = imp[:, 0]
    X_B["imp_D"] = imp[:, 1]
    X_B["imp_H"] = imp[:, 2]
    X_B["log_odds_A"] = np.log(odds_best[:, 0])
    X_B["log_odds_D"] = np.log(odds_best[:, 1])
    X_B["log_odds_H"] = np.log(odds_best[:, 2])
    X_B["overround"] = margin[:, 0] - 1.0
    for name in ["pinnacle", "b365", "max", "williamhill", "interwetten", "avg"]:
        X_B[f"src_{name}"] = (src == name).astype(np.float32)
    X_B = X_B.astype(np.float32)

    # ------------------------- computed history (XL) ----------------------
    home = df["home_team"].astype(str).to_numpy()
    away = df["away_team"].astype(str).to_numpy()
    dates = df["match_date"].to_numpy()
    gh = NUM(df["full_time_home_goals"]).to_numpy(dtype=float)
    ga = NUM(df["full_time_away_goals"]).to_numpy(dtype=float)
    xg_h = NUM(df["xg_home"]).to_numpy(dtype=float)
    xg_a = NUM(df["xg_away"]).to_numpy(dtype=float)
    elo_h = NUM(df["home_elo"]).to_numpy(dtype=float)
    elo_a = NUM(df["away_elo"]).to_numpy(dtype=float)
    goals_ok = np.isfinite(gh) & np.isfinite(ga) & played

    H = {}
    P = {}
    N10, N5, NV, NH2H = 10, 5, 5, 6

    f_w10h = np.full(n, np.nan); f_p10h = np.full(n, np.nan)
    f_w10a = np.full(n, np.nan); f_p10a = np.full(n, np.nan)
    f_s10h = np.full(n, np.nan); f_c10h = np.full(n, np.nan)
    f_s10a = np.full(n, np.nan); f_c10a = np.full(n, np.nan)
    f_nh = np.zeros(n); f_na = np.zeros(n)
    f_vph = np.full(n, np.nan); f_vsh = np.full(n, np.nan); f_vch = np.full(n, np.nan)
    f_vpa = np.full(n, np.nan); f_vsa = np.full(n, np.nan); f_vca = np.full(n, np.nan)
    f_h2n = np.zeros(n); f_h2ps = np.full(n, np.nan)
    f_h2gd = np.full(n, np.nan); f_h2last = np.full(n, np.nan)
    f_edh = np.full(n, np.nan); f_eda = np.full(n, np.nan)
    f_cgh = np.zeros(n); f_cga = np.zeros(n)
    f_xgh_f = np.full(n, np.nan); f_xgh_a = np.full(n, np.nan)
    f_xga_f = np.full(n, np.nan); f_xga_a = np.full(n, np.nan)

    t0 = time.time()
    for i in range(n):
        d = dates[i]
        h, a = home[i], away[i]
        hh = H.get(h, [])
        ha = H.get(a, [])

        def roll(hist, w10, p10, s10, c10, ncnt):
            m = hist[-N10:]
            ncnt[i] = len(m)
            if m:
                pts = sum(e[4] for e in m); w = sum(1 for e in m if e[4] == 3)
                p10[i] = pts / len(m); w10[i] = w / len(m)
                s10[i] = np.mean([e[2] for e in m]); c10[i] = np.mean([e[3] for e in m])

        roll(hh, f_w10h, f_p10h, f_s10h, f_c10h, f_nh)
        roll(ha, f_w10a, f_p10a, f_s10a, f_c10a, f_na)

        def venue(hist, want, ppg, sc, cc):
            m = [e for e in hist if e[1] == want][-NV:]
            if m:
                ppg[i] = np.mean([e[4] for e in m])
                sc[i] = np.mean([e[2] for e in m]); cc[i] = np.mean([e[3] for e in m])

        venue(hh, "H", f_vph, f_vsh, f_vch)
        venue(ha, "A", f_vpa, f_vsa, f_vca)

        for hist, arr in ((hh, f_cgh), (ha, f_cga)):
            c = 0
            for e in reversed(hist):
                dd = (d - e[0]) / np.timedelta64(1, "D")
                if dd > 21:
                    break
                c += 1
            arr[i] = c

        if len(hh) >= N10 and np.isfinite(elo_h[i]):
            f_edh[i] = elo_h[i] - hh[-N10][7]
        if len(ha) >= N10 and np.isfinite(elo_a[i]):
            f_eda[i] = elo_a[i] - ha[-N10][7]

        for hist, ff, fa in ((hh, f_xgh_f, f_xgh_a), (ha, f_xga_f, f_xga_a)):
            m = [e for e in hist if e[5] is not None and np.isfinite(e[5])][-N5:]
            if m:
                ff[i] = np.mean([e[5] for e in m]); fa[i] = np.mean([e[6] for e in m])

        key = (h, a) if h <= a else (a, h)
        past = P.get(key, [])
        f_h2n[i] = min(len(past), 10)
        m6 = past[-NH2H:]
        if m6:
            pts_h = 0
            gds = []
            for (pd_, hs, hg, ag) in m6:
                if hs == h:
                    p = 3 if hg > ag else (1 if hg == ag else 0)
                    gds.append(hg - ag)
                else:
                    p = 3 if ag > hg else (1 if hg == ag else 0)
                    gds.append(ag - hg)
                pts_h += p
            f_h2ps[i] = pts_h / (3 * len(m6))
            f_h2gd[i] = float(np.mean(gds))
            pd_, hs, hg, ag = m6[-1]
            if hs == h:
                f_h2last[i] = 3 if hg > ag else (1 if hg == ag else 0)
            else:
                f_h2last[i] = 3 if ag > hg else (1 if hg == ag else 0)

        # append current match to histories ONLY when it was actually played
        if goals_ok[i]:
            pts_h = 3 if gh[i] > ga[i] else (1 if gh[i] == ga[i] else 0)
            pts_a = 3 if ga[i] > gh[i] else (1 if gh[i] == ga[i] else 0)
            xh = float(xg_h[i]) if np.isfinite(xg_h[i]) else None
            xa = float(xg_a[i]) if np.isfinite(xg_a[i]) else None
            eh = float(elo_h[i]) if np.isfinite(elo_h[i]) else np.nan
            ea = float(elo_a[i]) if np.isfinite(elo_a[i]) else np.nan
            H.setdefault(h, []).append((d, "H", gh[i], ga[i], pts_h, xh, xa, eh))
            H.setdefault(a, []).append((d, "A", ga[i], gh[i], pts_a, xa, xh, ea))
            P.setdefault(key, []).append((d, h, gh[i], ga[i]))

        if progress and (i + 1) % 20000 == 0:
            print(f"  features {i+1}/{n} rows ({time.time()-t0:.0f}s)", flush=True)

    NEW = pd.DataFrame(index=df.index)
    NEW["hist_n_home"] = f_nh; NEW["hist_n_away"] = f_na
    NEW["form10_w_home"] = f_w10h; NEW["form10_w_away"] = f_w10a
    NEW["form10_pts_home"] = f_p10h; NEW["form10_pts_away"] = f_p10a
    NEW["form10_diff"] = f_p10h - f_p10a
    NEW["goals10_scored_home"] = f_s10h; NEW["goals10_conceded_home"] = f_c10h
    NEW["goals10_scored_away"] = f_s10a; NEW["goals10_conceded_away"] = f_c10a
    NEW["att10_h_vs_def_a"] = f_s10h - f_c10a
    NEW["att10_a_vs_def_h"] = f_s10a - f_c10h
    NEW["venue_ppg_home"] = f_vph; NEW["venue_scored_home"] = f_vsh
    NEW["venue_conceded_home"] = f_vch
    NEW["venue_ppg_away"] = f_vpa; NEW["venue_scored_away"] = f_vsa
    NEW["venue_conceded_away"] = f_vca
    NEW["venue_diff"] = f_vph - f_vpa
    NEW["h2h_n"] = f_h2n; NEW["h2h_ptshare"] = f_h2ps
    NEW["h2h_gd"] = f_h2gd; NEW["h2h_last"] = f_h2last
    NEW["elo_delta_home"] = f_edh; NEW["elo_delta_away"] = f_eda
    NEW["congestion_21d_home"] = f_cgh; NEW["congestion_21d_away"] = f_cga
    NEW["xg_for5_home"] = f_xgh_f; NEW["xg_against5_home"] = f_xgh_a
    NEW["xg_for5_away"] = f_xga_f; NEW["xg_against5_away"] = f_xga_a

    ref_cols = ["referee_career_matches", "referee_career_avg_yellows",
                "referee_career_avg_reds", "referee_career_home_win_pct",
                "referee_season_matches", "referee_season_avg_yellows"]
    ref = {c: NUM(df[c]) for c in ref_cols}
    NEW["ref_known"] = ref["referee_career_matches"].notna().astype(np.float32)
    NEW["ref_matches"] = ref["referee_career_matches"]
    NEW["ref_yellows"] = ref["referee_career_avg_yellows"]
    NEW["ref_reds"] = ref["referee_career_avg_reds"]
    NEW["ref_homewin"] = ref["referee_career_home_win_pct"]
    NEW["ref_seas_matches"] = ref["referee_season_matches"]
    NEW["ref_seas_yellows"] = ref["referee_season_avg_yellows"]

    # --------------------------- market block -----------------------------
    MK = pd.DataFrame(index=df.index)

    ah_line = NUM(df["AHh"])
    for hc, ac in (("PAHH", "PAHA"), ("B365AHH", "B365AHA"), ("AvgAHH", "AvgAHA")):
        ph = NUM(df[hc]); pa = NUM(df[ac])
        need = MK.get("mk_ah_home", pd.Series(np.nan, index=df.index)).isna() & ph.notna() & pa.notna()
        base = MK.get("mk_ah_home", pd.Series(np.nan, index=df.index)).copy()
        base[need] = (1.0 / ph[need]) / (1.0 / ph[need] + 1.0 / pa[need])
        MK["mk_ah_home"] = base
    MK["mk_ah_known"] = MK["mk_ah_home"].notna().astype(np.float32)
    MK["mk_ah_line"] = ah_line
    MK["mk_ah_abs"] = ah_line.abs()

    for oc, uc in (("P>2.5", "P<2.5"), ("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5")):
        po = NUM(df[oc]); pu = NUM(df[uc])
        need = MK.get("mk_over25", pd.Series(np.nan, index=df.index)).isna() & po.notna() & pu.notna()
        base = MK.get("mk_over25", pd.Series(np.nan, index=df.index)).copy()
        base[need] = (1.0 / po[need]) / (1.0 / po[need] + 1.0 / pu[need])
        MK["mk_over25"] = base
    MK["mk_ou_known"] = MK["mk_over25"].notna().astype(np.float32)

    by = NUM(df["btts_yes_odds_betexplorer"]); bn = NUM(df["btts_no_odds_betexplorer"])
    MK["mk_btts"] = np.where(by.notna() & bn.notna(),
                             (1.0 / by) / (1.0 / by + 1.0 / bn), np.nan)
    MK["mk_btts_known"] = MK["mk_btts"].notna().astype(np.float32)

    bf_h = NUM(df["BFH"]); bf_d = NUM(df["BFD"]); bf_a = NUM(df["BFA"])
    bfv = bf_h.notna() & bf_d.notna() & bf_a.notna() & (bf_h > 1) & (bf_d > 1) & (bf_a > 1)
    inv_sum = 1.0 / bf_h[bfv] + 1.0 / bf_d[bfv] + 1.0 / bf_a[bfv]
    MK["mk_bf_H"] = np.nan; MK["mk_bf_D"] = np.nan; MK["mk_bf_A"] = np.nan
    MK.loc[bfv, "mk_bf_H"] = (1.0 / bf_h[bfv]) / inv_sum
    MK.loc[bfv, "mk_bf_D"] = (1.0 / bf_d[bfv]) / inv_sum
    MK.loc[bfv, "mk_bf_A"] = (1.0 / bf_a[bfv]) / inv_sum
    MK["mk_bf_known"] = bfv.astype(float)

    BOOKS = [("B365", "B365H", "B365D", "B365A"), ("BW", "BWH", "BWD", "BWA"),
             ("IW", "IWH", "IWD", "IWA"), ("PS", "PSH", "PSD", "PSA"),
             ("WH", "WHH", "WHD", "WHA"), ("VC", "VCH", "VCD", "VCA"),
             ("LB", "LBH", "LBD", "LBA"), ("CL", "CLH", "CLD", "CLA")]
    book_odds = {name: np.column_stack([NUM(df[h]).to_numpy(dtype=float),
                                        NUM(df[d]).to_numpy(dtype=float),
                                        NUM(df[a]).to_numpy(dtype=float)])
                 for name, h, d, a in BOOKS}
    for ci, cls in enumerate(["H", "D", "A"]):
        stack = np.column_stack([book_odds[b][:, ci] for b in book_odds])
        valid = stack > 1.0
        cnt = valid.sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mx = np.where(cnt > 0, np.nanmax(np.where(valid, stack, -np.inf), axis=1), np.nan)
            mn = np.where(cnt > 1, np.nanmin(np.where(valid, stack, np.inf), axis=1), np.nan)
            mean = np.where(cnt > 0, np.nanmean(np.where(valid, stack, np.nan), axis=1), np.nan)
        spread = np.where(cnt > 1, (mx - mn) / mean, np.nan)
        MK[f"mk_spread_{cls}"] = spread
    MK["mk_n_books"] = np.sum(np.column_stack(
        [(book_odds[b][:, 0] > 1.0) for b in book_odds]), axis=1).astype(float)

    X_B_XL = pd.concat([X_B.reset_index(drop=True),
                        NEW.astype(np.float32),
                        MK.astype(np.float32)], axis=1).astype(np.float32)

    # reindex to the exact model feature list (fixed 135 columns, order-safe)
    feats = FEATURE_COLUMNS
    missing = [c for c in feats if c not in X_B_XL.columns]
    if missing:
        raise RuntimeError(f"feature builder missing columns: {missing[:5]}...")
    X_B_XL = X_B_XL.reindex(columns=feats).astype(np.float32)
    # one-hot columns are structural zeros, never NaN, when a category is absent
    dummy_cols = [c for c in feats if c.startswith("c_") or c.startswith("src_")]
    X_B_XL[dummy_cols] = X_B_XL[dummy_cols].fillna(0.0)

    meta = pd.DataFrame({
        "date": df["match_date"].dt.strftime("%Y-%m-%d"),
        "home": df["home_team"],
        "away": df["away_team"],
        "competition": df["competition"],
        "country": df["Country"],
        "goals_h": df["full_time_home_goals"],
        "goals_a": df["full_time_away_goals"],
    })

    payload = {
        "X_B_XL": X_B_XL, "y": y,
        "date": df["match_date"].to_numpy(),
        "odds_best": odds_best, "odds_pinnacle": odds_pinnacle,
        "has_odds": has_odds, "src": src, "meta": meta,
    }

    # -------- market odds surfacing (odds features + reference odds) ------
    # First available PAIR in the chain (Pinnacle -> B365 -> Avg) per market;
    # pair integrity matters (mixing books would fabricate EV). The 1X2
    # odds_best keeps its validated fallback-chain semantics.
    def _pair_chain(pairs):
        o1 = np.full(n, np.nan)
        o2 = np.full(n, np.nan)
        for c1, c2 in pairs:
            a = NUM(df[c1]).to_numpy(dtype=float)
            b = NUM(df[c2]).to_numpy(dtype=float) if c2 else np.ones(n)
            ok = (a > 1.0) & np.isfinite(a) & np.isfinite(b)
            if c2:
                ok &= b > 1.0
            need = np.isnan(o1) & ok
            o1[need] = a[need]
            o2[need] = b[need]
        return o1, o2

    ov, un = _pair_chain([("P>2.5", "P<2.5"), ("B365>2.5", "B365<2.5"),
                          ("Avg>2.5", "Avg<2.5")])
    payload["odds_ou25"] = np.column_stack([ov, un]).astype(np.float32)
    by_, bn_ = _pair_chain([("btts_yes_odds_betexplorer", "btts_no_odds_betexplorer")])
    payload["odds_btts"] = np.column_stack([by_, bn_]).astype(np.float32)
    alh = NUM(df["AHh"]).to_numpy(dtype=float)
    ah_h, ah_a = _pair_chain([("PAHH", "PAHA"), ("B365AHH", "B365AHA"),
                              ("AvgAHH", "AvgAHA")])
    payload["odds_ah"] = np.column_stack([alh, ah_h, ah_a]).astype(np.float32)
    if frame is None and cache_key and all(p.exists() for p in csv_paths):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_DIR / f"features_{cache_key}.pkl", "wb") as f:
                pickle.dump(payload, f, protocol=4)
        except Exception:                                      # noqa: BLE001
            pass
    return payload


def discover_csvs():
    """CSVs in ./upload/ (the master-training files), sorted by name."""
    if not UPLOAD_DIR.exists():
        return []
    return sorted(str(p) for p in UPLOAD_DIR.glob("*.csv"))


# feature column list: loaded from the extracted champion metadata (the
# single source of truth the models were trained with); auto-extracts the
# bundles first when the kit has not been unpacked yet
def _load_feature_columns():
    p = MODELS_DIR / "champion_meta.json"
    if not p.exists():
        try:
            ensure_extracted(verbose=False)
        except Exception:                                     # noqa: BLE001
            return None
    if p.exists():
        return list(json.loads(p.read_text())["feature_columns"])
    return None


FEATURE_COLUMNS = _load_feature_columns()


# ============================================================================
# Soft-factor helpers (stdlib + numpy only)
# ============================================================================
def _sf_float(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _parse_ts(value):
    """Best-effort ISO-8601 parser -> aware datetime (UTC) or None."""
    import datetime as _dt
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        d = _dt.datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def _norm_tier(t):
    s = str(t or "").strip().upper()
    return s if s in _SOURCE_TIER_ALIASES else "T5"


_MULTI_PART_TLD = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au",
                   "co.jp", "ne.jp", "com.br", "com.tr", "com.cn", "com.mx",
                   "co.za", "co.kr"}
_PLACEHOLDER_HOSTS = {"example.com", "example.org", "example.net", "test.com",
                      "test.org", "localhost", "domain.com", "website.com",
                      "your-source.com", "source.com", "url.com", "foo.com",
                      "bar.com", "somewhere.com", "insert-url.com"}


def _url_domain(url):
    """Registrable domain of a source URL, with placeholder detection.

    Returns (domain, ok). ok is False for missing/malformed/placeholder
    URLs - the engine-side enforcement of "do not fabricate URLs".
    """
    from urllib.parse import urlparse
    raw = str(url or "").strip()
    if not raw or " " in raw or raw.startswith("#"):
        return None, False
    try:
        p = urlparse(raw if "://" in raw else "https://" + raw)
        host = (p.netloc or "").lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        if not host or "." not in host:
            return None, False
        if host in _PLACEHOLDER_HOSTS:
            return None, False
        labels = host.split(".")
        if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_PART_TLD:
            return ".".join(labels[-3:]), True
        if len(labels) >= 2:
            return ".".join(labels[-2:]), True
        return host, True
    except Exception:                                     # noqa: BLE001
        return None, False


def _norm_age(age):
    """Validated age in days, or None when missing/implausible."""
    x = _sf_float(age, float("nan"))
    if not np.isfinite(x) or x < 0 or x > 400:
        return None
    return float(x)


def _sf_recency(age_days):
    """Recency decay, linearly interpolated between the measured anchors."""
    if age_days is None or not np.isfinite(float(age_days)):
        return 0.30
    a = float(age_days)
    if a <= 0.0:
        return 1.00
    anchors = [(0.0, 1.00), (3.0, 1.00), (7.0, 0.85), (14.0, 0.65), (30.0, 0.30)]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if a <= x1:
            return y0 + (y1 - y0) * (a - x0) / (x1 - x0)
    return 0.30


def _sf_corroboration(n_independent):
    """Corroboration multiplier over INDEPENDENT credible source units."""
    if n_independent >= 3:
        return 1.20
    if n_independent == 2:
        return 1.00
    return 0.70


_TOKEN_RE = None


def _snippet_tokens(text):
    """Informative token set of a quoted snippet (>=4 chars, lowercased)."""
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re
        _TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _corroboration_units(credible):
    """RULE 24: independent corroboration units after syndication dedup.

    Two-stage: one representative per registrable domain, then greedy
    snippet clustering (token-Jaccard >= SYNDICATION_JACCARD, both snippets
    >= 8 informative tokens). Returns (n_units, clusters).
    """
    reps, seen = [], set()
    for s in credible:
        if s.get("_domain") in seen:
            continue
        seen.add(s.get("_domain"))
        reps.append(s)
    clusters, seeds = [], []
    for s in reps:
        toks = _snippet_tokens(s.get("snippet"))
        eligible = len(toks) >= 8
        placed = False
        if eligible:
            for ci, seed in enumerate(seeds):
                union = toks | seed
                if union and len(toks & seed) / len(union) >= SYNDICATION_JACCARD:
                    clusters[ci].append(0)
                    placed = True
                    break
        if not placed:
            clusters.append([0])
            seeds.append(toks)
    return len(clusters), clusters


# ============================================================================
# SoftFactorProcessor - verifies, multiplies, caps, coherences and nets an
# LLM-compiled Adjustment Register onto the model baselines.
# ============================================================================
class SoftFactorProcessor:
    """Pipeline (fail-closed verification -> multiplier chain -> collisions
    -> cross-market coherence -> caps -> netting -> flags -> shadow log):

      1. RULE 16 verification per entry against the code-side MODULE_CATALOG
         (known module, ADJUST role, allowed market, delta within range,
         RULE 0 satisfied, credible-source floor after RULE 24 dedup,
         T4/T5 sole-basis rejection, URL/snippet sanity)
      2. effective delta = delta_pp x tier x recency x corroboration
         x priced_in x model_overlap (credible sources only)
      3. collision check keyed on the CATALOG causal family
      4. RULE 22 cross-market coherence (DC mechanical cap + shared-signal
         haircuts over the extended WF pair set)
      5. RULE 10 cap ladder (measured, tier-conditional) + total drift clip
      6. observed line-move netting + measured transfer beta
      7. FSC flags + data-integrity gates -> NO_BET
      8. shadow-log record for offline calibration
    """

    MARKET_TARGETS = MARKET_TARGETS

    # ---------------------------------------------------------- sources
    @staticmethod
    def _entry_sources(entry):
        srcs = entry.get("sources") or []
        out = []
        for s in srcs:
            if not isinstance(s, dict):
                continue
            dom, ok = _url_domain(s.get("url"))
            if not ok or not str(s.get("snippet") or "").strip():
                continue
            out.append({**s, "_domain": dom, "_tier": _norm_tier(s.get("tier")),
                        "_age": _norm_age(s.get("age_days"))})
        return out

    # ------------------------------------------------------ verification
    @classmethod
    def verify_entry(cls, entry):
        """RULE 16 verification against the fixed module catalog."""
        if not isinstance(entry, dict):
            return False, "register entry is not a dict"
        if not entry.get("rule0_satisfied", False):
            return False, "RULE 0 not satisfied (data not collected live this turn)"
        module_id = str(entry.get("module_id", "") or "").strip().upper()
        cat = MODULE_CATALOG.get(module_id)
        if cat is None:
            return False, (f"unknown module_id {module_id or '(missing)'}: not in "
                           "the 31-module catalog (00_README PART S)")
        if cat["role"] != "ADJUST":
            return False, (f"{module_id} is FLAG_ONLY in the module catalog: "
                           "flags only, never a delta")
        market = norm_market(entry.get("market"))
        target = str(entry.get("target", "") or "").strip().lower()
        if market not in cls.MARKET_TARGETS or target not in cls.MARKET_TARGETS[market]:
            return False, (f"unknown market/target: {market}/{target} "
                           f"(allowed targets: {cls.MARKET_TARGETS.get(market)})")
        allowed = cat.get("markets")
        if allowed is not None and market not in allowed:
            return False, (f"{module_id} is restricted to markets {list(allowed)}; "
                           f"got {market}")
        srcs = cls._entry_sources(entry)
        if not srcs:
            return False, ("no usable source (each needs a real non-placeholder "
                           "URL and a quoted snippet)")
        credible = [s for s in srcs if s["_tier"] in CREDIBLE_TIERS and s["_domain"]]
        n_units, _clusters = _corroboration_units(credible)
        floor = EVIDENCE_FLOOR.get(cat["basis"], 3)
        if n_units < floor:
            return False, (f"RULE 16 floor not met: {n_units} independent "
                           f"corroboration unit(s) after syndication dedup, needs "
                           f"{floor} ({module_id} is {cat['basis']} in the catalog)")
        if all(s["_tier"] in ("T4", "T5") for s in srcs):
            return False, "T4/T5 sole basis: needs at least one T1-T3 corroboration"
        d = abs(_sf_float(entry.get("delta_pp"), 0.0))
        if not np.isfinite(d):
            return False, "delta_pp is not a finite number"
        if d > cat["max_pp"] + 1e-9:
            return False, (f"delta_pp {d:.1f} beyond {module_id} documented range "
                           f"(max {cat['max_pp']:.1f}pp)")
        return True, "verified"

    # --------------------------------------------------- effective delta
    @classmethod
    def effective_delta_parts(cls, entry, priced_in=1.00):
        """Signed effective delta in pp + the full multiplier breakdown."""
        parts = {}
        srcs = cls._entry_sources(entry)
        credible = [s for s in srcs if s["_tier"] in CREDIBLE_TIERS and s["_domain"]]
        if not credible:
            return 0.0, {"note": "no credible sources"}
        module_id = str(entry.get("module_id", "") or "").strip().upper()
        is_stats = module_id in STATS_MODULES
        tier_w = max((SOURCE_TIER_WEIGHTS[s["_tier"]]
                      if not (s["_tier"] == "T2_DATA" and not is_stats)
                      else SOURCE_TIER_WEIGHTS["T2"]) for s in credible)
        n_units, clusters = _corroboration_units(credible)
        corrob = _sf_corroboration(n_units)
        ages = [s["_age"] for s in credible if s["_age"] is not None]
        recency = _sf_recency(min(ages)) if ages else 0.30
        overlap = MODEL_OVERLAP_DISCOUNT.get(
            MODULE_CATALOG.get(module_id, {}).get("family", ""),
            MODEL_OVERLAP_DEFAULT) if module_id else MODEL_OVERLAP_DEFAULT
        incremental = bool(entry.get("model_incremental", False))
        if incremental and overlap < 1.0:
            overlap = 1.00
        d = _sf_float(entry.get("delta_pp"), 0.0)
        if str(entry.get("direction", "UP")).upper() == "DOWN":
            d = -abs(d)
        else:
            d = abs(d)
        val = d * tier_w * recency * corrob * float(priced_in) * overlap
        parts = {
            "tier_weight": round(tier_w, 3),
            "best_tier": max((s["_tier"] for s in credible),
                             key=lambda t: SOURCE_TIER_WEIGHTS.get(t, 0.0)),
            "t2_data_as_stats": is_stats,
            "recency": round(recency, 4),
            "freshest_age_days": round(min(ages), 2) if ages else None,
            "corroboration": corrob, "n_credible_sources": len(credible),
            "n_corroboration_units": n_units,
            "syndication_clusters": [len(c) for c in clusters],
            "priced_in": float(priced_in),
            "model_overlap": overlap, "model_incremental_claimed": incremental,
            "claim_pp": d,
        }
        return val, parts

    # --------------------------------------------------------- RULE 10
    @staticmethod
    def rule10_cap(delta_pp, tier, entry, phase6):
        """Tier-conditional magnitude caps (RULE 10, measured ladder)."""
        full, b2, b3 = RULE10_BANDS.get(tier, RULE10_BANDS[DEFAULT_LEAGUE_TIER])
        d = float(delta_pp)
        mag = abs(d)
        sign = 1.0 if d >= 0 else -1.0
        phase6 = str(entry.get("phase6") or phase6 or "NOT_RUN").strip().upper()
        if "_fresh" in entry:
            fresh = int(entry["_fresh"])
        else:
            fresh = sum(1 for s in SoftFactorProcessor._entry_sources(entry)
                        if s["_tier"] in CREDIBLE_TIERS and s["_age"] is not None
                        and s["_age"] <= 3.0)
        if phase6 == "FAILED":
            return 0.0, "REJECTED (Phase 6 FAILED adversarial review)"
        if mag < full:
            return d, "APPLIED_FULL"
        if mag <= b2:
            if fresh >= 2:
                return d, "APPLIED_FULL (2+ fresh sourced points)"
            return sign * full, f"CAPPED to {full:g}pp (single/dated source)"
        if mag <= b3:
            if phase6 == "SURVIVES":
                if fresh >= 2:
                    return 0.5 * d, "APPLIED_50 (Phase 6 survived)"
                applied = sign * min(mag * 0.5, full)
                return applied, (f"APPLIED_50 capped at {full:g}pp (Phase 6 "
                                 "survived but fewer than 2 fresh sourced points)")
            return sign * full, f"CAPPED to {full:g}pp (Phase 6 {phase6 or 'NOT_RUN'})"
        return 0.0, "REJECTED (beyond max band -> DIVERGENCE_UNRESOLVED)"

    # ------------------------------------------------------ league tier
    @staticmethod
    def _league_tier(soft_layer, competition):
        warns = []
        comp = str(competition or "")
        engine_tier = None
        for t, leagues in LEAGUE_TIERS.items():
            if comp in leagues:
                engine_tier = t
                break
        declared = None
        if soft_layer:
            raw = str(soft_layer.get("league_tier", "") or "").strip().upper()
            if raw == "D":
                declared = "C"
                warns.append("league_tier D mapped to C (the cap table has no D row)")
            elif raw in RULE10_BANDS:
                declared = raw
        if engine_tier and declared and declared != engine_tier:
            stricter = min(engine_tier, declared, key=lambda t: RULE10_BANDS[t][2])
            warns.append(f"league_tier mismatch: register declared {declared}, "
                         f"engine table says {engine_tier} for {comp!r}; using "
                         f"stricter tier {stricter}")
            return stricter, warns
        if engine_tier:
            return engine_tier, warns
        if declared:
            return declared, warns
        return DEFAULT_LEAGUE_TIER, warns

    # -------------------------------------------------------- priced-in
    @staticmethod
    def _priced_in_discount(entry, odds_ts, now_ts):
        """RULE 20: explicit declaration > three-way auto-detection > UNKNOWN."""
        declared = str(entry.get("market_priced_in", "") or "").strip().upper()
        if declared in PRICED_IN_DISCOUNT:
            return PRICED_IN_DISCOUNT[declared], declared
        if odds_ts is not None and now_ts is not None:
            srcs = SoftFactorProcessor._entry_sources(entry)
            ages = [s["_age"] for s in srcs
                    if s["_tier"] in CREDIBLE_TIERS and s["_age"] is not None]
            if ages:
                import datetime as _dt
                freshest_pub = now_ts - _dt.timedelta(days=min(ages))
                if odds_ts >= freshest_pub + _dt.timedelta(hours=6):
                    return PRICED_IN_DISCOUNT["YES"], "AUTO_YES"
                if odds_ts >= freshest_pub - _dt.timedelta(hours=6):
                    return PRICED_IN_DISCOUNT["PARTIAL"], "AUTO_PARTIAL"
                return PRICED_IN_DISCOUNT["NO"], "AUTO_NO"
        return 1.00, "UNKNOWN"

    @staticmethod
    def _flag_has_source(f):
        for s in (f.get("sources") or []):
            if isinstance(s, dict) and _url_domain(s.get("url"))[1]:
                return True
        return False

    # ---------------------------------------------------- main entry
    def process(self, soft_layer, baselines, competition,
                kickoff_utc=None, odds_as_of=None, match_id=None):
        """Run the full audit. baselines: {market: {target: p}}. Returns the
        soft_factors result block (audited_delta_pp keyed 'market:target')."""
        kickoff_ts = _parse_ts(kickoff_utc)
        odds_ts = _parse_ts(odds_as_of)
        import datetime as _dt
        now_ts = _dt.datetime.now(_dt.timezone.utc)
        hours_to_kickoff = None
        if kickoff_ts is not None:
            hours_to_kickoff = max(0.0, (kickoff_ts - now_ts).total_seconds() / 3600.0)

        result = {
            "applied": bool(soft_layer), "engine_version": ENGINE_VERSION,
            "league_tier": None, "entries_total": 0, "entries_verified": 0,
            "verification": [], "collisions_removed": [], "adjustments": [],
            "audited_delta_pp": {}, "uncapped_delta_pp": {},
            "uncapped_probabilities": {}, "uncapped_note": None,
            "collisions_dropped_pp": {}, "coherence": [],
            "flags": [], "flag_warnings": [], "fsc": None,
            "integrity_gates": None, "no_bet_reasons": [], "no_bet_markets": {},
            "warnings": [], "directional_conflicts": [], "line_move_netting": [],
            "register_verification_rate": None, "shadow_log": None,
        }
        if not soft_layer:
            tier, _w = self._league_tier(None, competition)
            result["league_tier"] = tier
            result["uncapped_note"] = (
                "No soft-factor register supplied: model probabilities only. "
                "Compile the 31-module audit (00_README PART S) and pass it "
                "via --soft-register.")
            return result

        tier, tier_warns = self._league_tier(soft_layer, competition)
        result["league_tier"] = tier
        result["warnings"].extend(tier_warns)

        phase6 = str(soft_layer.get("phase6_status", "NOT_RUN")).upper()
        register = soft_layer.get("adjustment_register") or []
        result["entries_total"] = len(register)

        # M9E weather-forecast-confidence gate (enforced)
        m9_gated = any(
            isinstance(f, dict)
            and str(f.get("module_id", "") or "").strip().upper() == "M9E"
            and int(min(5.0, max(0.0, _sf_float(f.get("severity"), 0.0)))) >= 2
            for f in (soft_layer.get("flags") or []) if isinstance(f, dict))
        if m9_gated:
            result["warnings"].append(
                "M9E weather-forecast-confidence flag raised (sev >= 2): all M9 "
                "weather register deltas are gated to zero")

        # 1-2) verify + catalog enforcement + effective deltas
        verified = []
        for entry in register:
            ok, reason = self.verify_entry(entry)
            if not ok:
                result["verification"].append({
                    "module": entry.get("module_id") if isinstance(entry, dict) else None,
                    "market": entry.get("market") if isinstance(entry, dict) else None,
                    "target": entry.get("target") if isinstance(entry, dict) else None,
                    "status": "REJECTED", "reason": reason})
                continue
            module_id = str(entry.get("module_id", "")).strip().upper()
            cat = MODULE_CATALOG[module_id]
            declared_basis = str(entry.get("evidence_basis", "") or "").strip().upper()
            if declared_basis and declared_basis != cat["basis"]:
                result["warnings"].append(
                    f"{module_id}: self-reported evidence_basis {declared_basis} "
                    f"ignored; catalog says {cat['basis']}")
            declared_family = str(entry.get("causal_family", "") or "").strip().lower()
            if declared_family and declared_family != cat["family"]:
                result["warnings"].append(
                    f"{module_id}: self-reported causal_family {declared_family!r} "
                    f"ignored; catalog family {cat['family']!r} used for collisions")
            discount, priced_in_state = self._priced_in_discount(entry, odds_ts, now_ts)
            eff, eff_parts = self.effective_delta_parts(entry, priced_in=discount)
            if (eff_parts.get("model_incremental_claimed")
                    and eff_parts.get("model_overlap", 1.0) >= 1.0
                    and cat["family"] in MODEL_OVERLAP_DISCOUNT):
                result["warnings"].append(
                    f"{module_id}: model_incremental=true waived the model-overlap "
                    f"discount ({cat['family']} family) - the shadow ledger will "
                    "measure whether this claim really carries new information")
            if hours_to_kickoff is not None and cat["family"] in KICKOFF_SENSITIVE_FAMILIES:
                srcs = self._entry_sources(entry)
                ages = [s["_age"] for s in srcs
                        if s["_tier"] in CREDIBLE_TIERS and s["_age"] is not None]
                if ages:
                    lead_days = min(ages) + hours_to_kickoff / 24.0
                    lead_rec = _sf_recency(lead_days)
                    age_rec = _sf_recency(min(ages))
                    if lead_rec < age_rec:
                        eff *= lead_rec / max(1e-9, age_rec)
                        result["warnings"].append(
                            f"{module_id}: kickoff is {hours_to_kickoff:.0f}h away; "
                            f"lead-time decay applied ({min(ages):.0f}d-old news, "
                            f"{lead_days:.0f}d before kickoff)")
            fresh = sum(1 for s in self._entry_sources(entry)
                        if s["_tier"] in CREDIBLE_TIERS and s["_age"] is not None
                        and s["_age"] <= 3.0)
            verified.append({**entry, "effective_delta_pp": eff,
                             "_module": module_id, "_family": cat["family"],
                             "_fresh": fresh, "_priced_in": priced_in_state,
                             "_parts": eff_parts})
            result["entries_verified"] += 1

        if verified and odds_ts is None:
            result["warnings"].append(
                "odds_as_of not supplied: RULE 20 priced-in auto-detection is "
                "disabled; supply --odds-as-of to net out already-absorbed news")

        # 3) collision check on the CATALOG causal family
        by_key = {}
        for entry in verified:
            mkey = f"{norm_market(entry.get('market'))}:{entry.get('target')}"
            key = (entry["_family"], mkey)
            cur = by_key.get(key)
            if cur is None or abs(entry["effective_delta_pp"]) > abs(cur["effective_delta_pp"]):
                if cur is not None:
                    result["collisions_removed"].append({
                        "module": cur.get("module_id"), "causal_family": key[0],
                        "target": key[1],
                        "dropped_delta_pp": round(cur["effective_delta_pp"], 2)})
                by_key[key] = entry
            else:
                result["collisions_removed"].append({
                    "module": entry.get("module_id"), "causal_family": key[0],
                    "target": key[1],
                    "dropped_delta_pp": round(entry["effective_delta_pp"], 2)})

        for c in result["collisions_removed"]:
            t = c["target"]
            result["collisions_dropped_pp"][t] = round(
                result["collisions_dropped_pp"].get(t, 0.0) + c["dropped_delta_pp"], 2)

        # 4) RULE 22a: DC mechanical coherence vs the same family's 1X2 move
        def _base(market, target):
            v = baselines.get(market, {}).get(target)
            return float(v) if v is not None else float("nan")

        p_home_b = _base("x12", "home_win")
        p_draw_b = _base("x12", "draw")
        p_away_b = _base("x12", "away_win")
        families = {e["_family"] for e in by_key.values()}
        for fam in sorted(families):
            for dc_market, comp_target, comp_p in (
                    ("dc1x", "home_win", p_home_b), ("dcx2", "away_win", p_away_b)):
                dc_entry = by_key.get((fam, f"{dc_market}:win"))
                if dc_entry is None:
                    continue
                d_comp = by_key.get((fam, f"x12:{comp_target}"))
                d_draw = by_key.get((fam, "x12:draw"))
                d_c = d_comp["effective_delta_pp"] if d_comp is not None else 0.0
                d_d = d_draw["effective_delta_pp"] if d_draw is not None else 0.0
                if abs(d_c) < 1e-9 and abs(d_d) < 1e-9:
                    continue
                if (not (np.isfinite(comp_p) and np.isfinite(p_draw_b))
                        or (comp_p + p_draw_b) <= 1e-9):
                    continue
                implied = (comp_p * d_c + p_draw_b * d_d) / (comp_p + p_draw_b)
                allowed = abs(implied) + DC_COHERENCE_TOLERANCE_PP
                if abs(dc_entry["effective_delta_pp"]) > allowed > 0:
                    old = dc_entry["effective_delta_pp"]
                    dc_entry["effective_delta_pp"] = float(
                        np.copysign(allowed, dc_entry["effective_delta_pp"]))
                    result["coherence"].append({
                        "rule": "DC_MECHANICAL_CAP", "family": fam,
                        "target": f"{dc_market}:win", "before_pp": round(old, 2),
                        "after_pp": round(dc_entry["effective_delta_pp"], 2),
                        "mechanical_implied_pp": round(implied, 2)})
                    result["warnings"].append(
                        f"RULE 22: {fam} {dc_market} delta clipped from {old:+.2f}pp "
                        f"to {dc_entry['effective_delta_pp']:+.2f}pp - the same "
                        "family already moved the underlying 1X2 (mechanical mix "
                        f"{implied:+.2f}pp)")

        # 4b) RULE 22b: shared-signal haircuts over the extended pair set
        for a_key, b_key in SHARED_SIGNAL_PAIRS:
            a_owner = next((e for (_fam, k), e in by_key.items() if k == a_key), None)
            b_owner = next((e for (_fam, k), e in by_key.items() if k == b_key), None)
            if a_owner is None or b_owner is None:
                continue
            if a_owner["_family"] != b_owner["_family"]:
                continue          # different families = independent evidence
            if ((a_owner["effective_delta_pp"] > 0) !=
                    (b_owner["effective_delta_pp"] > 0)):
                continue          # opposite directions are a conflict, not a share
            a_mag = abs(a_owner["effective_delta_pp"])
            b_mag = abs(b_owner["effective_delta_pp"])
            if a_mag < 1e-9 or b_mag < 1e-9:
                continue
            smaller, bigger = (a_owner, b_owner) if a_mag <= b_mag else (b_owner, a_owner)
            smaller_key = a_key if smaller is a_owner else b_key
            old = smaller["effective_delta_pp"]
            smaller["effective_delta_pp"] = old * SHARED_SIGNAL_HAIRCUT
            result["coherence"].append({
                "rule": "SHARED_SIGNAL_HAIRCUT", "family": smaller["_family"],
                "target": smaller_key, "before_pp": round(old, 2),
                "after_pp": round(smaller["effective_delta_pp"], 2),
                "haircut": SHARED_SIGNAL_HAIRCUT})
            result["warnings"].append(
                f"RULE 22: {smaller['_family']} moved both {a_key} and {b_key} "
                f"(correlated pair); the smaller delta was haircut "
                f"x{SHARED_SIGNAL_HAIRCUT:g} so one signal is not claimed twice")

        # 5) RULE 10 caps on the audited path
        audited, uncapped = {}, {}
        for entry in by_key.values():
            market = norm_market(entry.get("market"))
            target = str(entry.get("target", "")).strip().lower()
            mkey = f"{market}:{target}"
            eff = entry["effective_delta_pp"]
            if m9_gated and entry["_module"] == "M9":
                capped, status = 0.0, "GATED (M9E forecast-confidence flag: M9 zeroed)"
            else:
                capped, status = self.rule10_cap(eff, tier, entry, phase6)
            rarity = claim_rarity_pct(market, entry.get("delta_pp"))
            result["adjustments"].append({
                "module": entry.get("module_id"),
                "module_name": entry.get("module_name"),
                "causal_family": entry["_family"],
                "market": market, "target": target,
                "direction": entry.get("direction"), "delta_pp": entry.get("delta_pp"),
                "effective_delta_pp": round(eff, 2),
                "audited_delta_pp": round(capped, 2), "status": status,
                "evidence_basis": MODULE_CATALOG[entry["_module"]]["basis"],
                "n_sources": len(self._entry_sources(entry)),
                "priced_in": entry["_priced_in"],
                "multipliers": entry.get("_parts", {}),
                "claim_rarity_pct": round(rarity * 100.0, 1) if rarity is not None else None,
                "claim_rarity_note": (None if rarity is None else
                                      f"a |{abs(_sf_float(entry.get('delta_pp'), 0.0)):.1f}pp claim "
                                      f"is a top-{rarity * 100.0:.1f}% event by real "
                                      f"{market} line-move standards"),
                "family_residual_pp": SOFT_LAYER_CALIBRATION["family_residual_pp"].get(
                    entry["_family"])})
            if "DIVERGENCE_UNRESOLVED" in status:
                result["no_bet_markets"].setdefault(market, []).append(
                    f"DIVERGENCE_UNRESOLVED: adjustment beyond the {tier}-tier cap "
                    f"was rejected outright -> NO_BET on {market}.")
            audited[mkey] = audited.get(mkey, 0.0) + capped
            if not (m9_gated and entry["_module"] == "M9"):
                uncapped[mkey] = uncapped.get(mkey, 0.0) + entry["effective_delta_pp"]

        # 5b) directional-conflict detection
        for mkey in audited:
            parts = [e for e in by_key.values()
                     if f"{norm_market(e.get('market'))}:{e.get('target')}" == mkey]
            net = sum(e["effective_delta_pp"] for e in parts)
            gross = sum(abs(e["effective_delta_pp"]) for e in parts)
            if gross >= 4.0 and gross >= 2.0 * max(1e-9, abs(net)):
                mods = ", ".join(sorted({str(e.get("module_id")) for e in parts}))
                result["directional_conflicts"].append({
                    "target": mkey, "net_pp": round(net, 2),
                    "gross_pp": round(gross, 2), "modules": mods})
                result["warnings"].append(
                    f"directional conflict on {mkey}: gross {gross:.1f}pp of verified "
                    f"adjustments cancels to net {net:+.1f}pp ({mods}) - the soft "
                    "layer is undecided on this target; treat with caution")

        # 5c) total drift clip at the tier's band-3 maximum
        full, b2, b3 = RULE10_BANDS.get(tier, RULE10_BANDS[DEFAULT_LEAGUE_TIER])
        for k, v in list(audited.items()):
            if abs(v) > b3:
                audited[k] = float(np.copysign(b3, v))
                result["adjustments"].append({
                    "module": "TOTAL", "market": k.split(":")[0],
                    "target": k.split(":")[1],
                    "status": f"TOTAL CAPPED at {b3:g}pp"})
                result["warnings"].append(
                    f"total audited drift on {k} exceeded the {tier}-tier maximum "
                    f"({b3:g}pp) and was clipped: check the register is not telling "
                    "one story through several causal families")

        # 5d) observed line-move netting (RULE 20)
        moves = soft_layer.get("observed_line_moves") or {}
        if isinstance(moves, dict):
            for mkey, mv in moves.items():
                m = _sf_float(mv, float("nan"))
                if not np.isfinite(m) or abs(m) < 1e-9 or mkey not in audited:
                    continue
                a = audited[mkey]
                if (a > 0) == (m > 0):
                    netted = float(np.copysign(max(0.0, abs(a) - abs(m)), a))
                    result["line_move_netting"].append({
                        "target": mkey, "audited_pp": round(a, 2),
                        "observed_move_pp": round(m, 2), "applied_pp": round(netted, 2)})
                    audited[mkey] = netted
                else:
                    result["warnings"].append(
                        f"observed line move {m:+.1f}pp on {mkey} OPPOSES the "
                        f"audited adjustment {a:+.1f}pp: the market disagrees "
                        "with the soft layer here")

        # 5e) measured transfer beta (parity ceiling 1.0, refittable)
        result["transfer"] = {}
        for mkey in list(audited.keys()):
            market = mkey.split(":", 1)[0]
            beta_m = float(SOFT_LAYER_CALIBRATION["transfer_beta"].get(market, 1.0))
            if abs(audited[mkey]) > 1e-9 and abs(beta_m - 1.0) > 1e-9:
                result["transfer"][mkey] = {
                    "beta": beta_m, "pre_transfer_pp": round(audited[mkey], 2),
                    "applied_pp": round(audited[mkey] * beta_m, 2)}
                audited[mkey] = audited[mkey] * beta_m

        result["audited_delta_pp"] = {k: round(v, 2) for k, v in audited.items()}
        result["uncapped_delta_pp"] = {k: round(v, 2) for k, v in uncapped.items()}
        result["uncapped_note"] = (
            "INFO ONLY - uncapped view of the soft-factor layer (collision "
            "survivors; dropped-duplicate mass per target is in "
            "collisions_dropped_pp). NOT used for picks, gates, or eligibility.")
        for mkey, dpp in uncapped.items():
            market, target = mkey.split(":", 1)
            base = baselines.get(market, {}).get(target)
            if base is not None and np.isfinite(float(base)):
                result["uncapped_probabilities"][mkey] = round(
                    min(0.99, max(0.01, float(base) + dpp / 100.0)), 4)

        # 6) FSC flags: validated, typed, scoped
        flags = soft_layer.get("flags") or []
        processed_flags = []
        fsc_per_market = {m: 0 for m in self.MARKET_TARGETS}
        sev5_markets = set()
        total_declared_sev = 0
        if isinstance(flags, list):
            for f in flags:
                if not isinstance(f, dict):
                    continue
                sev = int(min(5.0, max(0.0, _sf_float(f.get("severity"), 0.0))))
                name = str(f.get("name") or "(unnamed)")
                module_id = str(f.get("module_id", "") or "").strip().upper()
                if module_id not in MODULE_CATALOG and sev > 2:
                    result["flag_warnings"].append(
                        f"flag {name!r} has no valid module_id; severity clamped "
                        f"to 2 (was {sev})")
                    sev = 2
                ftype = str(f.get("type", "") or "").strip().upper()
                if ftype not in ("VARIANCE", "DIRECTIONAL"):
                    ftype = "DIRECTIONAL"
                if sev >= 4 and not self._flag_has_source(f):
                    result["flag_warnings"].append(
                        f"flag {name!r} severity {sev} needs at least one sourced "
                        "URL; severity clamped to 3")
                    sev = 3
                scope_raw = f.get("markets")
                if isinstance(scope_raw, list) and scope_raw:
                    scope = [norm_market(m) for m in scope_raw]
                    scope = [m for m in scope if m in self.MARKET_TARGETS]
                    scope = scope or list(self.MARKET_TARGETS)
                else:
                    scope = list(self.MARKET_TARGETS)
                contribution = int(np.ceil(sev / 2.0)) if ftype == "VARIANCE" else sev
                total_declared_sev += sev
                for mkt in scope:
                    fsc_per_market[mkt] += contribution
                    if sev >= 5:
                        sev5_markets.add(mkt)
                processed_flags.append({**f, "severity": sev, "type": ftype,
                                        "module_id": module_id or None,
                                        "markets": scope,
                                        "fsc_contribution": contribution})
        result["flags"] = processed_flags
        result["warnings"].extend(result["flag_warnings"])
        vetoed_markets = ({m for m, s in fsc_per_market.items() if s >= 10}
                          | sev5_markets)
        result["fsc"] = {
            "total_severity": total_declared_sev,
            "per_market_severity": dict(fsc_per_market),
            "any_severity_5_markets": sorted(sev5_markets),
            "no_bet_markets": sorted(vetoed_markets),
            "no_bet": bool(vetoed_markets),
        }
        for mkt in sorted(vetoed_markets):
            result["no_bet_markets"].setdefault(mkt, []).append(
                f"FSC veto: scoped flag severity on {mkt} reached the NO_BET threshold")

        # 7) data-integrity gates: self-reported + engine-measured
        integ = soft_layer.get("data_integrity") or {}
        if not isinstance(integ, dict):
            integ = {}
        vr = integ.get("verification_rate_pct")
        unknown = integ.get("unknown_data_points")
        total_dp = integ.get("total_data_points")
        gates = {"verification_rate_pct": vr, "unknown_data_points": unknown,
                 "total_data_points": total_dp}
        devils = False
        if vr is not None and _sf_float(vr, 100.0) < 50:
            devils = True
            result["no_bet_reasons"].append(
                f"DEVILS_VETO: verification rate {vr}% < 50%")
        if unknown is not None:
            unknown_f = _sf_float(unknown, 0.0)
            if total_dp is not None and _sf_float(total_dp, 0.0) > 0:
                threshold = max(5.0, 0.05 * _sf_float(total_dp, 0.0))
            else:
                threshold = 5.0
            if unknown_f > threshold:
                devils = True
                result["no_bet_reasons"].append(
                    f"DEVILS_VETO: {unknown} unverified data points > {threshold:.0f}")
        reg_rate = None
        if result["entries_total"] >= 4:
            reg_rate = round(result["entries_verified"] / result["entries_total"], 3)
            result["register_verification_rate"] = reg_rate
            if reg_rate < 0.5:
                devils = True
                result["no_bet_reasons"].append(
                    f"DEVILS_VETO: engine-measured register verification rate "
                    f"{reg_rate:.0%} < 50% ({result['entries_verified']}/"
                    f"{result['entries_total']} entries passed RULE 16)")
        gates["devils_veto"] = devils
        gates["register_verification_rate"] = reg_rate

        actual_counts = {"t1_t2_count": 0, "t3_count": 0, "t4_t5_count": 0}
        for entry in register:
            if not isinstance(entry, dict):
                continue
            for s in self._entry_sources(entry):
                t = s["_tier"]
                if t in ("T1", "T1_CLUB", "T2", "T2_DATA"):
                    actual_counts["t1_t2_count"] += 1
                elif t == "T3":
                    actual_counts["t3_count"] += 1
                else:
                    actual_counts["t4_t5_count"] += 1
        tier_count_mismatch = False
        for k, v in actual_counts.items():
            declared_v = integ.get(k)
            if declared_v is not None and abs(_sf_float(declared_v, 0.0) - v) > 2:
                tier_count_mismatch = True
                result["warnings"].append(
                    f"data_integrity.{k} self-reports {declared_v} but the register "
                    f"contains {v}: treat the integrity statement with suspicion")
        penalty = bool(vr is not None and 50 <= _sf_float(vr, 100.0) < 70)
        if reg_rate is not None and 0.5 <= reg_rate < 0.7:
            penalty = True
        if tier_count_mismatch:
            penalty = True
        gates["confidence_penalty"] = penalty
        gates["tier_count_mismatch"] = tier_count_mismatch
        result["integrity_gates"] = gates

        # 8) shadow-log record (offline calibration sample)
        shadow = {
            "match_id": match_id, "league_tier": tier,
            "engine_version": ENGINE_VERSION,
            "calibration_provenance": SOFT_LAYER_CALIBRATION["measured_on"],
            "entries_total": result["entries_total"],
            "entries_verified": result["entries_verified"],
            "register_verification_rate": reg_rate,
            "phase6_global": phase6,
            "fsc_no_bet_markets": sorted(vetoed_markets),
            "devils_veto": devils,
            "per_target": {}, "entries": [],
        }
        for mkey in sorted(set(list(audited.keys()) + list(uncapped.keys()))):
            market, target = mkey.split(":", 1)
            base = baselines.get(market, {}).get(target)
            a = audited.get(mkey, 0.0)
            u = uncapped.get(mkey, 0.0)
            shadow["per_target"][mkey] = {
                "baseline_p": round(float(base), 4) if base is not None else None,
                "audited_delta_pp": round(a, 2),
                "uncapped_delta_pp": round(u, 2),
                "audited_p": (round(min(0.99, max(0.01, float(base) + a / 100.0)), 4)
                              if base is not None and np.isfinite(float(base)) else None),
                "uncapped_p": (round(min(0.99, max(0.01, float(base) + u / 100.0)), 4)
                               if base is not None and np.isfinite(float(base)) else None),
                "transfer_beta": float(SOFT_LAYER_CALIBRATION["transfer_beta"].get(
                    market, 1.0)),
            }
        for entry in by_key.values():
            srcs = self._entry_sources(entry)
            credible = [s for s in srcs if s["_tier"] in CREDIBLE_TIERS and s["_domain"]]
            ages = [s["_age"] for s in credible if s["_age"] is not None]
            mkey = f"{norm_market(entry.get('market'))}:{entry.get('target')}"
            shadow["entries"].append({
                "module": entry.get("module_id"), "market_target": mkey,
                "family": entry["_family"],
                "basis": MODULE_CATALOG[entry["_module"]]["basis"],
                "claim_pp": entry.get("delta_pp"),
                "effective_delta_pp": round(entry["effective_delta_pp"], 2),
                "n_credible_sources": len({s["_domain"] for s in credible}),
                "best_tier": (max((s["_tier"] for s in credible),
                                  key=lambda t: SOURCE_TIER_WEIGHTS.get(t, 0.0))
                              if credible else None),
                "freshest_age_days": round(min(ages), 2) if ages else None,
                "priced_in_state": entry.get("_priced_in"),
                "multipliers": entry.get("_parts", {}),
            })
        result["shadow_log"] = shadow
        return result


# ============================================================================
# Application: audited deltas -> adjusted probability vectors, with FULL
# delta delivery (a moved target receives the whole audited delta; the
# complementary mass is removed proportionally) and overflow guards.
# ============================================================================
def apply_soft_factors(market_probs, audited_delta):
    """Apply audited deltas to per-market probability vectors.

    market_probs: {market: {"classes": [...], "p": [floats]}}
    audited_delta: {"market:target": pp}
    Returns {market: adjusted_p_list} (same shapes; None where missing).
    """
    adj = {m: list(v["p"]) for m, v in market_probs.items()}
    per_market = {}
    for mkey, dpp in audited_delta.items():
        try:
            market, target = mkey.split(":", 1)
        except ValueError:
            continue
        if market not in adj or market not in TARGETS_BY_CLASS:
            continue
        per_market.setdefault(market, {})[target] = \
            per_market.get(market, {}).get(target, 0.0) + dpp

    for market, deltas in per_market.items():
        p = adj.get(market)
        if p is None:
            continue
        if market == "ah":
            # deltas move cover_home; cover_away absorbs the change (push kept)
            d = sum(deltas.values()) / 100.0
            p[0] = min(0.99, max(0.01, p[0] + d))
            p[-1] = max(0.01, min(0.99, 1.0 - p[0]))
            continue
        tmap = TARGETS_BY_CLASS[market]
        if market in ("x12", "ht13"):
            # full delivery: target gets the whole delta; the complementary
            # mass is removed PROPORTIONALLY from the untouched outcomes
            touched = {}
            for ci, tgt in enumerate(tmap):
                if tgt in deltas:
                    touched[ci] = deltas[tgt] / 100.0
            if not touched:
                continue
            dsum = sum(touched.values())
            p_adj_sum = sum(float(p[ci]) for ci in touched)
            n_other = len(p) - len(touched)
            q_cap = 1.0 - 0.01 * n_other
            dsum_max = max(0.0, q_cap - p_adj_sum)
            if dsum > dsum_max and dsum > 1e-12:
                k = dsum_max / dsum
                touched = {ci: d * k for ci, d in touched.items()}
            q = {ci: min(0.98, max(0.01, float(p[ci]) + d))
                 for ci, d in touched.items()}
            q_sum = sum(q.values())
            other = [ci for ci in range(len(p)) if ci not in q]
            other_mass = max(0.0, 1.0 - q_sum)
            other_base = sum(float(p[ci]) for ci in other) if other else 0.0
            for ci in other:
                share = (float(p[ci]) / other_base) if other_base > 1e-9 \
                    else (1.0 / len(other))
                p[ci] = other_mass * share
            for ci, qv in q.items():
                p[ci] = qv
            s = sum(float(v) for v in p)
            if abs(s - 1.0) > 1e-9 and s > 0:
                big = int(np.argmax(p))
                p[big] += (1.0 - s)
            continue
        # binary + single-target markets: full delta on the target, the
        # complement absorbs it (per-target values carry the audit)
        for tgt, dpp in deltas.items():
            if tgt not in tmap:
                continue
            ci = tmap.index(tgt)
            new = min(0.99, max(0.01, float(p[ci]) + dpp / 100.0))
            p[ci] = new
            others = [j for j in range(len(p)) if j != ci]
            rest = 1.0 - new
            base_rest = sum(float(p[j]) for j in others)
            if len(others) == 1:
                p[others[0]] = rest
            elif base_rest > 1e-9:
                for j in others:
                    p[j] = rest * (float(p[j]) / base_rest)
            else:
                for j in others:
                    p[j] = rest / len(others)
    return {m: [min(0.995, max(0.005, float(v))) for v in p] for m, p in adj.items()}


# ============================================================================
# XAI - exact TreeSHAP per final pick (CatBoost-native, no extra deps)
# ============================================================================
def xai_shap_vector(model, xrow, class_idx=None):
    """Exact TreeSHAP contributions for one row.

    Returns (contribs, bias, feature_names) where contribs is a float vector
    in the model's margin space for the requested class (binary models take
    class_idx=1 semantics at the caller), or (None, None, None) on failure.
    Handles both CatBoost models (get_feature_importance) and LightGBM
    Boosters (pred_contrib) - htft / cs are LightGBM text models.
    """
    try:
        if "lightgbm" in (type(model).__module__ or ""):
            nf = int(model.num_feature())
            pc = model.predict(np.asarray(xrow, dtype=np.float32),
                               pred_contrib=True)
            sv = np.asarray(pc)
            if sv.ndim == 3:                   # some builds: (n, n_class, n+1)
                vec = sv[0, int(class_idx if class_idx is not None else 0)]
            elif sv.ndim == 2:
                ncls = sv.shape[1] // (nf + 1)
                if ncls > 1:                   # flat (n, n_class*(n+1)) layout
                    vec = sv[0].reshape(ncls, nf + 1)[
                        int(class_idx if class_idx is not None else 0)]
                else:                          # binary (n, n+1)
                    vec = sv[0]
            else:
                return None, None, None
            names = (list(FEATURE_COLUMNS) if nf == len(FEATURE_COLUMNS)
                     else [f"feature_{i}" for i in range(nf)])
            return vec[:-1].astype(float), float(vec[-1]), names
        from catboost import Pool
        pool = Pool(np.asarray(xrow, dtype=np.float32),
                    feature_names=list(FEATURE_COLUMNS))
        sv = np.asarray(model.get_feature_importance(pool, type="ShapValues"))
        if sv.ndim == 3:                       # multiclass (1, n_class, n_feat+1)
            vec = sv[0, int(class_idx if class_idx is not None else 0)]
        elif sv.ndim == 2:                     # binary (1, n_feat+1)
            vec = sv[0]
        else:
            return None, None, None
        return vec[:-1].astype(float), float(vec[-1]), list(FEATURE_COLUMNS)
    except Exception:                                      # noqa: BLE001
        return None, None, None


def xai_explain_bet(model, xrow, class_idx, top_k=5, invert=False):
    """TreeSHAP top drivers for one bet, in plain language + structured.

    invert=True flips the margin sign. CatBoost's ShapValues for a BINARY
    model explain the positive class only (yes / over), so a pick on the
    negative side (no / under) is explained by the negated margin - the
    summary then describes the picked side, not its opposite.
    """
    contribs, bias, names = xai_shap_vector(model, xrow, class_idx)
    if contribs is None:
        return None
    if invert:
        contribs = -contribs
        bias = -bias
    order = np.argsort(-np.abs(contribs))
    sup, agn = [], []
    for i in order:
        if len(sup) >= top_k and len(agn) >= top_k:
            break
        label = FEATURE_LABELS.get(names[i], names[i])
        rec = {"feature": names[i], "label": label,
               "contribution": round(float(contribs[i]), 4)}
        if contribs[i] >= 0 and len(sup) < top_k:
            sup.append(rec)
        elif contribs[i] < 0 and len(agn) < top_k:
            agn.append(rec)
    net = float(np.sum(contribs))
    drv = ", ".join(f"{r['label']} {r['contribution']:+.2f}" for r in sup[:3]) or "n/a"
    cnt = ", ".join(f"{r['label']} {r['contribution']:+.2f}" for r in agn[:2]) or "none"
    text = (f"TreeSHAP (exact): net margin {net:+.2f} toward the pick; top "
            f"drivers: {drv}; main counter-evidence: {cnt}.")
    native = ("LightGBM-native" if "lightgbm" in (type(model).__module__ or "")
              else "CatBoost-native")
    return {"method": f"TreeSHAP (exact, {native})",
            "units": "model margin points (log-odds scale)",
            "bias": round(bias, 4), "net_margin": round(net, 4),
            "top_support": sup, "top_against": agn, "summary": text}


def xai_compose(bet, xai, soft_adj, evidence_note):
    """The per-bet XAI summary: model drivers + soft-factor deltas + evidence."""
    lines = []
    if xai:
        lines.append("Model: " + xai["summary"])
    if soft_adj:
        parts = "; ".join(
            f"{a['module']} {a['market']}:{a['target']} "
            f"{a['audited_delta_pp']:+.1f}pp audited ({a['status'].split(' (')[0]}"
            + (f", priced-in {a['priced_in']}" if a.get("priced_in") else "") + ")"
            for a in soft_adj)
        lines.append(f"Soft layer ({bet['soft_total_pp']:+.1f}pp net): {parts}.")
    else:
        lines.append("Soft layer: no audited adjustment touched this market.")
    if evidence_note:
        lines.append("Evidence: " + evidence_note)
    lines.append("Gate check: p_final {:.4f} vs headline gate {:.2f} -> {}; "
                 "confidence rating: {}.".format(
                     bet["p_final"], bet["gate"],
                     bet.get("gate_label") or "n/a",
                     bet.get("confidence") or "n/a"))
    return " | ".join(lines)


# ============================================================================
# AI INTUITION BLOCK - reasoning scaffolding, deliberately NOT probability-
# derived. The engine compiles the narrative digest (what the AUDITED register
# claims say, what was rejected, what is priced in); the pick slot is left to
# the operator AI running the engine. Nothing in here reads p_final.
# ============================================================================
_INTUITION_SIDE = {"home_win": "HOME", "draw": "DRAW", "away_win": "AWAY"}


def _ai_intuition_block(sf, report=None):
    """Reasoning-based intuition scaffolding for one match.

    engine_lean  - the net direction of the AUDITED register narrative on the
                   1X2 targets (sourced claims after RULE 16/10/20/22), with
                   its drivers and the contra-evidence. This is a narrative
                   aggregate, not a probability: no p_final, no gate, no stake.
    operator_pick- the reserved slot the RUNNING AI completes with its own
                   reasoning-based pick (its "intuition"), explicitly not the
                   model's numbers.
    """
    drivers = []
    net = {}
    for a in (sf.get("adjustments") or []):
        if a.get("market") != "x12":
            continue
        tgt = a.get("target")
        pp = float(a.get("audited_delta_pp") or 0.0)
        net[tgt] = round(net.get(tgt, 0.0) + pp, 2)
        drivers.append({
            "module": a.get("module"), "target": tgt,
            "claim_pp": a.get("delta_pp"),
            "audited_pp": a.get("audited_delta_pp"),
            "status": str(a.get("status", "")).split(" (")[0],
            "priced_in": bool(a.get("priced_in")),
        })
    drivers.sort(key=lambda d: -abs(float(d.get("audited_pp") or 0.0)))
    rejected = [{"module": v.get("module"), "market": v.get("market"),
                 "target": v.get("target"), "reason": v.get("reason")}
                for v in (sf.get("verification") or [])]
    if net and max(net.values()) >= 1.0:
        tgt, pp = max(net.items(), key=lambda kv: kv[1])
        side = _INTUITION_SIDE.get(tgt, tgt).upper()
        strength = ("STRONG" if pp >= 4.0
                    else "MODERATE" if pp >= 2.0 else "FAINT")
    else:
        side, pp, strength = "NONE", 0.0, "NONE"
    fsc = sf.get("fsc") or {}
    fails = (report or {}).get("model_fails") or {}
    return {
        "kind": ("AI INTUITION - a reasoning lean, NOT a probability-"
                 "qualified pick; never derived from the model "
                 "probabilities or the gates"),
        "engine_lean": {
            "side": side, "net_pp": pp, "strength": strength,
            "basis": ("net audited register narrative on the 1X2 targets "
                      "(sourced claims after RULE 16/10/20/22)"),
            "net_by_target": net,
            "drivers": drivers[:5],
            "contra": {
                "rejected_claims": len(rejected),
                "rejected_detail": rejected[:5],
                "priced_in_claims": sum(1 for d in drivers if d["priced_in"]),
                "no_bet_markets": sorted(sf.get("no_bet_markets") or []),
                "fsc_veto": bool(fsc.get("no_bet")),
                "artifact_fails": len(fails),
            },
        },
        "operator_pick": {
            "pick": None, "reasoning": None,
            "instruction": (
                "The running AI completes this slot from its OWN reading of "
                "the sourced evidence (news, motivation, tactics, "
                "conditions, matchup) - NOT from the model probabilities. "
                "State one pick (a team/market lean), cite which register "
                "entries or facts drove it, and write 'no lean' if nothing "
                "justifies one."),
        },
    }


# ============================================================================
# Human-readable report (the readable companion written next to the JSON)
# ============================================================================
def match_report_text(rep):
    """Readable per-match report: probabilities, soft-factor audit, final
    picks with confidence ratings and XAI, the AI-intuition digest, notes."""
    L = []
    bar = "=" * 78
    L.append(bar)
    tag = "  ** SYNTHETIC DEMO **" if rep.get("demo") else ""
    warn = "  ** IN-SAMPLE **" if rep.get("in_sample_warning") else ""
    L.append(f"PARLAY ENGINE V1 - MATCH REPORT{tag}{warn}")
    L.append(f"{rep.get('date', '?')}  {rep.get('home', '?')} vs "
             f"{rep.get('away', '?')}  ({rep.get('competition', '?')})")
    if rep.get("result"):
        L.append(f"Result: {rep['result']}")
    L.append(bar)

    # [1] probabilities
    L.append("[1] MARKET PROBABILITIES (model -> final, after audited soft factors)")
    sf = rep.get("soft_factors") or {}
    audited = sf.get("audited_delta_pp") or {}
    L.append(f"  {'market':9s} {'pick':>6s} {'p_raw':>7s} {'soft_pp':>8s} "
             f"{'p_final':>8s} {'gate':>5s}  confidence")
    for mid in sorted(rep.get("markets", {})):
        blk = rep["markets"][mid]
        if blk.get("status") != "ok":
            L.append(f"  {mid:9s} FAIL: {str(blk.get('reason', '?'))[:56]}")
            continue
        fin = (sf.get("final_probabilities") or {}).get(mid) or blk["p"]
        dpp = sum(v for k, v in audited.items() if k.split(":")[0] == mid)
        gate = blk.get("gate")
        conf = (sf.get("final_confidence") or {}).get(mid) or blk.get("confidence")
        pick = blk.get("pick")
        if fin and blk.get("classes"):
            ci = int(np.argmax(fin))
            pick = blk["classes"][ci]
        rating = confidence_rating(conf, mid) if conf is not None else "-"
        L.append(f"  {mid:9s} {str(pick):>6s} {blk.get('confidence', 0):7.3f} "
                 f"{dpp:+8.2f} {(conf if conf is not None else 0):8.3f} "
                 f"{(gate if gate else 0):5.2f}  {rating}")

    # input coverage (single-match mode only)
    ic = rep.get("input_coverage")
    if ic:
        L.append("")
        L.append(f"  input coverage: {ic.get('n_filled', 0)}/135 model features "
                 f"filled ({ic.get('pct', 0):.0f}%)")
        for g, d in (ic.get("groups") or {}).items():
            L.append(f"    {g:22s} {d['filled']:>3d}/{d['total']:<3d} filled")
        for nte in (ic.get("notes") or []):
            L.append(f"    note: {nte}")

    # [2] soft-factor audit
    L.append("")
    L.append("[2] SOFT-FACTOR LAYER - AUDITED ADJUSTMENTS (RULE 16 verified, "
             "RULE 10 capped)")
    if sf.get("applied"):
        for v in sf.get("verification", []):
            L.append(f"  {str(v.get('module')):<6} {v.get('market')}:{v.get('target')}"
                     f"  REJECTED at verification - {v.get('reason')}")
        for a in sf.get("adjustments", []):
            mp = a.get("multipliers", {}) or {}
            L.append(f"  {str(a.get('module')):<6} {a.get('market')}:{a.get('target')}"
                     f"  claim {a.get('delta_pp')}pp -> effective "
                     f"{a.get('effective_delta_pp')}pp -> audited "
                     f"{a.get('audited_delta_pp')}pp  [{a.get('status')}]")
            L.append(f"         multipliers: tier {mp.get('best_tier')} "
                     f"x{mp.get('tier_weight')} | recency x{mp.get('recency')} "
                     f"({mp.get('freshest_age_days')}d) | corroboration "
                     f"x{mp.get('corroboration')} ({mp.get('n_corroboration_units')} "
                     f"units) | priced-in {a.get('priced_in')} x{mp.get('priced_in')} "
                     f"| model-overlap x{mp.get('model_overlap')}")
        fsc = sf.get("fsc") or {}
        ig = sf.get("integrity_gates") or {}
        L.append(f"  FSC: total severity {fsc.get('total_severity')}"
                 f"{' - VETO (NO_PICK)' if fsc.get('no_bet') else ''}"
                 f"  |  Integrity: verification {ig.get('verification_rate_pct')}%"
                 f"{' - confidence penalty applied' if ig.get('confidence_penalty') else ''}")
        for r in sf.get("no_bet_reasons", []):
            L.append(f"  !! {r}")
        for w in (sf.get("warnings") or [])[:6]:
            L.append(f"  ~ {w}")
    else:
        L.append("  Not applied (no register supplied - model probabilities only).")

    # [3] final picks with confidence + XAI
    L.append("")
    L.append("[3] FINAL PICKS (probability + per-market headline gate + "
             "HIGH/MEDIUM/LOW confidence; no odds-based value, no staking)")
    picks = rep.get("final_picks") or []
    if not picks:
        L.append("  (none - no pick market served on this match)")
    for i, b in enumerate(picks, 1):
        odds_s = (f" @ {b['odds']:.2f}" if b.get("odds") is not None else "")
        L.append(f"  PICK {i}  {b['market']:>5s} {b['pick_label']}{odds_s}")
        L.append(f"        p_final {b['p_final']:.4f} (raw {b['p_raw']:.4f}, soft "
                 f"{b['soft_total_pp']:+.2f}pp) | gate {b['gate']:.2f} "
                 f"{b.get('gate_label', '')} | confidence {b.get('confidence', '-')}")
        if b.get("market") == "ah":
            L.append(f"        line {b.get('line', 0):+g} | push probability "
                     f"{b.get('push_prob', 0):.3f}")
        for part in (b.get("xai_summary") or "").split(" | "):
            if part:
                L.append(f"        {part}")
    for mid in sorted(sf.get("no_bet_markets") or []):
        if mid in PICK_MARKETS:
            L.append(f"  PICK -    {mid:>5s}  BLOCKED (soft-factor NO_PICK - "
                     "see the audit above)")

    # [4] AI intuition pick (reasoning-based, not probability-derived)
    L.append("")
    L.append("[4] AI INTUITION PICK (reasoning-based - NOT a "
             "probability-qualified bet)")
    ai = rep.get("ai_intuition") or {}
    lean = ai.get("engine_lean") or {}
    L.append(f"  register narrative lean: {lean.get('side', 'NONE')} "
             f"({lean.get('strength', 'NONE')}, net "
             f"{lean.get('net_pp', 0.0):+.1f}pp on the 1X2 targets)")
    for d in (lean.get("drivers") or [])[:4]:
        try:
            pp_s = f"{float(d.get('audited_pp')):+.1f}pp"
        except (TypeError, ValueError):
            pp_s = "?pp"
        L.append(f"    driver: {d.get('module')} {d.get('market', 'x12')}"
                 f":{d.get('target')} {pp_s} audited"
                 + ("  [priced-in]" if d.get("priced_in") else ""))
    con = lean.get("contra") or {}
    if con.get("rejected_claims"):
        L.append(f"    contra: {con['rejected_claims']} register claim(s) "
                 "rejected at verification")
    if con.get("no_bet_markets"):
        L.append("    contra: NO_PICK on "
                 f"{', '.join(con['no_bet_markets'])}")
    if con.get("fsc_veto"):
        L.append("    contra: FSC VETO active (see the audit above)")
    op = ai.get("operator_pick") or {}
    L.append(f"  AI operator slot: pick = {op.get('pick') or '______'}")
    L.append(f"    reasoning: {op.get('reasoning') or '______'}")
    L.append("    (the running AI completes this from its OWN reading of the "
             "evidence - never from p_final; write 'no lean' if none)")

    # [5] notes
    fails = rep.get("model_fails") or {}
    L.append("")
    L.append("[5] NOTES")
    L.append(f"  artifact failures: {len(fails)}" +
             (f" ({', '.join(sorted(fails)[:4])}...)" if fails else ""))
    L.append("  probabilities always shown; gates and confidence ratings are "
             "labels - the decision is yours")
    return "\n".join(L)


# ============================================================================
# Adjustment Register loading + the built-in demo register
# ============================================================================
def load_register(path):
    """Load an Adjustment Register JSON (single register or {"by_match": ...})."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "by_match" in data:
        return None, data["by_match"]
    return data, None


DEMO_REGISTER = {
    "description": ("Built-in demo register - exercises the soft-factor audit "
                    "path end to end on the synthetic demo match"),
    "league_tier": "A",
    "phase6_status": "SURVIVES",
    "odds_as_of": None,
    "adjustment_register": [
        {
            "module_id": "M3", "module_name": "Lineup Impact Model",
            "market": "x12", "target": "home_win", "direction": "UP",
            "delta_pp": 2.0,
            "sources": [
                {"url": "https://www.bbc.co.uk/sport/football/teams/arsenal",
                 "tier": "T1", "age_days": 1,
                 "snippet": "Arsenal captain returns to full training and starts "
                            "the weekend fixture, club confirmed in the press "
                            "conference ahead of the league match."},
                {"url": "https://www.theguardian.com/football/arsenal",
                 "tier": "T2", "age_days": 1,
                 "snippet": "The Gunners welcome their captain back to the "
                            "starting lineup for the weekend league fixture "
                            "after a month on the sidelines."},
            ],
            "rule0_satisfied": True,
            "notes": "demo entry: captain returns, two independent outlets",
        },
        {
            "module_id": "M9", "module_name": "Weather Modifier",
            "market": "ou25", "target": "under", "direction": "UP",
            "delta_pp": 1.5,
            "sources": [
                {"url": "https://weather.com/forecasts/news/london-weekend-storm",
                 "tier": "T2", "age_days": 2,
                 "snippet": "Meteorologists forecast gusts above 50 km/h and "
                            "heavy rain for the London area across the weekend "
                            "window, with official warnings issued."},
            ],
            "rule0_satisfied": True,
            "notes": "demo entry: gale-force wind, extreme-conditions channel",
        },
        {
            "module_id": "M8", "module_name": "Referee Modifier",
            "market": "x12", "target": "home_win", "direction": "UP",
            "delta_pp": 1.0,
            "sources": [
                {"url": "https://www.referee-world.example.org/m8-claim",
                 "tier": "T3", "age_days": 30,
                 "snippet": "Referee career averages reportedly favour home "
                            "sides in this fixture history."},
            ],
            "rule0_satisfied": True,
            "notes": "demo entry: MUST BE REJECTED (M8 is FLAG_ONLY)",
        },
    ],
    "flags": [
        {"name": "storm-front variance", "severity": 2, "type": "VARIANCE",
         "module_id": "M9", "markets": ["ou25", "corners"],
         "detail": "gale warnings raise variance on totals and corners"},
    ],
    "data_integrity": {
        "total_data_points": 40, "verified_with_source_url": 38,
        "verification_rate_pct": 95, "unknown_data_points": 2,
        "t1_t2_count": 3, "t3_count": 0, "t4_t5_count": 0,
        "rule0_satisfied": True, "hallucination_self_check": "pass",
    },
}


# ============================================================================
# ================================================== model loading ============
class Engine:
    """Lazy-loading registry of every kit artifact. A load failure is
    recorded in self.fails and the affected market serves an explicit FAIL
    entry - the engine NEVER substitutes or hides."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else ROOT
        self.fails = {}                    # artifact name -> reason
        self.loaded = False
        self.champion = None
        self.champion_meta = {}
        self.allmeta = {}
        self.models = {}                   # market id -> ("cat"|"lgbm", model)
        self.soft = {}                     # member id -> (kind, obj)
        self.blend_weights = {}
        self.iso_cat = None
        self.iso_blend = None
        self.stack_meta = None
        self.evidence = {}

    # ------------------------------------------------------------- loaders
    def _fail(self, name, exc):
        self.fails[name] = f"{type(exc).__name__}: {exc}"
        return None

    def _load_cat(self, path):
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(str(path))
        return m

    def _load_lgb(self, path):
        import lightgbm as lgb
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt") as gf:
                return lgb.Booster(model_str=gf.read())
        return lgb.Booster(model_file=str(path))

    def _load_xgb(self, path):
        import xgboost as xgb
        b = xgb.Booster(model_file=str(path))
        # the booster may carry its training feature names - predictions
        # must then use a DMatrix with the SAME names (mirrored at predict)
        self._xgb_names = getattr(b, "feature_names", None)
        return b

    def _load_joblib(self, path, name):
        try:
            import sklearn
            fitted = self.sklearn_fitted()
            if sklearn.__version__ != fitted:
                raise RuntimeError(
                    f"runtime scikit-learn {sklearn.__version__} != fitted "
                    f"{fitted} - sklearn gives no cross-version pickle "
                    f"guarantee (pip install 'scikit-learn=={fitted}')")
            import joblib
            return joblib.load(str(path))
        except Exception as e:                               # noqa: BLE001
            return self._fail(name, e)

    def sklearn_fitted(self):
        p = MODELS_DIR / "softlayer" / "sklearn_meta.json"
        if p.exists():
            return json.loads(p.read_text()).get("sklearn_fitted_version", "?")
        return "?"

    # ------------------------------------------------------------- loading
    def load_all(self, verbose=False):
        if self.loaded:
            return self
        if not ensure_extracted(verbose=verbose):
            raise RuntimeError(
                "no model bundles found - the kit needs the *MODELS*.py "
                "bundle files (02 champion+soft layer, 03 markets, 04 "
                "correct score) next to this file")
        self.evidence = load_kit_evidence()

        # champion (1X2)
        try:
            self.champion_meta = json.loads(
                (MODELS_DIR / "champion_meta.json").read_text())
            if FEATURE_COLUMNS is None:
                globals()["FEATURE_COLUMNS"] = list(
                    self.champion_meta["feature_columns"])
            self.champion = self._load_cat(MODELS_DIR / "champion_cat.cbm")
        except Exception as e:                               # noqa: BLE001
            self._fail("champion_cat.cbm", e)

        # all-market models
        try:
            self.allmeta = json.loads(
                (MODELS_DIR / "allmarkets_meta.json").read_text())
        except Exception as e:                               # noqa: BLE001
            self._fail("allmarkets_meta.json", e)
            self.allmeta = {"markets": {}}
        for mid, rec in self.allmeta.get("markets", {}).items():
            path = MODELS_DIR / "allmarkets" / rec["model_file"]
            try:
                if not path.exists():
                    raise FileNotFoundError(str(path))
                if rec["model_type"] == "LightGBMClassifier":
                    self.models[mid] = ("lgbm", self._load_lgb(path))
                else:
                    self.models[mid] = ("cat", self._load_cat(path))
            except Exception as e:                           # noqa: BLE001
                self._fail(f"allmarkets/{rec['model_file']}", e)

        # soft-layer members
        sl = MODELS_DIR / "softlayer"
        try:
            self.blend_weights = json.loads(
                (sl / "blend_weights.json").read_text())["weights"]
        except Exception as e:                               # noqa: BLE001
            self._fail("blend_weights.json", e)
            self.blend_weights = {}
        for mid, loader, fname in (
                ("cat_tuned", self._load_cat, "cat_tuned.cbm"),
                ("lgbm_b", self._load_lgb, "lgbm_b.txt"),
                ("xgb_b", self._load_xgb, "xgb_b.json")):
            try:
                if not (sl / fname).exists():
                    raise FileNotFoundError(str(sl / fname))
                self.soft[mid] = ("model", loader(sl / fname))
            except Exception as e:                           # noqa: BLE001
                self._fail(f"softlayer/{fname}", e)
        self.soft["lr_b"] = ("sk", self._load_joblib(sl / "lr_b.joblib", "softlayer/lr_b.joblib"))
        self.stack_meta = self._load_joblib(sl / "stack_meta_lr.joblib",
                                            "softlayer/stack_meta_lr.joblib")
        self.iso_cat = self._load_joblib(sl / "iso_cat_b.joblib",
                                         "softlayer/iso_cat_b.joblib")
        self.iso_blend = self._load_joblib(sl / "iso_blend_wf.joblib",
                                           "softlayer/iso_blend_wf.joblib")
        self.loaded = True
        return self

    # ------------------------------------------------------- member predict
    def _member_proba(self, mid, xrow):
        """One soft-layer member's [pA,pD,pH] for a 1x135 row. None on FAIL."""
        if mid == "cat_b":
            if self.champion is None:
                return None
            return self.champion.predict_proba(xrow)[0]
        if mid == "book":
            return "odds"                    # handled by caller
        obj = self.soft.get(mid)
        if obj is None or obj[1] is None:
            return None
        kind, m = obj
        try:
            if kind == "sk":
                return m.predict_proba(xrow)[0]
            if mid == "xgb_b":
                import xgboost as xgb
                names = getattr(self, "_xgb_names", None)
                dmat = xgb.DMatrix(xrow, feature_names=names)
                return m.predict(dmat)[0]
            return m.predict(xrow)[0]
        except Exception as e:                               # noqa: BLE001
            self._fail(f"softlayer/{mid}", e)
            return None

    # ------------------------------------------------------------ soft layer
    def soft_layer(self, xrow, odds3=None):
        """All soft-layer 1X2 variants for one feature row."""
        out = {}

        p_cat = self._member_proba("cat_b", xrow)
        p_ct = self._member_proba("cat_tuned", xrow)
        p_lg = self._member_proba("lgbm_b", xrow)
        p_xg = self._member_proba("xgb_b", xrow)
        p_lr = self._member_proba("lr_b", xrow)

        p_book = None
        if odds3 is not None:
            o = np.asarray(odds3, dtype=float)
            if np.all(np.isfinite(o)) and np.all(o > 1.0):
                inv = 1.0 / o
                p_book = inv / inv.sum()

        def emit(name, p, note=None):
            if p is None:
                out[name] = {"status": "FAIL",
                             "reason": self.fails.get(f"softlayer/{name}",
                                                      "member unavailable")}
                return
            rec = {"status": "ok",
                   "p": [round(float(v), 4) for v in p],
                   "pick": CLASSES[int(np.argmax(p))],
                   "confidence": round(float(np.max(p)), 4)}
            if note:
                rec["note"] = note
            out[name] = rec

        # vote GBM trio (soft vote)
        trio = [p for p in (p_xg, p_lg, p_cat) if p is not None]
        if len(trio) == 3:
            emit("vote_gbm", norm(sum(trio)))
        else:
            missing = [n for n, p in (("xgb_b", p_xg), ("lgbm_b", p_lg),
                                       ("cat_b", p_cat)) if p is None]
            self.fails.setdefault("softlayer/vote_gbm",
                                   f"members unavailable: {missing}")
            emit("vote_gbm", None)

        # fixed 5-3-2 blend
        if p_lg is not None and p_xg is not None and p_cat is not None:
            emit("blend_532", norm(0.5 * p_lg + 0.3 * p_xg + 0.2 * p_cat))
        else:
            emit("blend_532", None)

        # tuned blend over available candidates (weights renormalized)
        p_blend_wf = None
        if self.blend_weights:
            cand = {"book": p_book, "lr_b": p_lr, "lgbm_b": p_lg,
                    "xgb_b": p_xg, "cat_b": p_cat, "cat_tuned": p_ct}
            dropped = [c for c in SOFT_CANDS if cand.get(c) is None]
            use = [c for c in SOFT_CANDS if cand.get(c) is not None]
            if use:
                w = np.array([self.blend_weights.get(c, 0.0) for c in use])
                if w.sum() <= 0:
                    w = np.ones(len(use))
                w = w / w.sum()
                p_blend_wf = norm(sum(wi * cand[c] for wi, c in zip(w, use)))
                note = None
                if dropped:
                    note = (f"weights renormalized over {use} "
                            f"(unavailable: {dropped})")
                emit("blend_wf", p_blend_wf, note)
            else:
                emit("blend_wf", None)
        else:
            emit("blend_wf", None)

        # stack -> LR meta on 12 OOF columns (each base's 3-class vector
        # CONCATENATED: [lr_b | lgbm_b | xgb_b | cat_b] -> shape (1, 12))
        bases = [p_lr, p_lg, p_xg, p_cat]
        if self.stack_meta is not None and all(b is not None for b in bases):
            try:
                Z = np.concatenate(
                    [np.asarray(b, dtype=float).ravel() for b in bases]
                ).reshape(1, -1)
                emit("stack_lr", self.stack_meta.predict_proba(Z)[0])
            except Exception as e:                           # noqa: BLE001
                self._fail("softlayer/stack_lr", e)
                emit("stack_lr", None)
        else:
            emit("stack_lr", None)

        # isotonic calibration x2 (blend_wf_iso calibrates the tuned blend;
        # both use the UNROUNDED member probabilities)
        def iso_apply(iso, p_src):
            if iso is None or p_src is None:
                return None
            cols = []
            for c in range(3):
                key = c if c in iso else str(c)
                if key not in iso:
                    raise KeyError(f"isotonic class key {c!r} missing")
                cols.append(float(np.atleast_1d(
                    iso[key].predict(np.atleast_1d(float(p_src[c]))))[0]))
            return norm(np.array(cols))

        for name, iso, src in (("cat_b_iso", self.iso_cat, p_cat),
                               ("blend_wf_iso", self.iso_blend, p_blend_wf)):
            try:
                emit(name, iso_apply(iso, src))
            except Exception as e:                           # noqa: BLE001
                self._fail(f"softlayer/{name}", e)
                emit(name, None)

        return out

    # ------------------------------------------------------------- markets
    def price_market(self, mid, xrow):
        """One all-markets model on one row -> the market block."""
        rec = self.allmeta.get("markets", {}).get(mid)
        if rec is None:
            return {"status": "FAIL", "reason": f"unknown market {mid}"}
        entry = self.models.get(mid)
        if entry is None:
            return {"status": "FAIL",
                    "reason": self.fails.get(
                        f"allmarkets/{rec['model_file']}", "model not loaded")}
        kind, model = entry
        try:
            if kind == "lgbm":
                p = np.asarray(model.predict(xrow)[0], dtype=float)
            else:
                p = np.asarray(model.predict_proba(xrow)[0], dtype=float)
        except Exception as e:                               # noqa: BLE001
            self._fail(f"allmarkets/{rec['model_file']}", e)
            return {"status": "FAIL",
                    "reason": self.fails[f"allmarkets/{rec['model_file']}"]}
        classes = rec["classes"]
        out = {"status": "ok", "classes": classes,
               "p": [round(float(v), 4) for v in p]}
        if len(p) == 2:
            pick = int(p[1] >= 0.5)
            out["pick"] = classes[pick]
            out["confidence"] = round(float(max(p)), 4)
        else:
            pick = int(np.argmax(p))
            out["pick"] = classes[pick]
            out["confidence"] = round(float(p[pick]), 4)
        gate = GATES.get(mid)
        if gate:
            out["gate"] = gate
            out["gate_label"] = ("STRONG" if out["confidence"] >= gate
                                 else "ADVISORY")
        ev = (self.evidence.get("markets", {}).get(mid, {}))
        out["evidence"] = {k: ev.get(k) for k in
                           ("acc", "base_acc", "n_test", "gate_row") if ev.get(k) is not None}
        if mid == "dnb":
            ph = float(p[1])
            out["directions"] = {
                "dnb_home": {"pick": bool(ph >= 0.5), "p": round(ph, 4)},
                "dnb_away": {"pick": bool(ph <= 0.5), "p": round(1 - ph, 4)}}
        return out

    def price_row(self, xrow, odds3=None, ah_line=None, odds_ou25=None,
                  odds_btts=None, odds_ah=None, soft_ctx=None):
        """Every market + the soft layers + the final picks (with TreeSHAP
        XAI) for one 1x135 feature row."""
        xrow = np.atleast_2d(np.asarray(xrow, dtype=np.float32))
        report = {"markets": {}, "model_fails": dict(self.fails)}

        # 1X2 champion
        if self.champion is None:
            report["markets"]["x12"] = {
                "status": "FAIL", "reason": self.fails.get("champion_cat.cbm",
                                                           "champion not loaded")}
        else:
            try:
                p = self.champion.predict_proba(xrow)[0]
                blk = {"status": "ok", "classes": CLASSES,
                       "p": [round(float(v), 4) for v in p],
                       "pick": CLASSES[int(np.argmax(p))],
                       "confidence": round(float(np.max(p)), 4),
                       "gate": GATES["x12"],
                       "gate_label": ("STRONG" if float(np.max(p)) >= GATES["x12"]
                                      else "ADVISORY")}
                ev = self.evidence.get("champion", {})
                blk["evidence"] = {"logloss": ev.get("logloss"),
                                   "accuracy": ev.get("accuracy"),
                                   "n_eligible": ev.get("n_eligible")}
                report["markets"]["x12"] = blk
                report["_p_x12"] = np.asarray(p, dtype=float)
            except Exception as e:                           # noqa: BLE001
                self._fail("champion_cat.cbm", e)
                report["markets"]["x12"] = {"status": "FAIL",
                                            "reason": self.fails["champion_cat.cbm"]}

        # every all-markets model
        for mid in sorted(self.allmeta.get("markets", {})):
            report["markets"][mid] = self.price_market(mid, xrow)

        # Asian handicap per-line cover/push at the quoted line.
        # Settlement semantics (3-way accounting: cover / push / lose):
        #   - censored +5/-5 margin buckets count as "always cover" /
        #     "never cover" only when the line sits INSIDE the model's
        #     margin range (|line| < 5); at |line| >= 5 a margin equal to
        #     the model's ceiling can push (|line| == 5) or lose (|line| > 5).
        #   - quarter-ball lines: x25 lines take their half-push at the
        #     adjacent integer below the threshold; on favorite-side x75
        #     lines the boundary margin wins one half-ball and pushes the
        #     other, so it counts half cover + half push.
        ah = report["markets"].get("ah", {})
        if ah.get("status") == "ok" and ah_line is not None and np.isfinite(ah_line):
            try:
                cls_m = self.allmeta["markets"]["ah"]["classes"]
                p = np.asarray(ah["p"], dtype=float)
                t = -float(ah_line)
                pw = pp = 0.0
                # favorite-side x75 line: the boundary margin (just above the
                # threshold) wins one half-ball and pushes the other
                fav_x75 = abs(t - round(t)) == 0.25 and round(t) > t
                for k, mcls in enumerate(cls_m):
                    mv = int(mcls)
                    if fav_x75 and mv == int(round(t)):
                        pw += 0.5 * float(p[k])
                        pp += 0.5 * float(p[k])
                    elif mv == 5:
                        if t < 5 - 1e-9:
                            pw += float(p[k])       # inside range: always covers
                        elif abs(t - 5.0) < 1e-9:
                            pp += float(p[k])       # line -5: margin +5 pushes
                        # t > 5: margin 5 (the model's ceiling) loses at home -5.5+
                    elif mv == -5:
                        if t > -5 + 1e-9:
                            pass                    # inside range: never covers
                        elif abs(t + 5.0) < 1e-9:
                            pp += float(p[k])       # line +5: margin -5 pushes
                        else:
                            pw += float(p[k])       # beyond: margin -5 covers
                    elif mv > t + 1e-9:
                        pw += float(p[k])
                    elif abs(mv - t) < 1e-9:
                        pp += float(p[k])
                    elif abs(t - round(t)) == 0.25:
                        # quarter-ball line: half-push at the adjacent integer
                        cands = [t - 0.25, t + 0.25]
                        cands = [c for c in cands if abs(c - round(c)) < 1e-9]
                        if cands and int(round(cands[0])) == mv:
                            pp += 0.5 * float(p[k])
                ah["line"] = float(ah_line)
                ah["cover_home"] = round(pw, 4)
                ah["push"] = round(pp, 4)
                ah["cover_away"] = round(1 - pw - pp, 4)
                ah["policy"] = ("cover/push/lose probabilities at the quoted "
                                "line; the final pick is the higher-cover side "
                                "with its confidence rating (no EV, no staking)")
            except Exception as e:                           # noqa: BLE001
                self._fail("allmarkets/ah", e)

        # soft layer (1X2 ensemble) + soft-FACTOR layer (register) + picks
        report["soft_layer"] = self.soft_layer(xrow, odds3=odds3)
        p_x12 = report.pop("_p_x12", None)
        self._soft_factors_and_picks(
            report, xrow, odds3=odds3, p_x12=p_x12, ah_line=ah_line,
            odds_ou25=odds_ou25, odds_btts=odds_btts, odds_ah=odds_ah,
            soft_ctx=soft_ctx or {})
        return report

    # ------------------------------------------------ soft factors + picks
    def _soft_factors_and_picks(self, report, xrow, odds3=None, p_x12=None,
                                ah_line=None, odds_ou25=None, odds_btts=None,
                                odds_ah=None, soft_ctx=None):
        """Run the Adjustment Register audit, apply audited deltas, and emit
        the final picks (x12 / ou25 / btts / ah): probability + 0.60 headline
        gate + HIGH/MEDIUM/LOW confidence, each with TreeSHAP XAI."""
        soft_ctx = soft_ctx or {}
        register = soft_ctx.get("soft_register")

        # ---- baselines: {market: {target: p}} for delta-capable markets ----
        baselines = {}
        market_probs = {}
        for mid, blk in report["markets"].items():
            if blk.get("status") != "ok" or mid not in TARGETS_BY_CLASS:
                continue
            p = [float(v) for v in blk["p"]]
            market_probs[mid] = {"classes": blk["classes"], "p": p}
            if mid == "ah":
                continue                      # ah handled via cover probabilities
            for ci, tgt in enumerate(TARGETS_BY_CLASS[mid]):
                baselines.setdefault(mid, {})[tgt] = p[ci]
        ahblk = report["markets"].get("ah", {})
        if ahblk.get("status") == "ok" and "cover_home" in ahblk:
            baselines["ah"] = {"home_cover": float(ahblk["cover_home"]),
                               "away_cover": float(ahblk["cover_away"])}

        # ---- run the audit + apply the audited deltas ----
        proc = SoftFactorProcessor()
        sf = proc.process(register, baselines,
                          competition=soft_ctx.get("competition"),
                          kickoff_utc=soft_ctx.get("kickoff_utc"),
                          odds_as_of=soft_ctx.get("odds_as_of"),
                          match_id=soft_ctx.get("match_id"))
        final_p = apply_soft_factors(market_probs, sf.get("audited_delta_pp", {}))
        if "cover_home" in baselines.get("ah", {}):
            pass                          # ah deltas applied below on the block
        # ah deltas: apply to cover_home / cover_away (push kept)
        for mkey, dpp in sf.get("audited_delta_pp", {}).items():
            if not mkey.startswith("ah:") or ahblk.get("status") != "ok" \
                    or "cover_home" not in ahblk:
                continue
            d = dpp / 100.0 * (1.0 if mkey.endswith("home_cover") else -1.0)
            new_h = min(0.99, max(0.01, float(ahblk["cover_home"]) + d))
            push = float(ahblk.get("push", 0.0))
            ahblk["cover_home"] = round(new_h, 4)
            ahblk["cover_away"] = round(max(0.0, 1.0 - new_h - push), 4)

        # ---- x12 final basis: best-calibrated ensemble variant + audited deltas
        # (one "final" everywhere: the [1] table, final_pick, and the picks all
        # use this same vector; markets.x12.p stays the RAW champion evidence)
        def final_x12():
            for name in FINAL_X12_PRIORITY:
                v = report["soft_layer"].get(name, {})
                if v.get("status") == "ok" and len(v.get("p", [])) == 3:
                    return np.asarray(v["p"], dtype=float), name
            return (np.asarray(p_x12, dtype=float), "champion") \
                if p_x12 is not None else (None, None)

        p_final_x12, x12_basis = final_x12()
        d_x12 = 0.0
        if p_final_x12 is not None:
            # apply the audited x12 deltas on the calibrated basis (full
            # delivery: touched targets move by the whole audited pp, the
            # complementary mass is removed proportionally)
            p_cal = list(p_final_x12)
            tmap = TARGETS_BY_CLASS["x12"]
            touched = []
            for mkey, dpp in sf.get("audited_delta_pp", {}).items():
                if not mkey.startswith("x12:"):
                    continue
                tgt = mkey.split(":", 1)[1]
                if tgt in tmap:
                    d_x12 += dpp
                    ci = tmap.index(tgt)
                    touched.append(ci)
                    p_cal[ci] = p_cal[ci] + dpp / 100.0
            if touched:
                others = [ci for ci in range(3) if ci not in touched]
                rest = 1.0 - sum(p_cal[ci] for ci in touched)
                base_rest = sum(float(p_cal[ci]) for ci in others)
                if base_rest > 1e-9:
                    for ci in others:
                        p_cal[ci] = rest * (float(p_cal[ci]) / base_rest)
                p_cal = np.clip(np.asarray(p_cal, dtype=float), 0.005, 0.995)
                p_final_x12 = p_cal / p_cal.sum()
            else:
                p_final_x12 = np.asarray(p_final_x12, dtype=float)
            final_p["x12"] = [float(v) for v in p_final_x12]

        sf["final_probabilities"] = {
            m: [round(float(v), 4) for v in p] for m, p in final_p.items()}
        sf["final_confidence"] = {}
        sf["final_pick"] = {}
        sf["final_basis"] = {}
        for m, p in final_p.items():
            blk = report["markets"].get(m, {})
            if not p or blk.get("status") != "ok":
                continue
            ci = int(np.argmax(p))
            sf["final_confidence"][m] = round(float(p[ci]), 4)
            sf["final_pick"][m] = blk["classes"][ci]
            sf["final_basis"][m] = (x12_basis if m == "x12" and x12_basis
                                    else "model + audited soft factors")
        report["soft_factors"] = sf

        no_bet = set(sf.get("no_bet_markets") or {})

        # ---- emit the final picks (probability + confidence; no value) ----
        picks = []

        def mk_pick(mid, pick_label, p_raw, p_final, soft_pp, model,
                    class_idx, soft_adj, evidence, odds=None,
                    xai_invert=False, extra=None):
            if mid in no_bet:
                return None
            g = GATES.get(mid, 0.60)
            xai = (xai_explain_bet(model, xrow, class_idx, invert=xai_invert)
                   if model is not None else None)
            pick = {
                "market": mid, "pick_label": pick_label,
                "p_raw": round(float(p_raw), 4),
                "p_final": round(float(p_final), 4),
                "soft_total_pp": round(float(soft_pp), 2),
                "gate": round(float(g), 2),
                "gate_label": ("STRONG" if float(p_final) >= g else "ADVISORY"),
                "confidence": confidence_rating(float(p_final), mid),
                "odds": (round(float(odds), 3)
                         if odds is not None and np.isfinite(odds)
                         and float(odds) > 1.0 else None),
                "basis": "final (calibrated ensemble + audited soft factors)",
                "evidence": evidence,
            }
            if xai:
                pick["xai"] = xai
            pick["xai_summary"] = xai_compose(pick, xai, soft_adj, evidence)
            if extra:
                pick.update(extra)
            return pick

        ev_ch = self.evidence.get("champion", {})
        bc = ev_ch.get("book_control") or {}
        ev_note_x12 = (f"champion walk-forward: accuracy "
                       f"{ev_ch.get('accuracy', '?')} on "
                       f"{ev_ch.get('n_eligible', 0):,} eligible matches "
                       f"(logloss {ev_ch.get('logloss', '?')} vs de-margined "
                       f"book {bc.get('logloss', '?')})")

        def ev_note(mid):
            ev = self.evidence.get("markets", {}).get(mid, {})
            gr = ev.get("gate_row") or {}
            if not gr:
                return None
            return (f"walk-forward @g{gr.get('g')}: {gr.get('acc', 0):.1%} hit "
                    f"on {gr.get('cov', 0):.1%} coverage (lift "
                    f"{gr.get('lift', 0):+.1%} against the "
                    f"{gr.get('base_acc', 0):.1%} base)")

        # x12
        if p_final_x12 is not None:
            raw = np.asarray(p_x12 if p_x12 is not None else p_final_x12,
                             dtype=float)
            pk = int(np.asarray(p_final_x12, dtype=float).argmax())
            soft_adj = [a for a in sf.get("adjustments", [])
                        if a.get("market") == "x12" and a.get("target")
                        in ("home_win", "draw", "away_win")]
            odds_pk = None
            if odds3 is not None:
                o = np.asarray(odds3, dtype=float)
                if np.all(np.isfinite(o)) and o[pk] > 1.0:
                    odds_pk = float(o[pk])
            p = mk_pick("x12", {"H": "HOME", "D": "DRAW", "A": "AWAY"}[CLASSES[pk]],
                        raw[pk], float(p_final_x12[pk]), d_x12,
                        self.champion, pk, soft_adj, ev_note_x12, odds=odds_pk)
            if p:
                p["pick"] = CLASSES[pk]
                p["probability_basis"] = x12_basis
                picks.append(p)

        # ou25 / btts (binary, model classes [no, yes]; p[1] is the yes/over side)
        for mid, odds_pair, labels in (
                ("ou25", odds_ou25, ("UNDER 2.5", "OVER 2.5")),
                ("btts", odds_btts, ("BTTS NO", "BTTS YES"))):
            blk = report["markets"].get(mid, {})
            if blk.get("status") != "ok":
                continue
            fp = final_p.get(mid)
            if fp is None or len(fp) != 2:
                continue
            rp = [float(v) for v in blk["p"]]
            fpv = np.asarray(fp, dtype=float)
            pk = int(fpv.argmax())
            soft_adj = [a for a in sf.get("adjustments", [])
                        if a.get("market") == mid]
            soft_pp = sum(a.get("audited_delta_pp", 0.0) for a in soft_adj)
            odds_pk = None
            if odds_pair is not None:
                o = np.asarray(odds_pair, dtype=float)
                if np.all(np.isfinite(o)) and np.all(o > 1.0) and o[pk] > 1.0:
                    odds_pk = float(o[pk])
            p = mk_pick(mid, labels[pk], rp[pk], float(fpv[pk]), soft_pp,
                        self.models.get(mid, (None, None))[1], pk, soft_adj,
                        ev_note(mid), odds=odds_pk, xai_invert=(pk == 0))
            if p:
                p["pick"] = blk["classes"][pk]
                picks.append(p)

        # ah (the higher-cover side at the quoted line)
        ahblk = report["markets"].get("ah", {})
        if ahblk.get("status") == "ok" and "cover_home" in ahblk:
            cov_h = float(ahblk["cover_home"])
            cov_a = float(ahblk["cover_away"])
            push = float(ahblk.get("push", 0.0))
            line = ahblk.get("line")
            side = "home" if cov_h >= cov_a else "away"
            cov = cov_h if side == "home" else cov_a
            soft_adj = [a for a in sf.get("adjustments", [])
                        if a.get("market") == "ah"]
            soft_pp = sum(a.get("audited_delta_pp", 0.0) for a in soft_adj)
            ah_model = self.models.get("ah", (None, None))[1]
            # TreeSHAP class: the highest-probability margin class that
            # actually covers for the picked side (boundary quarter-ball
            # class included; global argmax only as a fallback)
            ah_cls = None
            try:
                cls_m = self.allmeta["markets"]["ah"]["classes"]
                p_vec = [float(v) for v in ahblk["p"]]
                t = -float(line) if line is not None else 0.0
                cands = ([k for k, mc in enumerate(cls_m)
                          if int(mc) > t + 1e-9]
                         if side == "home" else
                         [k for k, mc in enumerate(cls_m)
                          if int(mc) < t - 1e-9])
                if line is not None and abs(t - round(t)) == 0.25 and (
                        (side == "home") == (round(t) > t)):
                    cands = cands + [k for k, mc in enumerate(cls_m)
                                     if int(mc) == round(t)]
                pool = cands or list(range(len(cls_m)))
                ah_cls = int(max(pool, key=lambda k: p_vec[k]))
            except (KeyError, TypeError, ValueError):
                ah_cls = None
            if odds_ah is not None:
                try:
                    o_line, o_h, o_a = [float(v) for v in odds_ah]
                    if np.isfinite(o_line):
                        line = o_line
                    cand = o_h if side == "home" else o_a
                    if np.isfinite(cand) and cand > 1.0:
                        odds_side = cand
                    else:
                        odds_side = None
                except (TypeError, ValueError):
                    odds_side = None
            else:
                odds_side = None
            label = (f"AH HOME {line:+g}" if side == "home"
                     else f"AH AWAY {-line:+g}") \
                if line is not None else f"AH {side.upper()}"
            p = mk_pick("ah", label, cov, cov, soft_pp, ah_model, ah_cls,
                        soft_adj,
                        ("AH margin-model walk-forward: 45.5% settled accuracy "
                         "(voids 7.0%); cover probabilities at the quoted line; "
                         "high-confidence rows are small-n"),
                        odds=odds_side,
                        extra={"pick": side, "push_prob": round(push, 4),
                               "line": (round(float(line), 3)
                                        if line is not None else None)})
            if p:
                picks.append(p)

        # ---- every remaining modeled market: the same probability + gate +
        # confidence contract (binary two-sided markets take the argmax side;
        # yes-only markets always sit on the yes/win side; htft / cs take the
        # argmax class at their own base-rate gate, model-only basis)
        for mid in PICK_MARKETS:
            if mid in ("x12", "ou25", "btts", "ah"):
                continue                  # bespoke blocks above (odds / line)
            blk = report["markets"].get(mid, {})
            if blk.get("status") != "ok" or not blk.get("classes"):
                continue
            classes = blk["classes"]
            rp = [float(v) for v in blk["p"]]
            fp = final_p.get(mid)
            fpv = (np.asarray(fp, dtype=float)
                   if fp is not None and len(fp) == len(classes)
                   else np.asarray(rp, dtype=float))
            if mid in YES_ONLY_MARKETS:
                ci = len(classes) - 1     # the yes/win side is the real market
            else:
                ci = int(fpv.argmax())
            soft_adj = [a for a in sf.get("adjustments", [])
                        if a.get("market") == mid]
            soft_pp = sum(a.get("audited_delta_pp", 0.0) for a in soft_adj)
            p = mk_pick(mid, pick_label_for(mid, classes, ci), rp[ci],
                        float(fpv[ci]), soft_pp,
                        self.models.get(mid, (None, None))[1], ci, soft_adj,
                        ev_note(mid), odds=None,
                        xai_invert=(len(classes) == 2 and ci == 0))
            if p:
                p["pick"] = classes[ci]
                if mid in MODEL_ONLY_MARKETS:
                    p["basis"] = ("model probabilities (no soft-factor "
                                  "targets for this market)")
                picks.append(p)

        order = {m: i for i, m in enumerate(PICK_MARKETS)}
        picks.sort(key=lambda b: order.get(b["market"], len(PICK_MARKETS)))
        report["final_picks"] = picks

        # ---- AI intuition scaffolding (reasoning-based, NOT probability-
        # derived): the engine compiles the narrative digest from the audited
        # register; the operator AI completes the pick slot in the report
        report["ai_intuition"] = _ai_intuition_block(sf, report)
        return report



# ============================================================ serving =======
def price_payload_rows(engine, payload, rows, min_conf=0.0, register=None,
                       register_by_match=None, odds_as_of=None):
    """Price selected rows of a feature payload -> list of match reports.

    register: a single Adjustment Register applied to every priced match, or
    register_by_match: {"Home|Away|YYYY-MM-DD": register} for per-match
    registers. odds_as_of: ISO timestamp of the odds snapshot (RULE 20).
    """
    X = payload["X_B_XL"]
    dates = pd.DatetimeIndex(payload["date"])
    meta = payload["meta"]
    odds_best = payload["odds_best"]
    odds_ou25 = payload.get("odds_ou25")
    odds_btts = payload.get("odds_btts")
    odds_ah = payload.get("odds_ah")
    out = []
    for i in rows:
        xrow = X.iloc[i].to_numpy(dtype=np.float32).reshape(1, -1)
        o = odds_best[i]
        odds3 = o if np.all(np.isfinite(o)) else None
        ah_line = None
        try:
            ah_line = float(X.iloc[i, FEATURE_COLUMNS.index("mk_ah_line")])
        except (ValueError, KeyError, TypeError):
            ah_line = None
        o25 = odds_ou25[i] if odds_ou25 is not None else None
        obt = odds_btts[i] if odds_btts is not None else None
        oah = odds_ah[i] if odds_ah is not None else None
        m = meta.iloc[i]
        reg = register
        if reg is None and register_by_match:
            key = f"{m['home']}|{m['away']}|{m['date']}"
            reg = register_by_match.get(key)
            if reg is None:
                key2 = f"{m['home']}|{m['away']}"
                reg = register_by_match.get(key2)
        soft_ctx = {"soft_register": reg,
                    "competition": str(m["competition"]),
                    "kickoff_utc": None,       # CSVs carry date only
                    "odds_as_of": odds_as_of,
                    "match_id": key if register_by_match else None}
        rep = engine.price_row(
            xrow, odds3=odds3, ah_line=ah_line,
            odds_ou25=(o25 if np.all(np.isfinite(o25)) else None),
            odds_btts=(obt if np.all(np.isfinite(obt)) else None),
            odds_ah=(oah if np.all(np.isfinite(oah)) else None),
            soft_ctx=soft_ctx)
        rec = {"date": str(m["date"]), "home": str(m["home"]),
               "away": str(m["away"]),
               "competition": str(m["competition"]),
               "in_sample_warning": bool(dates[i] <= pd.Timestamp(TRAINED_THROUGH)),
               }
        gh, ga = m["goals_h"], m["goals_a"]
        if pd.notna(gh) and pd.notna(ga):
            rec["result"] = "H" if gh > ga else ("A" if ga > gh else "D")
        rep.update(rec)
        if min_conf > 0:
            rep["markets"] = {
                k: v for k, v in rep["markets"].items()
                if v.get("status") != "ok" or v.get("confidence", 1.0) >= min_conf}
        rep["report_text"] = match_report_text(rep)
        out.append(rep)
    return out


TRAINED_THROUGH = None


def _resolve_trained_through(engine):
    global TRAINED_THROUGH
    cands = []
    if engine.champion_meta.get("trained_through"):
        cands.append(engine.champion_meta["trained_through"])
    if engine.allmeta.get("trained_through"):
        cands.append(engine.allmeta["trained_through"])
    TRAINED_THROUGH = max(cands) if cands else None
    return TRAINED_THROUGH


def demo_row():
    """A synthetic, clearly-labeled feature row (smoke/demo only)."""
    feats = FEATURE_COLUMNS
    row = pd.Series(np.nan, index=feats, dtype=np.float32)
    for c in feats:
        if c.startswith("c_") or c.startswith("src_"):
            row[c] = 0.0
    row["c_England"] = 1.0
    row["elo_home"] = 1650.0
    row["elo_away"] = 1450.0
    row["elo_diff"] = 200.0
    row["form_pts_home"] = 2.2
    row["form_pts_away"] = 0.9
    row["rest_home"] = 6.0
    row["rest_away"] = 3.0
    row["mk_ah_line"] = -0.75
    return row.to_numpy(dtype=np.float32).reshape(1, -1)


def run_demo(engine):
    _resolve_trained_through(engine)
    xrow = demo_row()
    odds3 = [3.10, 3.45, 2.28]           # a plausible synthetic market
    rep = engine.price_row(
        xrow, odds3=odds3, ah_line=-0.75,
        odds_ou25=[1.85, 2.05], odds_btts=[1.80, 2.00], odds_ah=[-0.75, 1.92, 1.98],
        soft_ctx={"soft_register": DEMO_REGISTER,
                  "competition": "Premier League",
                  "kickoff_utc": None,
                  "odds_as_of": DEMO_REGISTER.get("odds_as_of"),
                  "match_id": "demo"})
    rep["demo"] = True
    rep["note"] = ("SYNTHETIC demo row - proves every artifact loads and "
                   "serves in this environment; not a real prediction")

    print("DEMO - synthetic match (England, elo 1650 v 1450, form 2.2 v 0.9)")
    print(f"{'market':10s} {'pick':>8s} {'conf':>7s}  {'gate':>5s}  label")
    for mid in sorted(rep["markets"]):
        v = rep["markets"][mid]
        if v.get("status") != "ok":
            print(f"{mid:10s} FAIL: {v.get('reason', '?')[:60]}")
            continue
        extra = ""
        if mid == "ah":
            extra = (f"  line {v['line']:+g}: cover H {v['cover_home']:.0%}"
                     f" / A {v['cover_away']:.0%} (push {v['push']:.0%})")
        print(f"{mid:10s} {str(v['pick']):>8s} {v['confidence']:7.2f}"
              f"  {v.get('gate', 0):5.2f}  {v.get('gate_label', '')}{extra}")
    print("\nsoft layer (1X2 variants):")
    for name, v in sorted(rep["soft_layer"].items()):
        if v.get("status") == "ok":
            print(f"  {name:13s} p(A/D/H) = "
                  + "/".join(f"{x:.3f}" for x in v["p"])
                  + f"  pick {v['pick']}")
        else:
            print(f"  {name:13s} FAIL: {v.get('reason', '?')[:60]}")
    sf = rep.get("soft_factors", {})
    print("\nsoft-factor layer (register audit):")
    print(f"  entries: {sf.get('entries_verified')}/{sf.get('entries_total')} "
          f"verified, league tier {sf.get('league_tier')}")
    for a in sf.get("adjustments", []):
        print(f"  {str(a.get('module')):<6} {a.get('market')}:{a.get('target')} "
              f"claim {a.get('delta_pp')}pp -> audited "
              f"{a.get('audited_delta_pp')}pp  [{a.get('status')}]")
    for v in sf.get("verification", []):
        print(f"  {str(v.get('module')):<6} {v.get('market')}:{v.get('target')} "
              f"REJECTED: {v.get('reason')[:70]}")
    print("\nFINAL PICKS (probability + gate + confidence rating):")
    picks = rep.get("final_picks", [])
    if not picks:
        print("  (none on the synthetic row)")
    for i, b in enumerate(picks, 1):
        odds_s = f" @ {b['odds']:.2f}" if b.get("odds") is not None else ""
        print(f"  PICK {i}  {b['market']:>5s} {b['pick_label']}{odds_s}"
              f"  p_final {b['p_final']:.4f} (gate {b['gate']:.2f} "
              f"{b.get('gate_label', '')})"
              f"  confidence {b.get('confidence', '-')}")
        if b.get("xai_summary"):
            print(f"        XAI: {b['xai_summary'][:150]}")
    ai = rep.get("ai_intuition") or {}
    lean = ai.get("engine_lean") or {}
    print(f"\nAI INTUITION (reasoning lean - not a probability bet): "
          f"{lean.get('side', 'NONE')} {lean.get('strength', 'NONE')} "
          f"({lean.get('net_pp', 0.0):+.1f}pp register narrative); "
          "operator pick slot left for the running AI")
    ok = sum(1 for v in rep["markets"].values() if v.get("status") == "ok")
    print(f"\n{ok}/{len(rep['markets'])} markets served, "
          f"{len(engine.fails)} artifact failures")
    rep["report_text"] = match_report_text(rep)
    return rep


def run_csv(engine, csvs, after=None, limit=None, min_conf=0.0, out_path=None,
            use_cache=True, register=None, register_by_match=None,
            odds_as_of=None):
    _resolve_trained_through(engine)
    payload = build_features(csvs, use_cache=use_cache)
    dates = pd.DatetimeIndex(payload["date"])
    after_ts = pd.Timestamp(after) if after else pd.Timestamp(TRAINED_THROUGH)
    rows = np.where(dates > after_ts)[0]
    if limit:
        rows = rows[:int(limit)]
    if len(rows) == 0:
        print(f"no matches after {after_ts.date()} - append fixtures to the "
              f"CSVs (unplayed rows keep their features; odds optional)")
        return None
    print(f"pricing {len(rows)} matches after {after_ts.date()} "
          f"(trained through {TRAINED_THROUGH}) ...")
    reports = price_payload_rows(
        engine, payload, rows, min_conf=min_conf, register=register,
        register_by_match=register_by_match, odds_as_of=odds_as_of)
    n_ok = sum(1 for r in reports for v in r["markets"].values()
               if v.get("status") == "ok")
    n_fail = sum(1 for r in reports for v in r["markets"].values()
                 if v.get("status") == "FAIL")
    n_picks = sum(len(r.get("final_picks", [])) for r in reports)
    print(f"done: {n_ok} market predictions served, {n_fail} FAILs, "
          f"{len(engine.fails)} artifact failures, {n_picks} final picks")
    for r in reports[:25]:
        tag = " ** IN-SAMPLE **" if r.get("in_sample_warning") else ""
        n_p = len(r.get("final_picks", []))
        picks_txt = f" picks={n_p}" if n_p else ""
        print(f"  {r['date']} {r['home'][:20]:20s} v {r['away'][:20]:20s}"
              f" x12={r['markets'].get('x12', {}).get('pick', '?'):>3s}"
              f" ou25={r['markets'].get('ou25', {}).get('pick', '?'):>6s}"
              f" btts={r['markets'].get('btts', {}).get('pick', '?'):>5s}"
              f"{picks_txt}{tag}")
    if out_path:
        # SEAL-V1: seal the kit + the inputs + THESE results (anti-fabrication)
        seal = make_seal(
            [Path(c) for c in csvs],
            params={"mode": "csv", "after": str(after_ts.date()),
                    "limit": limit, "min_conf": min_conf,
                    "odds_as_of": odds_as_of},
            matches=reports)
        Path(out_path).write_text(json.dumps(
            {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "engine_version": ENGINE_VERSION,
             "trained_through": TRAINED_THROUGH,
             "after": str(after_ts.date()), "n": len(reports),
             "seal": seal, "matches": reports}, indent=1, default=str))
        print(f"full report -> {out_path}")
        # readable companion: the same matches, human-format, final picks
        # with their confidence ratings + XAI summaries
        txt_path = str(out_path)
        if txt_path.lower().endswith(".json"):
            txt_path = txt_path[:-5]
        txt_path += ".txt"
        sep = "\n\n" + "#" * 78 + "\n\n"
        Path(txt_path).write_text(
            f"PARLAY ENGINE V1 - READABLE MATCH REPORTS\n"
            f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{len(reports)} matches | trained through {TRAINED_THROUGH}\n"
            + sep.join(match_report_text(r) for r in reports)
            + "\n\n" + _seal_footer(seal, out_path) + "\n")
        print(f"readable report (final picks + confidence + XAI) -> {txt_path}")
        print(f"SEAL::V1::{seal['run_id']}  "
              f"(verify: python 01_ENGINE.py --check {out_path})")
    return reports


# ====================================================== single-match mode ====
# A fixture CSV with ANY SUBSET of the columns below (one row per match) is a
# complete input: absent columns become NaN, and every model tolerates missing
# values. The rolling-history features (form10 / venue / h2h / xg5 /
# elo_delta / congestion) are only computable when past matches are supplied
# via --history-csv; without it they stay NaN (documented, tolerated).
FIXTURE_COLUMNS = [
    # identity (match_date + home_team + away_team are the only required ones)
    "match_date", "home_team", "away_team", "competition", "Country", "FTR",
    # table position / elo / points (league-table snapshot)
    "home_elo", "away_elo", "elo_diff",
    "home_position", "away_position",
    "home_points", "away_points",
    "home_games_played", "away_games_played",
    "home_goal_diff", "away_goal_diff",
    # last-5 form (count + averages; the engine derives rates)
    "home_form_5_wins", "home_form_5_draws", "home_form_5_losses",
    "away_form_5_wins", "away_form_5_draws", "away_form_5_losses",
    "home_avg_goals_scored_5", "home_avg_goals_conceded_5",
    "away_avg_goals_scored_5", "away_avg_goals_conceded_5",
    # schedule / rest
    "home_days_since_last_match", "away_days_since_last_match",
    "home_short_turnaround", "away_short_turnaround",
    "home_midweek_match", "away_midweek_match",
    "matchday_number", "match_day_of_week", "is_weekend", "kickoff_hour",
    # xG (optional; feeds the 5-match xG features when history is supplied)
    "xg_home", "xg_away",
    # result (history rows only - leave EMPTY on fixture rows)
    "full_time_home_goals", "full_time_away_goals",
    # 1X2 odds - any subset; fallback chain Pinnacle -> B365 -> Max -> WH ->
    # IW -> Avg picks the first complete book
    "PSH", "PSD", "PSA", "B365H", "B365D", "B365A",
    "MaxH", "MaxD", "MaxA", "WHH", "WHD", "WHA", "IWH", "IWD", "IWA",
    "AvgH", "AvgD", "AvgA",
    "BWH", "BWD", "BWA", "VCH", "VCD", "VCA", "LBH", "LBD", "LBA",
    "CLH", "CLD", "CLA",
    # Asian handicap line + odds
    "AHh", "PAHH", "PAHA", "B365AHH", "B365AHA", "AvgAHH", "AvgAHA",
    # O/U 2.5 odds
    "P>2.5", "P<2.5", "B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5",
    # BTTS odds
    "btts_yes_odds_betexplorer", "btts_no_odds_betexplorer",
    # Betfair exchange odds
    "BFH", "BFD", "BFA",
    # referee profile (optional)
    "referee_career_matches", "referee_career_avg_yellows",
    "referee_career_avg_reds", "referee_career_home_win_pct",
    "referee_season_matches", "referee_season_avg_yellows",
]

_COVERAGE_GROUPS = [
    ("odds & market", ("log_odds_", "imp_", "overround", "src_", "mk_")),
    ("elo / table", ("elo_home", "elo_away", "elo_diff", "pos_",
                     "ppg_", "gd_", "games_")),
    ("last-5 form", ("win_rate_", "draw_rate_", "form_pts", "scored_",
                     "conceded_", "attack_", "form_diff")),
    ("history (needs --history-csv)", ("form10_", "venue_", "h2h_", "goals10_",
                                       "att10_", "elo_delta_", "congestion_",
                                       "xg_for5_", "xg_against5_")),
    ("schedule / rest", ("rest_", "short_turn", "midweek_", "matchday",
                         "day_of_week", "is_weekend", "kickoff_hour")),
    ("referee", ("ref_",)),
    ("country", ("c_",)),
]


def _input_coverage(xrow):
    """How much of the 135-feature vector is actually filled (single-match
    honesty block: shows which feature groups the prediction is missing)."""
    s = pd.Series(np.asarray(xrow, dtype=float).ravel(), index=FEATURE_COLUMNS)
    filled = s.notna()
    groups = {}
    claimed = set()
    for gname, prefixes in _COVERAGE_GROUPS:
        cols = [c for c in FEATURE_COLUMNS
                if c not in claimed
                and any(c.startswith(p) or c == p for p in prefixes)]
        claimed.update(cols)
        groups[gname] = {"filled": int(filled[cols].sum()), "total": len(cols)}
    rest = [c for c in FEATURE_COLUMNS if c not in claimed]
    if rest:
        groups["other"] = {"filled": int(filled[rest].sum()), "total": len(rest)}
    n = int(filled.sum())
    core_prefixes = ("form10_", "venue_", "goals10_", "att10_", "xg_for5_",
                     "xg_against5_", "elo_delta_")
    core_cols = [c for c in FEATURE_COLUMNS
                 if any(c.startswith(p) for p in core_prefixes)]
    core_filled = int(filled[core_cols].sum())
    out = {"n_filled": n, "n_total": len(FEATURE_COLUMNS),
           "pct": 100.0 * n / max(1, len(FEATURE_COLUMNS)), "groups": groups,
           "history_core_filled": core_filled, "notes": []}
    if core_filled == 0:
        out["notes"].append(
            "no history supplied - form10/venue/h2h/xG5 features are NaN; "
            "add --history-csv (recent matches per team) to activate them")
    if groups.get("odds & market", {}).get("filled", 0) == 0:
        out["notes"].append(
            "no odds supplied - the 1X2 ensemble's book member and the "
            "market features are NaN; fill PSH/PSD/PSA (or AvgH/AvgD/AvgA) "
            "for materially better predictions")
    return out


def _fixture_frame(path, for_history=False):
    """Read a fixture/history CSV and coerce it to the engine's input schema.

    Missing columns become NaN; elo_diff / match_day_of_week / is_weekend are
    auto-derived when absent. Returns (DataFrame, label) ready to be stacked.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FAIL: file not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reindex(columns=FIXTURE_COLUMNS)      # absent -> NaN, extra dropped
    for c in ("match_date", "home_team", "away_team"):
        if df[c].isna().all():
            raise SystemExit(
                f"FAIL: {p.name}: required column '{c}' is empty/missing")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    bad = int(df["match_date"].isna().sum())
    if bad:
        raise SystemExit(
            f"FAIL: {p.name}: {bad} row(s) have an unparseable match_date")
    # friendly auto-derivations (only where the CSV left them blank)
    need = df["elo_diff"].isna() & df["home_elo"].notna() & df["away_elo"].notna()
    df.loc[need, "elo_diff"] = df.loc[need, "home_elo"] - df.loc[need, "away_elo"]
    need = df["match_day_of_week"].isna()
    df.loc[need, "match_day_of_week"] = df.loc[need, "match_date"].dt.dayofweek
    need = df["is_weekend"].isna()
    df.loc[need, "is_weekend"] = (df.loc[need, "match_day_of_week"] >= 4).astype(float)
    if for_history:
        miss_ftr = df["FTR"].isna()
        if miss_ftr.any():
            raise SystemExit(
                f"FAIL: {p.name}: history rows need FTR (H/D/A) - "
                f"{int(miss_ftr.sum())} row(s) missing it")
        miss_g = df["full_time_home_goals"].isna() | df["full_time_away_goals"].isna()
        if miss_g.any():
            raise SystemExit(
                f"FAIL: {p.name}: history rows need final scores - "
                f"{int(miss_g.sum())} row(s) missing them")
    return df


def write_features_template(path="features_template.csv"):
    """Write a fill-in fixture CSV: header + one EXAMPLE row (delete it)."""
    ex = {
        "match_date": "2026-10-27", "home_team": "Example Home FC",
        "away_team": "Example Away FC", "competition": "Bundesliga",
        "Country": "Germany", "home_elo": 1650, "away_elo": 1600,
        "home_position": 8, "away_position": 5, "home_points": 21,
        "away_points": 26, "home_games_played": 12, "away_games_played": 12,
        "home_goal_diff": 4, "away_goal_diff": 9,
        "home_form_5_wins": 2, "home_form_5_draws": 1, "home_form_5_losses": 2,
        "away_form_5_wins": 3, "away_form_5_draws": 1, "away_form_5_losses": 1,
        "home_avg_goals_scored_5": 1.4, "home_avg_goals_conceded_5": 1.2,
        "away_avg_goals_scored_5": 1.8, "away_avg_goals_conceded_5": 1.0,
        "home_days_since_last_match": 6, "away_days_since_last_match": 6,
        "matchday_number": 13, "kickoff_hour": 19,
        "PSH": 3.10, "PSD": 3.45, "PSA": 2.28,
        "AHh": -0.25, "AvgAHH": 1.92, "AvgAHA": 1.98,
        "Avg>2.5": 1.85, "Avg<2.5": 2.05,
        "btts_yes_odds_betexplorer": 1.80, "btts_no_odds_betexplorer": 2.00,
    }
    row = {c: ex.get(c, "") for c in FIXTURE_COLUMNS}
    out = Path(path)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"fixture template -> {out}")
    print("  one row per match; fill what you know, leave the rest empty;")
    print("  required: match_date, home_team, away_team. Recommended: 1X2")
    print("  odds (PSH/PSD/PSA or AvgH/AvgD/AvgA), AHh, elo, last-5 form.")
    print("  Then: python 01_ENGINE.py --features-csv " + out.name)
    print("  Optional: --history-csv <recent matches CSV, same columns + FTR")
    print("  + final scores> activates form10/venue/h2h/xG5 features.")
    return out


def register_template(path="register_template.json"):
    """Write a schema-correct Adjustment Register skeleton (PART S)."""
    ids = sorted(MODULE_CATALOG)
    tpl = {
        "_how_to": [
            "One JSON file per matchday. Fill 'adjustment_register' with the",
            "sourced claims you verified THIS turn (real retrieval, not memory).",
            "Every entry is verified fail-closed: unknown module_id, missing",
            "URL/snippet, T4/T5-only basis or delta_pp above the module max",
            "-> the entry is REJECTED and the reason lands in the audit.",
            "Valid module_id values: " + ", ".join(ids),
            "Delete every '_...' comment key before use. Flags are optional.",
        ],
        "league_tier": "A",
        "phase6_status": "NOT_RUN",
        "odds_as_of": None,
        "adjustment_register": [
            {
                "_comment": "FORMAT EXAMPLE - replace with real sourced claims",
                "module_id": "M3",
                "module_name": "Lineup Impact Model",
                "market": "x12",
                "target": "home_win",
                "direction": "UP",
                "delta_pp": 2.0,
                "sources": [
                    {"url": "https://www.example-real-news-site.com/...",
                     "tier": "T1", "age_days": 1,
                     "snippet": "exact quoted sentence from the source page"},
                ],
                "rule0_satisfied": True,
                "notes": "what happened, in one line",
            },
        ],
        "flags": [],
        "observed_line_moves": {},
        "data_integrity": {"total_data_points": 0, "verification_rate_pct": 0,
                           "unknown_data_points": 0, "t1_t2_count": 0,
                           "t3_count": 0, "t4_t5_count": 0},
    }
    out = Path(path)
    out.write_text(json.dumps(tpl, indent=1))
    print(f"register template -> {out}")
    print("  fill adjustment_register with sourced, verified claims (PART S);")
    print(f"  valid module_id values: {', '.join(ids)}")
    return out


def run_features_csv(engine, fixture_csv, history_csv=None, min_conf=0.0,
                     out_path=None, register=None, register_by_match=None,
                     odds_as_of=None, limit=None):
    """Price fixture rows from a single-match CSV (no master CSVs needed).

    With --history-csv, the past matches are stacked BEFORE the fixtures so
    the rolling-history features (form10 / venue / h2h / xG5 / elo_delta /
    congestion) are computed exactly as in training.
    """
    _resolve_trained_through(engine)
    fix = _fixture_frame(fixture_csv, for_history=False)
    frames = [fix]
    if history_csv:
        hist = _fixture_frame(history_csv, for_history=True)
        frames.insert(0, hist)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("match_date", kind="stable").reset_index(drop=True)
    payload = build_features(None, use_cache=False, frame=combined)
    y = np.asarray(payload["y"])
    rows = np.where(y == -1)[0]          # unplayed rows = the fixtures
    if limit:
        rows = rows[:int(limit)]
    if len(rows) == 0:
        print("no unplayed fixture rows found - leave FTR/goals EMPTY on "
              "fixture rows")
        return None
    n_hist = len(payload["y"]) - len(rows)
    print(f"pricing {len(rows)} fixture row(s)"
          + (f" with {n_hist} history row(s) behind them" if n_hist else "")
          + " ...")
    reports = price_payload_rows(
        engine, payload, rows, min_conf=min_conf, register=register,
        register_by_match=register_by_match, odds_as_of=odds_as_of)
    X = payload["X_B_XL"]
    for i, r in zip(rows, reports):
        r["input_coverage"] = _input_coverage(X.iloc[i].to_numpy(dtype=float))
    n_ok = sum(1 for r in reports for v in r["markets"].values()
               if v.get("status") == "ok")
    n_fail = sum(1 for r in reports for v in r["markets"].values()
                 if v.get("status") == "FAIL")
    n_picks = sum(len(r.get("final_picks", [])) for r in reports)
    print(f"done: {n_ok} market predictions served, {n_fail} FAILs, "
          f"{len(engine.fails)} artifact failures, {n_picks} final picks")
    for r in reports[:25]:
        n_p = len(r.get("final_picks", []))
        picks_txt = f" picks={n_p}" if n_p else ""
        print(f"  {r['date']} {r['home'][:20]:20s} v {r['away'][:20]:20s}"
              f" x12={r['markets'].get('x12', {}).get('pick', '?'):>3s}"
              f" ou25={r['markets'].get('ou25', {}).get('pick', '?'):>6s}"
              f" btts={r['markets'].get('btts', {}).get('pick', '?'):>5s}"
              f"{picks_txt}")
    if out_path:
        # SEAL-V1: seal the kit + the inputs + THESE results (anti-fabrication)
        seal = make_seal(
            [Path(p) for p in (fixture_csv, history_csv) if p],
            params={"mode": "features-csv", "min_conf": min_conf,
                    "odds_as_of": odds_as_of, "limit": limit},
            matches=reports)
        Path(out_path).write_text(json.dumps(
            {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "engine_version": ENGINE_VERSION,
             "trained_through": TRAINED_THROUGH,
             "mode": "features-csv",
             "n": len(reports), "seal": seal, "matches": reports},
            indent=1, default=str))
        print(f"full report -> {out_path}")
        txt_path = str(out_path)
        if txt_path.lower().endswith(".json"):
            txt_path = txt_path[:-5]
        txt_path += ".txt"
        sep = "\n\n" + "#" * 78 + "\n\n"
        Path(txt_path).write_text(
            f"PARLAY ENGINE V1 - READABLE MATCH REPORTS (single-match mode)\n"
            f"generated {time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{len(reports)} matches\n"
            + sep.join(match_report_text(r) for r in reports)
            + "\n\n" + _seal_footer(seal, out_path) + "\n")
        print(f"readable report (final picks + confidence + XAI) -> {txt_path}")
        print(f"SEAL::V1::{seal['run_id']}  "
              f"(verify: python 01_ENGINE.py --check {out_path})")
    return reports


# ============================================================ verify-env ====
REQUIREMENTS_TXT = """\
# parlay_engine_v1 - serving environment pin
#
# Core serving path (the 21 market models):
numpy>=2.1          # tested 2.1.3
pandas>=2.2         # tested 2.2.3
catboost>=1.2.8     # tested 1.2.8 (.cbm native format)
lightgbm>=4.5       # tested 4.5.0 (text models, cross-version safe)
xgboost>=2.1        # tested 2.1.3 (json booster, cross-version safe)
#
# Soft layer's sklearn artifacts (lr_b, stack meta, isotonic x2). scikit-learn
# has NO pickle compatibility across versions: under a different version those
# 4 artifacts FAIL loudly (never silently substituted); every market model
# keeps serving.
scikit-learn==1.7.2
joblib>=1.3         # tested 1.5.3
scipy>=1.11
#
# Python 3.10+ (tested 3.12)
"""

FITTED = {"numpy": "2.1.3", "pandas": "2.2.3", "catboost": "1.2.8",
          "lightgbm": "4.5.0", "xgboost": "2.1.3"}


def verify_env() -> int:
    """The kit's environment verifier (absorbed from 00_SETUP_ENV.py).

    Verify-only: no installs. Checks, proves by loading, and tells you
    exactly what to install when something is missing or mismatched.
    """
    print("=" * 74)
    print("PARLAY_ENGINE_V1 - ENVIRONMENT VERIFIER (--verify-env)")
    print("=" * 74)
    print(f"python: {sys.version.split()[0]}  |  kit: {ROOT}")
    problems, warns, vers = [], [], {}

    # ---------------- step 1: core dependencies ----------------
    print("\nStep 1: core dependencies")
    for mod in ("numpy", "pandas", "catboost", "lightgbm", "xgboost"):
        try:
            m = __import__(mod)
            vers[mod] = m.__version__
            print(f"  PASS  {mod:10s} {m.__version__}")
        except Exception as e:
            vers[mod] = None
            print(f"  FAIL  {mod:10s} not importable ({type(e).__name__})")
            problems.append(f"pip install {mod}")
    for mod, v in vers.items():
        if v and FITTED[mod].split(".")[:2] != v.split(".")[:2]:
            warns.append(f"{mod} {v} differs from the fitted {FITTED[mod]} "
                         f"(major.minor) - formats are native/cross-version, "
                         f"but re-verify with: python 01_ENGINE.py --sanity")

    # ---------------- step 2: kit bundles + extraction ----------------
    print("\nStep 2: kit bundles (self-extracting .py model files)")
    bundles = find_bundles()
    if not bundles:
        problems.append("no *MODELS*.py bundle files found next to this file")
        print("  FAIL  no *MODELS*.py bundles found")
    else:
        covered = set()
        for b in bundles:
            print(f"  PASS  {b.name}  {b.stat().st_size / 1e6:.1f} MB")
            try:
                covered |= set(_bundle_module(b).FILES)
            except Exception as e:
                problems.append(f"bundle {b.name} failed to import: {e}")
                print(f"  FAIL  {b.name}: {e}")
        for need, why in (("champion_cat.cbm", "1X2 champion"),
                          ("softlayer/blend_weights.json", "soft layer"),
                          ("allmarkets/htft.txt", "HT/FT market")):
            if need not in covered:
                problems.append(f"no bundle covers {need} ({why})")
        if "allmarkets/cs.txt" not in covered:
            warns.append("correct-score bundle missing - the cs market will "
                         "FAIL (the other 20 markets + champion serve)")
            print("  WARN  no bundle covers allmarkets/cs.txt")
        try:
            ensure_extracted(verbose=False)
            n = len([p for p in MODELS_DIR.rglob("*") if p.is_file()
                     and p.name not in (".extracted.json", "metadata.json")])
            print(f"  PASS  models_v1/ ready ({n} artifacts)")
        except Exception as e:
            problems.append(f"extraction failed: {e}")
            print(f"  FAIL  extraction: {e}")

    # ---------------- step 3: sklearn pin ----------------
    print("\nStep 3: scikit-learn pin (soft layer)")
    try:
        import sklearn
        pin_md = MODELS_DIR / "softlayer" / "sklearn_meta.json"
        pin = (json.loads(pin_md.read_text())["sklearn_fitted_version"]
               if pin_md.exists() else "1.7.2")
        if sklearn.__version__ == pin:
            print(f"  PASS  runtime sklearn {sklearn.__version__} == fitted {pin}")
        else:
            warns.append(f"runtime sklearn {sklearn.__version__} != fitted "
                         f"{pin}: the soft layer's sklearn artifacts FAIL by "
                         f"design (stack_lr, cat_b_iso, blend_wf_iso, lr_b "
                         f"member); all 21 market models still serve. Fix: "
                         f"pip install 'scikit-learn=={pin}'")
            print(f"  WARN  sklearn {sklearn.__version__} != fitted {pin}")
    except Exception as e:
        warns.append(f"scikit-learn not importable ({type(e).__name__}) - "
                     "soft-layer sklearn artifacts unavailable; markets serve")
        print("  WARN  scikit-learn missing")

    # ---------------- step 4: per-family load proofs ----------------
    print("\nStep 4: per-family load proofs")
    if problems:
        print("  (skipped - fix Step 1/2 first)")
    else:
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(MODELS_DIR / "champion_cat.cbm"))
            p = m.predict_proba(np.zeros((2, 135), dtype=float))
            assert p.shape == (2, 3)
            print("  PASS  catboost: champion .cbm loads + predict_proba")
        except Exception as e:
            problems.append(f"catboost probe failed: {e}")
            print(f"  FAIL  catboost probe: {e}")
        try:
            import lightgbm as lgb
            b = lgb.Booster(model_file=str(MODELS_DIR / "allmarkets"
                                            / "htft.txt"))
            b.predict(np.zeros((1, 135), dtype=float))
            print("  PASS  lightgbm: htft text model loads + predict")
        except Exception as e:
            problems.append(f"lightgbm probe failed: {e}")
            print(f"  FAIL  lightgbm probe: {e}")
        try:
            import xgboost as xgb
            b = xgb.Booster(model_file=str(MODELS_DIR / "softlayer"
                                            / "xgb_b.json"))
            # the booster carries its training feature names: the DMatrix
            # must mirror them (plain numpy input is otherwise rejected)
            d = xgb.DMatrix(np.zeros((1, 135), dtype=float),
                            feature_names=b.feature_names)
            b.predict(d)
            print("  PASS  xgboost: xgb_b.json loads + predict")
        except Exception as e:
            problems.append(f"xgboost probe failed: {e}")
            print(f"  FAIL  xgboost probe: {e}")
        try:
            import sklearn as sk
            pin_md = MODELS_DIR / "softlayer" / "sklearn_meta.json"
            pin = (json.loads(pin_md.read_text())["sklearn_fitted_version"]
                   if pin_md.exists() else "1.7.2")
            if sk.__version__ != pin:
                print(f"  SKIP  sklearn joblib probe (version {sk.__version__} "
                      f"!= pin {pin} - expected FAIL, see Step 3)")
            else:
                import joblib
                obj = joblib.load(str(MODELS_DIR / "softlayer" / "lr_b.joblib"))
                obj.predict_proba(np.zeros((2, 135), dtype=float))
                print("  PASS  sklearn: lr_b.joblib unpickles + predict_proba")
        except Exception as e:
            warns.append(f"sklearn joblib probe failed: {e}")
            print(f"  WARN  sklearn joblib probe: {e}")

    # ---------------- verdict ----------------
    (ROOT / "requirements.txt").write_text(REQUIREMENTS_TXT)
    print("\n" + "=" * 74)
    if problems:
        print("VERDICT: ENV NOT READY")
        for p in problems:
            print(f"   - {p}")
        print("\nInstall, then re-run:  pip install -r requirements.txt")
        return 1
    status = "ok" if not warns else "ok-with-warnings"
    (ROOT / "_env_ready.json").write_text(json.dumps(
        {"status": status, "problems": problems, "warnings": warns,
         "python": sys.version.split()[0], "versions": vers}, indent=1))
    print("requirements.txt written")
    if warns:
        print("VERDICT: READY WITH WARNINGS")
        for w in warns:
            print(f"   ! {w}")
        print("\nnext: python 01_ENGINE.py --sanity  (markets serve; soft "
              "layer limited as warned)")
    else:
        print("VERDICT: ENV READY")
        print("next: python 01_ENGINE.py --sanity")
    return 0


# ============================================================ sanity ========
def sanity():
    print("=" * 74)
    print("PARLAY_ENGINE_V1 - STAGED SANITY CHECK")
    print("=" * 74)
    print(f"python {sys.version.split()[0]}  |  kit root {ROOT}")
    problems, warns = [], []

    # stage 0: extract the model bundles first (the pin metadata lives inside)
    try:
        if find_bundles():
            ensure_extracted(verbose=True)
    except Exception as e:                               # noqa: BLE001
        problems.append(f"extraction failed: {e}")

    # stage 1: environment report
    print("\n-- stage 1: environment")
    vers = {}
    for mod in ("numpy", "pandas", "catboost", "lightgbm", "xgboost",
                "sklearn", "joblib"):
        try:
            m = __import__(mod)
            vers[mod] = m.__version__
            print(f"   {mod:10s} {m.__version__}")
        except Exception as e:                               # noqa: BLE001
            vers[mod] = None
            print(f"   {mod:10s} MISSING ({type(e).__name__})")
    for core in ("numpy", "pandas", "catboost", "lightgbm", "xgboost"):
        if not vers.get(core):
            problems.append(f"missing core dependency: {core}")
    if not vers.get("sklearn"):
        warns.append("scikit-learn missing - soft-layer sklearn artifacts "
                     "will FAIL (markets unaffected)")
    else:
        md = MODELS_DIR / "softlayer" / "sklearn_meta.json"
        fitted = (json.loads(md.read_text())["sklearn_fitted_version"]
                  if md.exists() else "?")
        if vers["sklearn"] != fitted:
            warns.append(f"runtime sklearn {vers['sklearn']} != fitted "
                         f"{fitted} - the 4 sklearn soft-layer artifacts "
                         "FAIL by design; all 21 markets still serve")

    # stage 2: kit integrity
    print("\n-- stage 2: kit integrity")
    bundles = find_bundles()
    covered = set()
    if not bundles:
        problems.append("no *MODELS*.py model bundles found next to this file")
    else:
        for b in bundles:
            print(f"   bundle: {b.name} ({b.stat().st_size / 1e6:.1f} MB)")
            try:
                covered |= set(_bundle_module(b).FILES)
            except Exception as e:
                problems.append(f"bundle {b.name} failed to import: {e}")
        if "champion_cat.cbm" not in covered:
            problems.append("no bundle covers the 1X2 champion")
        if "allmarkets/cs.txt" not in covered:
            warns.append("correct-score bundle not found - the correct-score "
                         "market will FAIL (all other markets serve)")
    if problems:
        print("\nVERDICT: NOT READY")
        for p in problems:
            print(f"   - {p}")
        return 1
    try:
        ensure_extracted()
        md = json.loads((MODELS_DIR / "metadata.json").read_text())
        n_files = len(md.get("files", {}))
        print(f"   extracted {n_files} artifacts, manifest read OK")
    except Exception as e:                                   # noqa: BLE001
        problems.append(f"extraction/manifest failed: {e}")

    # stage 3: per-family load proofs
    print("\n-- stage 3: per-family load proofs")
    eng = Engine().load_all()
    lr_obj = eng.soft.get("lr_b", (None, None))
    fam = {"catboost": eng.champion is not None,
           "lightgbm": any(k == "lgbm" for k, _ in eng.models.values()),
           "xgboost": eng.soft.get("xgb_b", (None, None))[1] is not None,
           "sklearn": all(x is not None for x in
                          (lr_obj[1], eng.stack_meta, eng.iso_cat,
                           eng.iso_blend))}
    for f, ok in fam.items():
        print(f"   {f:10s} {'OK' if ok else 'FAIL'}")
        if not ok and f != "sklearn":
            problems.append(f"{f} artifacts failed to load")
    if not fam["sklearn"]:
        warns.append("sklearn artifacts unavailable (version pin) - "
                     "stack_lr / iso variants FAIL, markets unaffected")
    print(f"   artifact fails: {eng.fails or 'none'}")

    # stage 4: every market serves on a synthetic row
    print("\n-- stage 4: all-market smoke predict")
    xrow = demo_row()
    rep = eng.price_row(
        xrow, odds3=[3.1, 3.4, 2.3], ah_line=-0.75,
        odds_ou25=[1.85, 2.05], odds_btts=[1.80, 2.00],
        odds_ah=[-0.75, 1.92, 1.98],
        soft_ctx={"soft_register": DEMO_REGISTER,
                  "competition": "Premier League",
                  "kickoff_utc": None,
                  "odds_as_of": DEMO_REGISTER.get("odds_as_of"),
                  "match_id": "sanity"})
    n_ok = n_fail = 0
    for mid, blk in sorted(rep["markets"].items()):
        if blk.get("status") == "ok":
            s = sum(blk["p"])
            good = abs(s - 1.0) < 5e-3
            n_ok += int(good)
            if not good:
                problems.append(f"market {mid} probabilities sum {s:.4f}")
        else:
            n_fail += 1
            print(f"   {mid:9s} FAIL: {blk.get('reason', '?')[:70]}")
    print(f"   {n_ok} markets OK, {n_fail} FAIL "
          f"(cs requires the 04 bundle; sklearn pins affect only soft layer)")

    # stage 5: soft layer variants
    print("\n-- stage 5: soft layer")
    for name, v in sorted(rep["soft_layer"].items()):
        st = v.get("status")
        print(f"   {name:13s} {st}")
        if st != "ok" and vers.get("sklearn") and name in (
                "stack_lr", "cat_b_iso", "blend_wf_iso"):
            continue    # covered by the sklearn warn
    ok_var = sum(1 for v in rep["soft_layer"].values() if v.get("status") == "ok")
    if ok_var < 3:
        problems.append(f"soft layer: only {ok_var}/6 variants serve")

    # stage 6: final picks
    print("\n-- stage 6: final picks")
    picks = rep.get("final_picks", [])
    ok = ({p["market"] for p in picks} == set(PICK_MARKETS) and all(
        p["gate"] == GATES.get(p["market"])
        and p["confidence"] in ("HIGH", "MEDIUM", "LOW")
        for p in picks))
    print(f"   {len(picks)} picks, every one with gate + confidence rating: "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        problems.append("final picks missing or malformed")

    # stage 7: soft-factor layer + final picks
    print("\n-- stage 7: soft-factor layer (demo register)")
    sf = rep.get("soft_factors", {})
    ok = sf.get("entries_verified", 0) >= 1 and sf.get("applied") is True
    print(f"   register audit: {sf.get('entries_verified')}/"
          f"{sf.get('entries_total')} verified, "
          f"{len(sf.get('adjustments', []))} audited: {'OK' if ok else 'FAIL'}")
    if not ok:
        problems.append("soft-factor register audit failed")
    ok = "dnb" in rep["markets"] and rep["markets"]["dnb"].get("gate") == 0.60
    print(f"   dnb gate label present: {'OK' if ok else 'FAIL'}")
    if not ok:
        problems.append("dnb market missing its headline gate")
    ok = all(p.get("xai_summary") for p in picks)
    print(f"   XAI summary present on every final pick: "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        problems.append("missing XAI summary on a final pick")
    txt = match_report_text(rep)
    ok = all(s in txt for s in ("[1]", "[2]", "[3]", "[4]", "[5]"))
    print(f"   readable report sections [1]..[5]: {'OK' if ok else 'FAIL'}")
    if not ok:
        problems.append("readable report missing sections")

    print("\n" + "=" * 74)
    if problems:
        print("VERDICT: NOT READY")
        for p in problems:
            print(f"   - {p}")
        return 1
    if warns:
        print("VERDICT: OPERATIONAL WITH WARNINGS")
        for w in warns:
            print(f"   ! {w}")
        return 0
    print("VERDICT: FULLY OPERATIONAL")
    print("   next: python 01_ENGINE.py --demo | --selftest | --csv ...")
    return 0


# ============================================================ self-test =====
def selftest():
    """40 adversarial checks - the engine's contract with itself."""
    print("=" * 74)
    print("PARLAY_ENGINE_V1 - ADVERSARIAL SELF-TEST (40 checks)")
    print("=" * 74)
    results = []

    def check(name, cond, info=""):
        results.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  ({info})" if info else ""))

    eng = Engine().load_all()
    ev = load_kit_evidence()

    # 1 champion loads with the right class order (after load_model the
    # runtime classes_ are positional [0,1,2]; the labels live in the meta
    # and the order was validated bit-for-bit against the source engine)
    check("champion classes == [A, D, H]",
          eng.champion is not None
          and list(eng.champion_meta.get("classes", [])) == CLASSES
          and list(np.atleast_1d(getattr(eng.champion, "classes_",
                                        [0, 1, 2]))) == [0, 1, 2])

    # 2 feature width
    check("feature width == 135",
          FEATURE_COLUMNS is not None and len(FEATURE_COLUMNS) == 135,
          f"{len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0}")

    # 3 all 20 all-market models loaded
    cs_present = (MODELS_DIR / "allmarkets" / "cs.txt").exists()
    expect_models = 20 if cs_present else 19
    check(f"{expect_models} all-market models loaded",
          len(eng.models) == expect_models, f"{len(eng.models)}")

    # 4 soft-layer members (lr_b requires the sklearn pin; a documented
    # mismatch downgrades the expectation to 4 + a skip note)
    import sklearn as _sk
    _pin_ok = (_sk.__version__ == eng.sklearn_fitted())
    n_members = sum(1 for m in ("cat_tuned", "lgbm_b", "xgb_b", "lr_b")
                    if eng.soft.get(m) is not None
                    and eng.soft[m][1] is not None) + (eng.champion is not None)
    expect_members = 5 if _pin_ok else 4
    check(f"{expect_members} soft-layer members available",
          n_members == expect_members, f"{n_members}"
          + ("" if _pin_ok else " (lr_b skipped: sklearn pin mismatch)"))

    # 5 blend weights parse + sum to ~1
    ws = sum(eng.blend_weights.values()) if eng.blend_weights else 0
    check("blend weights sum ~ 1", abs(ws - 1.0) < 0.01, f"{ws:.4f}")

    # 6 demo row: all markets sum to 1
    rep = eng.price_row(demo_row(), odds3=[3.1, 3.4, 2.3], ah_line=-0.75)
    sums = [sum(v["p"]) for v in rep["markets"].values()
            if v.get("status") == "ok"]
    expect_served = 21 if (MODELS_DIR / "allmarkets" / "cs.txt").exists() else 20
    check("all served markets sum to 1",
          len(sums) >= expect_served and all(abs(s - 1) < 5e-3 for s in sums),
          f"{len(sums)} markets")

    # 7 dnb directions are consistent
    dnb = rep["markets"].get("dnb", {})
    dd = dnb.get("directions", {})
    ph = dd.get("dnb_home", {}).get("p")
    pa = dd.get("dnb_away", {}).get("p")
    check("dnb directions p_home + p_away == 1",
          ph is not None and pa is not None and abs(ph + pa - 1) < 1e-9,
          f"{ph} + {pa}")

    # 8 AH cover + push + cover_away == 1
    ah = rep["markets"].get("ah", {})
    tot = (ah.get("cover_home", 0) + ah.get("push", 0)
           + ah.get("cover_away", 0))
    check("AH cover+push+cover_away == 1", abs(tot - 1) < 5e-3, f"{tot:.4f}")

    # 9 AH settlement arithmetic on a constructed distribution
    classes = eng.allmeta["markets"]["ah"]["classes"]
    def _idx(v):
        return classes.index(v) if v in classes else classes.index(str(v))
    p = np.zeros(len(classes))
    p[_idx(2)] = 0.5      # margin = +2 (home by 2)
    p[_idx(0)] = 0.5      # margin = 0 (draw)
    pw = pp = 0.0
    t = -(-1.5)                       # line -1.5 -> t = 1.5
    for k, mcls in enumerate(classes):
        mv = int(mcls)
        if mv == 5:
            pw += p[k]
        elif mv == -5:
            pass
        elif mv > t:
            pw += p[k]
        elif mv == t:
            pp += p[k]
    check("AH line -1.5: margin +2 covers, draw loses",
          abs(pw - 0.5) < 1e-9 and abs(pp - 0.0) < 1e-9, f"cover {pw}")

    # 10 confidence rating boundaries (0.50 / headline gate)
    ok10 = (confidence_rating(0.62, "x12") == "HIGH"
            and confidence_rating(0.55, "x12") == "MEDIUM"
            and confidence_rating(0.45, "x12") == "LOW"
            and confidence_rating(0.52, "ah") == "MEDIUM"
            and confidence_rating(0.60, "ah") == "HIGH")
    check("confidence rating boundaries (LOW < 0.50, MEDIUM < 0.60, HIGH)",
          ok10)

    # 11 every pick market serves with gate + confidence on the demo row
    picks1 = rep.get("final_picks", [])
    ok11 = ({p["market"] for p in picks1} == set(PICK_MARKETS) and all(
        p["gate"] == GATES[p["market"]]
        and p["confidence"] in ("HIGH", "MEDIUM", "LOW")
        and isinstance(p.get("odds"), (float, type(None)))
        for p in picks1))
    check("final picks: every modeled market serves with gate + confidence",
          ok11, f"{len(picks1)} picks")

    # 12 single-match templates: fixture schema + register skeleton
    ok12 = ({"match_date", "home_team", "away_team"} <= set(FIXTURE_COLUMNS)
            and len(FIXTURE_COLUMNS) >= 90)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tpl_path = CACHE_DIR / "_selftest_register.json"
    try:
        register_template(str(tpl_path))
        loaded = json.loads(tpl_path.read_text())
        ok12 = ok12 and (loaded["adjustment_register"][0]["module_id"]
                         in MODULE_CATALOG)
    except Exception:                                          # noqa: BLE001
        ok12 = False
    check("templates: fixture columns + register schema valid", ok12,
          f"{len(FIXTURE_COLUMNS)} fixture columns")

    # 13 isotonic calibration is monotone (skips under a documented pin
    # mismatch - the artifacts fail loudly by design, see FAIL semantics)
    if eng.iso_cat is not None:
        grid = np.linspace(0.05, 0.95, 19)
        key = 2 if 2 in eng.iso_cat else "2"
        cal = np.array([eng.iso_cat[key].predict([g])[0] for g in grid])
        check("isotonic calibration monotone (class H)",
              bool(np.all(np.diff(cal) >= -1e-12)))
    elif not _pin_ok:
        check("isotonic calibration monotone (class H)", True,
              "SKIP - sklearn pin mismatch, artifacts FAIL by design")
    else:
        check("isotonic calibration monotone (class H)", False,
              "sklearn artifacts unavailable")

    # 14 tuned blend renormalizes without odds
    sl = rep["soft_layer"].get("blend_wf", {})
    check("tuned blend serves without odds (renormalized)",
          sl.get("status") == "ok")

    # 15 gate labels match the headline gates
    ok15 = True
    for mid, blk in rep["markets"].items():
        if blk.get("status") == "ok" and "gate" in blk:
            want = ("STRONG" if blk["confidence"] >= blk["gate"] else "ADVISORY")
            if blk.get("gate_label") != want:
                ok15 = False
    check("gate labels consistent (STRONG/ADVISORY)", ok15)

    # 16 FAIL semantics: a broken artifact path fails loudly, never crashes
    e2 = Engine()
    broken = e2._load_joblib(MODELS_DIR / "softlayer" / "nope.joblib",
                             "softlayer/nope.joblib")
    check("FAIL semantics: missing artifact -> None + registry entry",
          broken is None and "softlayer/nope.joblib" in e2.fails)

    # ---------------- soft-factor layer + bet contract ----------------
    proc = SoftFactorProcessor()

    def reg_entry(module, market, target, delta, sources, **kw):
        e = {"module_id": module, "market": market, "target": target,
             "direction": kw.pop("direction", "UP"), "delta_pp": delta,
             "sources": sources, "rule0_satisfied": True}
        e.update(kw)
        return e

    SRC1 = [{"url": "https://www.bbc.co.uk/sport/football",
             "tier": "T1", "age_days": 1,
             "snippet": "the club captain returns to full training and starts "
                        "the weekend fixture after a month out injured"}]
    SRC2 = [{"url": "https://www.theguardian.com/football",
             "tier": "T2", "age_days": 1,
             "snippet": "the captain is back in the squad and expected to "
                        "start the weekend league fixture, the manager said"}]

    # 17 GATES covers every market id
    check("GATES covers every market id",
          all(GATES.get(mid) is not None
              for mid in eng.allmeta.get("markets", {})),
          f"dnb gate = {GATES.get('dnb')}")

    # 18 unknown module rejected (market alias normalization also shown)
    ok, why = proc.verify_entry(reg_entry("M99", "O/U 2.5", "over", 1.0, SRC1))
    check("verification: unknown module rejected + alias normalization",
          (not ok) and norm_market("O/U 2.5") == "ou25", why[:50])

    # 19 FLAG_ONLY module rejected (M8 referee on outcome markets)
    ok, why = proc.verify_entry(reg_entry("M8", "x12", "home_win", 1.0, SRC1))
    check("verification: FLAG_ONLY module rejected (M8)", not ok, why[:50])

    # 20 placeholder source URL rejected
    ok, why = proc.verify_entry(reg_entry(
        "M3", "x12", "home_win", 1.0,
        [{"url": "https://example.com/news", "tier": "T1", "age_days": 1,
          "snippet": "a snippet that is long enough to pass token checks"}]))
    check("verification: placeholder URL rejected", not ok, why[:50])

    # 21 delta beyond the documented range rejected (M3 max 5.0pp)
    ok, why = proc.verify_entry(reg_entry("M3", "x12", "home_win", 6.0,
                                          SRC1 + SRC2))
    check("verification: delta beyond max_pp rejected", not ok, why[:50])

    # 22 multiplier chain: T1 single source -> claim x 1.00 x 1.00 x 0.70
    eff, parts = proc.effective_delta_parts(
        reg_entry("M3", "x12", "home_win", 2.0, SRC1))
    check("multiplier chain: single fresh T1 source = claim x 0.70",
          abs(eff - 1.40) < 1e-9 and parts["n_corroboration_units"] == 1,
          f"eff {eff:.3f}")

    # 23 RULE 24: syndicated rewires collapse to one corroboration unit
    eff2, parts2 = proc.effective_delta_parts(
        reg_entry("M3", "x12", "home_win", 2.0, SRC1 + [
            {"url": "https://www.mirror.co.uk/sport",
             "tier": "T2", "age_days": 1, "snippet": SRC1[0]["snippet"]}]))
    check("RULE 24: identical snippets on 2 domains = 1 unit (x 0.70)",
          abs(eff2 - 1.40) < 1e-9 and parts2["n_corroboration_units"] == 1,
          f"eff {eff2:.3f}")

    # 24 RULE 10: tier B claim 3.0pp without fresh sources -> capped to 2.0
    capped, status = proc.rule10_cap(3.0, "B", {"_fresh": 0}, "NOT_RUN")
    check("RULE 10: band-2 claim capped to the full-apply cap",
          abs(capped - 2.0) < 1e-9 and "CAPPED" in status, status[:40])

    # 25 RULE 10: beyond the band-3 max -> REJECTED (DIVERGENCE_UNRESOLVED)
    capped, status = proc.rule10_cap(12.0, "B", {"_fresh": 2}, "NOT_RUN")
    check("RULE 10: beyond band-3 rejected outright",
          capped == 0.0 and "REJECTED" in status, status[:40])

    # 26 RULE 20: odds snapshot >= 6h after the news -> AUTO_YES 0.10
    import datetime as _dt
    _now = _dt.datetime.now(_dt.timezone.utc)
    disc, state = proc._priced_in_discount(
        reg_entry("M3", "x12", "home_win", 2.0, SRC1), _now, _now)
    check("RULE 20: odds 24h after 1d-old news -> AUTO_YES x0.10",
          state == "AUTO_YES" and abs(disc - 0.10) < 1e-9, f"{state} x{disc}")

    # 27 collision: same family + same target keeps the largest delta
    reg = {"adjustment_register": [
        reg_entry("M3", "x12", "home_win", 2.0, SRC1 + SRC2),
        reg_entry("M3", "x12", "home_win", 1.5, SRC2),
    ]}
    base = {"x12": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}}
    res = proc.process(reg, base, "Premier League")
    check("collision: same family+target keeps the largest claim",
          len(res["collisions_removed"]) == 1
          and abs(res["audited_delta_pp"].get("x12:home_win", 0) - 2.0) < 1e-9,
          f"kept {res['audited_delta_pp'].get('x12:home_win')}")

    # 28 RULE 22: ou25:over + btts:yes from ONE family -> smaller haircut x0.5
    reg = {"adjustment_register": [
        reg_entry("M3", "ou25", "over", 2.0, SRC1 + SRC2),
        reg_entry("M3", "btts", "yes", 1.0, SRC1 + SRC2),
    ]}
    base = {"ou25": {"over": 0.5, "under": 0.5},
            "btts": {"yes": 0.5, "no": 0.5}}
    res = proc.process(reg, base, "Premier League")
    hc = [c for c in res["coherence"] if c["rule"] == "SHARED_SIGNAL_HAIRCUT"]
    check("RULE 22: shared-signal haircut on the smaller delta",
          len(hc) == 1 and abs(res["audited_delta_pp"].get("btts:yes", 0)
                               - 0.5) < 1e-9,
          f"btts audited {res['audited_delta_pp'].get('btts:yes')}")

    # 29 1X2 full delta delivery
    fin = apply_soft_factors(
        {"x12": {"classes": CLASSES, "p": [0.3, 0.3, 0.4]}},
        {"x12:home_win": 3.0})
    check("1X2 application: +3pp audited delivers exactly +0.03",
          abs(fin["x12"][2] - 0.43) < 1e-9 and abs(sum(fin["x12"]) - 1) < 1e-9,
          f"p_home {fin['x12'][2]:.4f}")

    # 30 overflow guard: a huge delta on a 0.97 target never breaks the simplex
    # (contract: sum == 1, adjusted target <= q_cap = 1 - 0.01*n_other, and
    # every outcome stays >= the 0.005 floor)
    fin = apply_soft_factors(
        {"x12": {"classes": CLASSES, "p": [0.01, 0.02, 0.97]}},
        {"x12:home_win": 3.0})
    check("overflow guard: simplex holds, no outcome below the floor",
          abs(sum(fin["x12"]) - 1) < 1e-6 and fin["x12"][2] <= 0.98 + 1e-9
          and min(fin["x12"]) >= 0.005 - 1e-9,
          f"sum {sum(fin['x12']):.4f} max {max(fin['x12']):.4f} "
          f"min {min(fin['x12']):.4f}")

    # 31 binary delivery: ou25 over +2pp moves p[1] by exactly +0.02
    fin = apply_soft_factors(
        {"ou25": {"classes": ["no", "yes"], "p": [0.45, 0.55]}},
        {"ou25:over": 2.0})
    check("binary application: ou25 over +2pp delivers exactly +0.02",
          abs(fin["ou25"][1] - 0.57) < 1e-9 and abs(fin["ou25"][0] - 0.43) < 1e-9,
          f"p_over {fin['ou25'][1]:.4f}")

    # 32 AH settlement: line -5.5, margin +5 bucket must NOT cover
    classes_ah = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    p_ah = np.zeros(11); p_ah[10] = 0.6; p_ah[5] = 0.4

    def _ah_settle(p_vec, line):
        t = -float(line)
        fav_x75 = abs(t - round(t)) == 0.25 and round(t) > t
        pw = pp = 0.0
        for k, mv in enumerate(classes_ah):
            if fav_x75 and mv == int(round(t)):
                pw += 0.5 * p_vec[k]
                pp += 0.5 * p_vec[k]
            elif mv == 5:
                if t < 5 - 1e-9:
                    pw += p_vec[k]
                elif abs(t - 5.0) < 1e-9:
                    pp += p_vec[k]
            elif mv == -5:
                if t > -5 + 1e-9:
                    pass
                elif abs(t + 5.0) < 1e-9:
                    pp += p_vec[k]
                else:
                    pw += p_vec[k]
            elif mv > t + 1e-9:
                pw += p_vec[k]
            elif abs(mv - t) < 1e-9:
                pp += p_vec[k]
            elif abs(t - round(t)) == 0.25:
                cands = [c for c in (t - 0.25, t + 0.25)
                         if abs(c - round(c)) < 1e-9]
                if cands and int(round(cands[0])) == mv:
                    pp += 0.5 * p_vec[k]
        return pw, pp

    pw, pp = _ah_settle(p_ah, -5.5)
    check("AH settlement: line -5.5, margin +5 loses (censored bucket)",
          abs(pw) < 1e-9 and abs(pp) < 1e-9, f"cover {pw:.3f}")

    # 33 AH quarter-ball: line -0.25, margin 0 -> half-push mass 0.5 x P
    pw, pp = _ah_settle(p_ah, -0.25)
    check("AH settlement: quarter line -0.25 gives half-push on margin 0",
          abs(pw - 0.6) < 1e-9 and abs(pp - 0.2) < 1e-9,
          f"cover {pw:.3f} push {pp:.3f}")

    # 34 the pick contract: probability + gate + confidence + XAI everywhere,
    # and no value/staking fields anywhere in the output
    rep2 = eng.price_row(
        demo_row(), odds3=[3.1, 3.4, 2.3], ah_line=-0.75,
        odds_ou25=[1.85, 2.05], odds_btts=[1.8, 2.0],
        odds_ah=[-0.75, 1.92, 1.98],
        soft_ctx={"soft_register": DEMO_REGISTER,
                  "competition": "Premier League",
                  "kickoff_utc": None,
                  "odds_as_of": DEMO_REGISTER.get("odds_as_of"),
                  "match_id": "selftest"})
    picks2 = rep2.get("final_picks", [])

    def _no_banned(d):
        if isinstance(d, dict):
            return (not any(any(b in str(k).lower() for b in ("kelly", "value",
                                                              "stake", "ev_"))
                             for k in d)
                    and all(_no_banned(v) for v in d.values()))
        if isinstance(d, list):
            return all(_no_banned(v) for v in d)
        return True

    ok34 = len(picks2) == len(PICK_MARKETS) and all(
        0.0 < p["p_final"] <= 1.0
        and p["gate"] == GATES[p["market"]]
        and p["confidence"] in ("HIGH", "MEDIUM", "LOW")
        and p.get("xai_summary") for p in picks2)
    ok34 = ok34 and _no_banned(rep2)
    txt2 = match_report_text(rep2)
    ok34 = ok34 and all(s in txt2 for s in ("[1]", "[2]", "[3]", "[4]", "[5]"))
    check("pick contract: prob + gate + confidence + XAI; no EV/Kelly fields",
          ok34, f"{len(picks2)} picks")

    # 35 AH favorite-side x75 line: the boundary margin half-covers + half-pushes
    p_x75 = np.zeros(11)
    p_x75[classes_ah.index(1)] = 0.3    # margin +1: half win, half push
    p_x75[classes_ah.index(0)] = 0.3    # margin 0: full lose
    p_x75[classes_ah.index(-1)] = 0.4   # margin -1: full lose
    pw, pp = _ah_settle(p_x75, -0.75)
    check("AH settlement: favorite x75 line -0.75 half-covers margin +1",
          abs(pw - 0.15) < 1e-9 and abs(pp - 0.15) < 1e-9,
          f"cover {pw:.3f} push {pp:.3f}")

    # 36 pick sides: yes-only markets always sit on the yes/win side; two-
    # sided binary markets take the argmax; htft / cs carry the model-only
    # basis note and their own base-rate gates
    def _side_ok(pk):
        mid = pk["market"]
        if mid in YES_ONLY_MARKETS:
            return pk["pick"] in ("yes", "win")
        if mid in ("htft", "cs"):
            return (pk.get("basis", "").startswith("model probabilities")
                    and pk["gate"] == GATES[mid])
        return True
    ok37 = all(_side_ok(p) for p in picks2)
    yes_picks = [p for p in picks2 if p["market"] in YES_ONLY_MARKETS]
    ok37 = ok37 and len(yes_picks) == len(YES_ONLY_MARKETS)
    check("pick sides: yes-only markets on the yes side; htft/cs model-only",
          ok37, f"{len(yes_picks)} yes-only picks")

    # 37 AI intuition block: reasoning scaffolding with an empty operator
    # slot - and it must NOT contain any model probability field
    ai = rep2.get("ai_intuition") or {}
    lean = ai.get("engine_lean") or {}
    ok38 = (ai.get("kind", "").startswith("AI INTUITION")
            and lean.get("side") in ("HOME", "DRAW", "AWAY", "NONE")
            and lean.get("strength") in ("STRONG", "MODERATE", "FAINT", "NONE")
            and isinstance(lean.get("drivers"), list)
            and ai.get("operator_pick", {}).get("pick") is None
            and ai.get("operator_pick", {}).get("reasoning") is None
            and "NOT from the model probabilities"
                in ai.get("operator_pick", {}).get("instruction", ""))
    ai_txt = json.dumps(ai)
    ok38 = ok38 and not any(b in ai_txt for b in ("p_final", "p_raw",
                                                  "kelly", "odds"))
    txt3 = match_report_text(rep2)
    ok38 = ok38 and "[4] AI INTUITION PICK" in txt3 \
        and "AI operator slot" in txt3
    check("AI intuition: digest + empty operator slot, no probabilities",
          ok38, f"lean {lean.get('side')} {lean.get('strength')}")

    # 38 single-match mode end-to-end on the generated fixture template
    _resolve_trained_through(eng)
    tpath = CACHE_DIR / "_selftest_fixture.csv"
    write_features_template(str(tpath))
    pay = build_features(None, use_cache=False,
                         frame=_fixture_frame(tpath))
    rows36 = np.where(np.asarray(pay["y"]) == -1)[0]
    rep3 = price_payload_rows(eng, pay, rows36) if len(rows36) else []
    ic = _input_coverage(
        pay["X_B_XL"].iloc[rows36[0]].to_numpy(dtype=float)) \
        if len(rows36) else {"n_filled": 0, "groups": {}}
    ok36 = (len(rows36) == 1 and rep3
            and len(rep3[0].get("final_picks", [])) == len(PICK_MARKETS)
            and ic["n_filled"] > 0
            and ic.get("history_core_filled") == 0)
    check("single-match mode: template -> features -> picks + coverage",
          ok36, f"{ic.get('n_filled', 0)}/135 features filled")

    # 39 SEAL-V1: run_id deterministic + sensitive to params and outputs
    rep3m = rep3 or [{"home": "T", "away": "T2", "final_picks": [
        {"market": "x12", "p_final": 0.61, "gate": GATES["x12"],
         "confidence": confidence_rating(0.61, "x12")}]}]
    s1 = make_seal([], params={"mode": "t"}, matches=rep3m)
    s2 = make_seal([], params={"mode": "t"}, matches=rep3m)
    s3 = make_seal([], params={"mode": "t2"}, matches=rep3m)
    s4 = make_seal([], params={"mode": "t"},
                   matches=rep3m + [{"home": "EXTRA"}])
    ok39 = (s1["run_id"] == s2["run_id"] and s1["run_id"] != s3["run_id"]
            and s1["run_id"] != s4["run_id"])
    check("SEAL-V1: run_id deterministic + sensitive to params/outputs",
          ok39, f"run_id {s1['run_id']}")

    # 40 SEAL-V1: genuine report verifies; edited + forged reports rejected
    genuine = {"engine_version": ENGINE_VERSION, "matches": rep3m,
               "seal": make_seal([], params={"mode": "selftest"},
                                 matches=rep3m)}
    tampered = json.loads(json.dumps(genuine, default=str))
    tampered["matches"][0]["final_picks"][0]["p_final"] = 0.9999
    forged = {"engine_version": ENGINE_VERSION, "matches": rep3m,
              "seal": {"protocol": SEAL_PROTOCOL,
                       "run_id": "deadbeefdeadbeef"}}
    unsealed = {"engine_version": ENGINE_VERSION, "matches": rep3m}
    ok40 = (not verify_seal(genuine)
            and bool(verify_seal(tampered))
            and bool(verify_seal(forged))
            and bool(verify_seal(unsealed)))
    check("SEAL-V1: genuine verifies; edited/forged/unsealed rejected",
          ok40, f"caught tampered={len(verify_seal(tampered))} "
                f"forged={len(verify_seal(forged))} "
                f"unsealed={len(verify_seal(unsealed))}")

    n_pass = sum(1 for _, ok in results if ok)
    print("-" * 74)
    print(f"{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILED:")
        for name, ok in results:
            if not ok:
                print(f"   - {name}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


# ============================================================ CLI ===========
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-env", action="store_true",
                    help="environment verifier (deps, bundles, sklearn pin, "
                         "per-family load proofs; writes requirements.txt)")
    ap.add_argument("--check", metavar="REPORT_JSON", default=None,
                    help="SEAL-V1: verify a report.json is a genuine, "
                         "unmodified ENGINE.py output (anti-fabrication); "
                         "works without loading the models")
    ap.add_argument("--sanity", action="store_true",
                    help="staged self-check of the whole kit")
    ap.add_argument("--selftest", action="store_true",
                    help="adversarial unit checks (40)")
    ap.add_argument("--demo", action="store_true",
                    help="price a synthetic match on every market (with the "
                         "built-in sample register -> soft factors, picks, "
                         "XAI)")
    ap.add_argument("--features-template", action="store_true",
                    help="write features_template.csv (single-match input) "
                         "and exit")
    ap.add_argument("--register-template", action="store_true",
                    help="write register_template.json (Adjustment Register "
                         "skeleton) and exit")
    ap.add_argument("--features-csv", default=None,
                    help="single-match mode: fixture CSV (any subset of the "
                         "template columns; get the template via "
                         "--features-template)")
    ap.add_argument("--history-csv", default=None,
                    help="optional recent-matches CSV for --features-csv "
                         "(same columns + FTR + final scores) - activates the "
                         "rolling-history features (form10/venue/h2h/xG5)")
    ap.add_argument("--csv", nargs="*", default=None,
                    help="master CSV path(s) (default: auto-discover ./upload/*.csv)")
    ap.add_argument("--after", default=None,
                    help="price matches strictly after this date")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="hide market picks below this confidence in output")
    ap.add_argument("--soft-register", default=None,
                    help="Adjustment Register JSON (single register, or "
                         '{"by_match": {"Home|Away|date": register}})')
    ap.add_argument("--odds-as-of", default=None,
                    help="ISO timestamp of the odds snapshot (enables RULE 20 "
                         "priced-in auto-detection)")
    ap.add_argument("--soft-log", default=None,
                    help="append the soft-layer shadow record to this JSONL "
                         "(calibration ledger; backfill 'won' per target)")
    ap.add_argument("--out", default=None, help="output JSON path (a readable "
                    "<out>.txt companion with the final picks + confidence "
                    "ratings is also written)")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the feature-pickle cache (rebuild from CSVs)")
    args = ap.parse_args()

    if args.verify_env:
        sys.exit(verify_env())
    if args.check:
        sys.exit(check_report(args.check))
    if args.sanity:
        sys.exit(sanity())
    if args.selftest:
        sys.exit(selftest())
    if args.features_template:
        write_features_template()
        sys.exit(0)
    if args.register_template:
        register_template()
        sys.exit(0)

    eng = Engine().load_all(verbose=True)
    _resolve_trained_through(eng)

    if args.demo:
        sys.exit(0 if run_demo(eng) is not None else 1)

    register, register_by_match = None, None
    if args.soft_register:
        try:
            register, register_by_match = load_register(args.soft_register)
        except Exception as e:                               # noqa: BLE001
            print(f"FAIL: could not load --soft-register {args.soft_register}: "
                  f"{type(e).__name__}: {e}")
            sys.exit(1)
        n = (len(register.get("adjustment_register", [])) if register
             else sum(len(r.get("adjustment_register", []))
                      for r in register_by_match.values()))
        print(f"soft register loaded: {n} adjustment entries "
              f"({'per-match' if register_by_match else 'global'})")

    reports = None
    if args.features_csv:
        reports = run_features_csv(
            eng, args.features_csv, history_csv=args.history_csv,
            min_conf=args.min_conf, out_path=args.out, register=register,
            register_by_match=register_by_match, odds_as_of=args.odds_as_of,
            limit=args.limit)
    else:
        csvs = args.csv if args.csv else discover_csvs()
        if not csvs:
            print("no CSVs found - put your master training CSVs in ./upload/ "
                  "or pass --csv path1.csv path2.csv\n"
                  "(single match? run --features-template, fill the row, then "
                  "--features-csv features_template.csv; or run --demo for a "
                  "synthetic smoke test)")
            sys.exit(1)
        reports = run_csv(eng, csvs, after=args.after, limit=args.limit,
                          min_conf=args.min_conf, out_path=args.out,
                          use_cache=not args.no_cache, register=register,
                          register_by_match=register_by_match,
                          odds_as_of=args.odds_as_of)
    # shadow ledger: one JSONL line per match with a register applied - the
    # calibration loop's raw material (backfill 'won' per target afterwards)
    if args.soft_log and reports:
        n_written = 0
        with open(args.soft_log, "a") as fh:
            for r in reports:
                sf = r.get("soft_factors") or {}
                shadow = sf.get("shadow_log")
                if not shadow:
                    continue
                shadow["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                shadow["match"] = {"date": r.get("date"), "home": r.get("home"),
                                   "away": r.get("away"),
                                   "competition": r.get("competition")}
                shadow["final_picks"] = [
                    {k: p.get(k) for k in ("market", "pick", "p_final")}
                    for p in r.get("final_picks", [])]
                fh.write(json.dumps(shadow, default=str) + "\n")
                n_written += 1
        print(f"soft shadow ledger: {n_written} records -> {args.soft_log} "
              "(backfill 'won' per target in per_target to calibrate)")


if __name__ == "__main__":
    main()
