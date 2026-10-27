# parlay_engine_v1 — Five-File QWEN Projects Kit (plain files, no zips)

A walk-forward-validated football prediction engine covering **21 bet
markets** with a live **1X2 ensemble soft layer**, the **soft-FACTOR layer**
(a 31-module LLM-compiled Adjustment Register: fail-closed verification,
measured caps, cross-market coherence), **a final pick on EVERY modeled
market — 21 of them** (each with its own headline gate — 0.60 for the
decision markets, the walk-forward base-rate gate for HT/FT 0.25 and
correct score 0.12 — a **HIGH / MEDIUM / LOW confidence rating** and an
exact **TreeSHAP XAI** summary), an **AI-intuition block** (the register
narrative digest + a reserved pick slot the running AI completes from its
own reasoning), **Asian-handicap per-line settlement** (incl. quarter-ball
half-push mass and censored-margin handling at extreme lines), and the
**embedded CSV → 135-feature pipeline** with a **single-match input mode** —
everything trained and measured under a strict no-leakage walk-forward
protocol (8 annual expanding-window folds, test 2019–2026, 96,222 matches,
seed 42). Probability and confidence only — no EV, no odds-value, no staking.

## The five files (QWEN Projects format: 5 files, ≤ 20 MB each, plain files)

| # | File | Role | Size |
|---|------|------|------|
| 0 | `parlay_engine_v1_00_README.md` | this guide (operating manual + soft-factor spec + module catalog) | ~27 KB |
| 1 | `parlay_engine_v1_01_ENGINE.py` | the complete engine: 21 markets + both soft layers + final picks on every market + confidence ratings + AI-intuition block + XAI + AH + feature pipeline + single-match mode + `--verify-env` / `--sanity` / `--selftest` (40 checks) / `--demo` + anti-fabrication `--check` | ~235 KB |
| 2 | `parlay_engine_v1_02_MODELS_CHAMPION_SOFTLAYER.py` | **self-extracting bundle**: the 1X2 champion + the complete ensemble soft layer (14 artifacts) + manifests + evidence | ~4.7 MB |
| 3 | `parlay_engine_v1_03_MODELS_19_MARKETS.py` | **self-extracting bundle**: the 18 CatBoost market models + the HT/FT LightGBM text model | ~14.5 MB |
| 4 | `parlay_engine_v1_04_MODELS_CORRECT_SCORE.py` | **self-extracting bundle**: the correct-score model — the one artifact that needs its own 20 MB file | ~19.6 MB |

**No zip archives anywhere.** The model payloads ship as ordinary Python
files: every artifact is xz-compressed, base85-encoded and embedded with its
exact sha256 + byte size. Bundles are stdlib-only and run on any Python
≥ 3.8; extraction is byte-identical, checksum-verified, idempotent, into
`models_v1/` (renaming the files works — bundles are located by the
`*MODELS*.py` pattern; keep all five files in one folder).

## Quick start

```bash
python parlay_engine_v1_01_ENGINE.py --verify-env   # deps, bundles, sklearn pin, load proofs
python parlay_engine_v1_01_ENGINE.py --sanity       # staged check: env -> integrity -> loads ->
                                                    #   all-market smoke -> soft layers -> picks
python parlay_engine_v1_01_ENGINE.py --demo         # synthetic match on EVERY market, with the
                                                    #   built-in sample register -> soft factors,
                                                    #   final picks + confidence + XAI
python parlay_engine_v1_01_ENGINE.py --selftest     # 40 adversarial checks

# SINGLE MATCH (no master CSVs needed) - the normal way to analyze one fixture:
python parlay_engine_v1_01_ENGINE.py --features-template
#   ... fill one row of features_template.csv (empty cells are OK), then:
python parlay_engine_v1_01_ENGINE.py --features-csv features_template.csv --out report.json
#   JSON + a readable report.txt (final picks + confidence + XAI) are written

# optionally activate the rolling-history features (form10 / venue / h2h / xG5):
#   put the recent played matches of both teams (same columns + FTR + scores)
#   into history.csv, then add:  --history-csv history.csv

# with an LLM-compiled Adjustment Register (PART S; --register-template writes a skeleton):
python parlay_engine_v1_01_ENGINE.py --features-csv features_template.csv \
    --history-csv history.csv \
    --soft-register register.json \
    --odds-as-of 2026-10-27T15:00:00Z \
    --soft-log shadow.jsonl \
    --out report.json

# batch mode (master training CSVs in ./upload/ or --csv):
python parlay_engine_v1_01_ENGINE.py --after 2026-10-16 --min-conf 0.60 --out report.json

# verify a report is a genuine, unmodified engine output (SEAL-V1):
python parlay_engine_v1_01_ENGINE.py --check report.json
```

### The single-match fixture CSV (`--features-csv`)

`--features-template` writes `features_template.csv`: every input column the
engine can consume + one EXAMPLE row (delete it). Rules of the road:

- **Required:** `match_date` (YYYY-MM-DD), `home_team`, `away_team`. Everything
  else is optional — absent columns/empty cells become NaN and every model
  tolerates missing values.
- **Fill for the best prediction:** 1X2 odds (`PSH/PSD/PSA` or
  `AvgH/AvgD/AvgA`), the AH line `AHh` (+ AH odds `AvgAHH/AvgAHA`),
  O/U 2.5 odds (`Avg>2.5` / `Avg<2.5`), BTTS odds
  (`btts_yes_odds_betexplorer` / `btts_no_odds_betexplorer`),
  `home_elo` / `away_elo`, last-5 form (`*_form_5_wins/draws/losses`,
  `*_avg_goals_scored_5` / `*_avg_goals_conceded_5`), table snapshot
  (`*_position`, `*_points`, `*_games_played`, `*_goal_diff`), rest days
  (`*_days_since_last_match`), `Country`, `competition`.
- **Auto-derived when left blank:** `elo_diff`, `match_day_of_week`,
  `is_weekend`.
- **History features** (`form10_*`, `venue_*`, `h2h_*`, `xg_for5_*`,
  `elo_delta_*`, …) need past matches: supply them with `--history-csv`
  (same columns, plus `FTR` and final scores, ~6–10 recent matches per team).
  Without them those features stay NaN — the prediction still serves, and the
  report's **input coverage** block shows exactly what was filled and what
  was missing.
- Never set `FTR`/scores on fixture rows (empty = unplayed). History rows
  MUST have both.

### Qwen sandbox survival rules

- Run the engine **as a script**: `python parlay_engine_v1_01_ENGINE.py ...
  The three `*MODELS*.py` bundles auto-extract next to it on first run —
  never run them via `subprocess`, never `open()`/read a bundle file (they
  are multi-MB payload carriers, not code to read).
- **NEVER edit, patch, regex-"fix" or re-save any of the five files.** They
  are checksum-verified as a unit; any modification invalidates the kit.
  If your interpreter reports an error, rerun; if a code cell keeps failing,
  restart the sandbox — the files themselves are valid Python.
- If `--verify-env` flags the sklearn pin, the four soft-layer sklearn
  artifacts fail loudly and the 21 markets keep serving — that is the
  documented behavior, not a bug to fix.
- **If you cannot execute code in this environment, reply exactly
  `NO_SANDBOX_EXECUTION` and stop.** That is a valid, honest answer. Do NOT
  hand-write, estimate or "simulate" a report — see the seal section below.

### The genuine-run seal (SEAL-V1) — never trust an unsealed report

Every genuine run **seals** its output. `run_id` = sha256 over the ENGINE.py
bytes, the bundle bytes, the input CSV bytes, the gate table, the run
parameters **and the generated results**. Same kit + same inputs → same
`run_id`; a single edited probability breaks it.

- The seal is **printed to stdout** as `SEAL::V1::<run_id>`, **embedded in
  the JSON** (`"seal"`) and **appended to `report.txt`**.
- Verify any report: `python parlay_engine_v1_01_ENGINE.py --check
  report.json` → `SEAL VERIFIED` or `SEAL INVALID` (works without loading
  the models). It re-checks the kit hashes, re-derives the `run_id`, and
  re-verifies the output contract (probabilities in range, gates +
  confidence ratings consistent with the code).
- **Operator ritual:** always paste the terminal tail of the run — including
  the `SEAL::V1` line — into the chat next to the report. A report without
  its SEAL line pasted from a real terminal is void.
- A hand-written, edited or "estimated" report **cannot** carry a valid
  seal: editing one number breaks the outputs digest; inventing a report
  breaks the kit hashes and the `run_id`. `SEAL INVALID` = the file was
  never produced by (or was modified after) this engine. Never use it.
- The engine never fakes a run; no operator should. If a genuine run is
  impossible this turn, `NO_SANDBOX_EXECUTION` is the only honest output.
- The `ai_intuition.operator_pick` slot is completed **in the chat** (or a
  separate file) — never by editing `report.json`, which would break the
  seal and void the report.

## Environment

```
numpy>=2.1        pandas>=2.2        catboost>=1.2.8
lightgbm>=4.5     xgboost>=2.1       scikit-learn==1.7.2
joblib>=1.3       scipy>=1.11        Python 3.10+ (tested 3.12)
```

- **Core serving path** (all 21 market models + vote/532/tuned blends):
  numpy, pandas, catboost, lightgbm, xgboost. Native cross-version formats.
- **Soft layer's sklearn artifacts** (lr_b member, stack meta, isotonic ×2):
  scikit-learn **1.7.2** — under a different version those artifacts FAIL
  loudly with the reason; the engine never silently substitutes, and all 21
  markets keep serving. The soft-FACTOR layer needs **no extra
  dependencies** (numpy + stdlib only) and is unaffected by the pin.
- `--verify-env` checks all of this and writes `requirements.txt`. In the
  QWEN sandbox: sklearn 1.7.2 ships natively; `pip install catboost
  lightgbm xgboost` covers the rest.

## The 21 markets (walk-forward evidence, all out-of-sample)

General accuracy = argmax pick on every eligible test match; gated = picks
with confidence ≥ the headline gate. Baselines are honest.

| Market | Model | General acc (base) | @ headline gate: acc on coverage (lift) |
|--------|-------|--------------------|------------------------------------------|
| 1X2 `x12` | CatBoost+odds (champion) | 54.9% (book 55.0%) | @60: **74.2%** on 32.1% (+20.9pp) |
| O/U 2.5 `ou25` | CatBoost | 57.6% (51.2%) | @60: 63.1% on 35.5% (+9.4pp) |
| BTTS `btts` | CatBoost | 56.0% (55.1%) | @60: 62.6% on 28.9% (+0.1pp) |
| O/U 1.5 `ou15` | CatBoost | 75.6% (75.6%) | @60: 75.7% on 98.7% (high base) |
| O/U 3.5 `ou35` | CatBoost | 70.4% (70.0%) | @60: 72.2% on 88.2% (+0.2pp) |
| Home over 0.5 `tt_h05` | CatBoost | 76.4% (76.3%) | @60: 77.8% on 93.5% |
| Home over 1.5 `tt_h15` | CatBoost | 63.6% (58.3%) | @60: 68.9% on 60.5% (+9.8pp) |
| Away over 0.5 `tt_a05` | CatBoost | 69.8% (69.6%) | @60: 72.5% on 83.3% |
| Away over 1.5 `tt_a15` | CatBoost | 68.2% (65.2%) | @60: 71.9% on 77.6% (+3.6pp) |
| DC 1X `dc1x` | CatBoost | 71.9% (70.7%) | @60: 75.9% on 80.2% (+5.2pp) |
| DC X2 `dcx2` | CatBoost | 65.4% (58.9%) | @60: 71.7% on 61.8% (+12.8pp) |
| DC 12 `dc12` | CatBoost | 77.0% (77.0%) | @60: 77.1% on 99.4% (high base) |
| DNB `dnb` | CatBoost (draws void) | 68.8% settled | @60: **75.4%** / **73.4%** per direction |
| Win-to-nil home `wtn_h` | CatBoost (one-sided) | 76.4% incl. "no" | @60: 57.3% on 0.8% (base 23.8, +33.5pp) |
| Win-to-nil away `wtn_a` | CatBoost (one-sided) | 82.9% incl. "no" | @60: 58.9% on 0.7% (base 17.2, +41.7pp) |
| Corners O/U 9.5 `corners` | CatBoost | 53.8% (51.5%) | @60: 57.5% on 19.3% (+6.0pp) |
| Cards O/U 3.5 `cards` | CatBoost | 63.4% (61.9%) | @60: 67.7% on 67.2% (+2.9pp) |
| HT 1X2 `ht13` | CatBoost | 45.4% (41.1%) | @60: 63.7% on 3.6% (+10.6pp) |
| HT/FT `htft` | LightGBM (9 classes) | 29.7% (25.4%) | @25: 30.6% on 90.4% (+4.1pp) |
| Correct score `cs` | LightGBM (top-20 + other) | 11.8% (10.7%) | @12: 11.9% on 98.3% (+1.2pp) |
| Asian handicap `ah` | CatBoost margin (−5..5) | 45.5% settled (void 7.0%) | cover probabilities at the quoted line |

**Asian handicap.** The margin model's cover probabilities are settled
per-line (push mass and quarter-ball half-pushes included), and the final AH
pick is simply the higher-cover side at the quoted line with its confidence
rating. In the archived walk-forward, gating the same model on its cover
probability lifted settled accuracy sharply (65.7% hit on the @60 band) —
small samples at the highest bands, flagged as such.

## The soft layer, part 1 — the 1X2 ensemble

Five live variants on top of the champion (plus two isotonic calibrations):

| Variant | Construction | Walk-forward log loss |
|---------|--------------|----------------------|
| `cat_b` (champion) | CatBoost 350/0.06/d6 + odds | 0.93042 |
| `vote_gbm` | soft vote of XGB + LGBM + CAT | 0.93244 |
| `blend_532` | 0.5·LGBM + 0.3·XGB + 0.2·CAT | 0.93528 |
| `blend_wf` | tuned weights over [book, LR, LGBM, XGB, CAT, CAT-tuned] | 0.93018 |
| `stack_lr` | LR meta on 12 OOF columns (4 bases × 3 classes) | 0.93741 |
| `cat_b_iso` | per-class isotonic calibration of the champion | **0.92888** |
| `blend_wf_iso` | per-class isotonic calibration of the tuned blend | **0.92887** |

Production weights (simplex search on the archived walk-forward OOF, 52,057
rows): `cat_b 0.4892 · xgb_b 0.3001 · cat_tuned 0.1354 · lr_b 0.0544 ·
lgbm_b 0.0206 · book 0.0003`. Missing candidates renormalize with an
explicit note (never silent). Isotonic calibrators were OOF-fitted.

## The soft layer, part 2 — the soft-FACTOR layer

An LLM (the "news desk") compiles an **Adjustment Register** of sourced,
per-market probability deltas; the engine-side `SoftFactorProcessor`
**verifies → multiplies → collision-checks → coherences → caps → nets →
applies → logs** them. Trust boundary: the LLM compiles claims + evidence;
the CODE owns the catalog, floors, caps and application. Self-reported
`evidence_basis` / `causal_family` are ignored — the code-side catalog wins.

### PART S.1 — the Adjustment Register (input JSON)

Register-level fields:

```json
{
  "league_tier": "A",                    // optional; engine table wins on conflict
  "phase6_status": "SURVIVES",           // global Phase-6 fallback
  "adjustment_register": [ ...entries... ],
  "flags": [ ... ],                      // FSC flags, see S.5
  "observed_line_moves": {"x12:home_win": 1.5},   // pp the market already moved
  "data_integrity": {"total_data_points": 40, "verification_rate_pct": 95,
                     "unknown_data_points": 2, "t1_t2_count": 3,
                     "t3_count": 0, "t4_t5_count": 0}
}
```

Per entry (every field verified fail-closed):

```json
{
  "module_id": "M3", "module_name": "Lineup Impact Model",
  "market": "x12",                       // engine id, or alias ("O/U 2.5", "1X2", ...)
  "target": "home_win",
  "direction": "UP" | "DOWN",
  "delta_pp": 2.0,
  "sources": [ {"url": "https://...", "tier": "T1|T1_CLUB|T2|T2_DATA|T3|T4|T5",
                "age_days": 1, "snippet": "<exact quoted text>"} ],
  "rule0_satisfied": true,               // LLM certifies live retrieval this turn
  "market_priced_in": "YES|PARTIAL|NO",  // optional RULE 20 override
  "phase6": "SURVIVES|FAILED|NOT_RUN",   // optional per-entry override
  "model_incremental": false,            // true waives RULE 23 (warned + logged)
  "notes": "..."
}
```

**Verification (RULE 16, fail-closed — one miss rejects the entry):** known
module_id · ADJUST role · valid market/target · module's market restrictions
respected · ≥1 usable source (real non-placeholder URL **and** quoted
snippet) · RULE 16 corroboration floor counted in RULE 24 **deduped units**
(ESTABLISHED needs 1, MIXED 2, SPECULATIVE 3) · not a T4/T5-only basis ·
`delta_pp` finite and within the module's documented max.

### PART S.2 — the module catalog (31 modules)

M1–M23 carry the measured families, evidence bases, roles and caps of the
original catalog, **including the evidence-sweep corrections**: M8 referee
demoted to FLAG_ONLY on outcome markets (channel measured as noise,
crowd-mediated per Bilalić 2021), M9 weather 3.0pp (Kepler 6,108-match
study), M6 chemistry 2.0pp, M7/M7B H2H 2.0pp, M11B managerial 2.0pp
(Ter Weel: new-manager bounce = mean reversion).

M24–M31 extend coverage to the markets beyond 1X2 / O/U 2.5 / BTTS / DC /
AH: cards, corners, team totals, HT markets, win-to-nil. Each is
market-restricted and hand-set; the shadow ledger exists to re-measure them
from outcomes.

| Module | Name | Family | Basis | Role | max_pp | Markets | Evidence anchor |
|--------|------|--------|-------|------|-------:|---------|-----------------|
| M24 | Referee Card Profile | referee | ESTABLISHED | ADJUST | 3.0 | cards | career yellow averages are the standard cards predictor (data-guide consensus 2026); crowd–cards link measured (Downward 2007; Portsmouth 2022). Cards model already carries referee features → RULE 23 ×0.60 applies |
| M25 | Derby & Rivalry Intensity | H2H | MIXED | ADJUST | 2.0 | cards, corners | practitioner consensus, mixed literature — hand-set |
| M26 | Match-State Script & Chase | script | ESTABLISHED | ADJUST | 2.0 | corners, cards, ou25, ou35, tt_h15, tt_a15 | ~60% of corners arrive in the 2nd half with a chasing side (Opta, via SharkBetting 2026) |
| M27 | Set-Piece & Crossing Style | tactics | MIXED | ADJUST | 1.5 | corners | corners driven by flank/cross/press style (TheStatBible 2026; postudium 2026) — hand-set |
| M28 | Weather Extremes on Undercard Markets | environment | MIXED | ADJUST | 2.0 | ou15/25/35, corners, cards | M9 owns the goals markets; M28 covers wind/pitch on the rest (collision check keeps the pair honest) |
| M29 | VAR Official Profile | referee | MIXED | ADJUST | 1.5 | cards | 2× yellow-cards/minute after VAR interventions, d = 0.40 (Samuel et al. 2024, Int. J. Sports Science & Coaching) |
| M30 | Dead Rubber / Table Symmetry | motivation | ESTABLISHED | ADJUST | 2.0 | ou25, ou35, corners, cards, ht13 | dead rubbers significantly more random (Expecting Goals 2026) |
| M31 | Early-Goal Expectancy | script | MIXED | ADJUST | 1.5 | ht13, ou15 | ~74.8% of matches see a 1st-half goal; goals cluster late in halves (goalstatistics 100k-match DB) — hand-set |

`htft` (9 classes) and `cs` (21 classes) are **flag-only markets** —
individual class deltas are meaningless at pp scale; flags still apply.
Everything else accepts deltas on its documented targets (e.g. `ou25:
over|under`, `tt_h15: home_over|home_under`, `dnb: home|away`, `wtn_h:
yes|no`, `ah: home_cover|away_cover`).

### PART S.3 — the multiplier chain and the caps

Effective delta = `delta_pp × tier × recency × corroboration × priced_in ×
model_overlap` (credible sources only — T4/T5 dilute nothing and amplify
nothing):

- **Tier weights** (measured table): T1 1.00 · T1_CLUB 0.80 · T2 0.80 ·
  T2_DATA 1.00 **for stats modules only** (M1, M3B, M4, M4B, M4D, M5G, M7,
  M7B, M7C, M11E, M13, M13B, M14B, M16, M22, M24, M26, M27, M31) · T3 0.85 ·
  T4 0.50 · T5 0.30.
- **Recency**: interpolated anchors 0d→1.00, 3d→1.00, 7d→0.85, 14d→0.65,
  30d→0.30, floor 0.30. RULE 21: lineup/fatigue/motivation/psychology/script
  evidence also decays with distance to kickoff (early news is not late team
  news).
- **Corroboration** (after RULE 24 syndication dedup — near-identical
  snippets across domains collapse to ONE unit): 1 unit 0.70 · 2 → 1.00 ·
  3+ → 1.20.
- **RULE 20 priced-in** (measured: an absorbed move keeps 2–20% residual):
  explicit declaration > three-way auto-detection from `--odds-as-of` vs the
  freshest source age — odds ≥ 6h after the news → **YES ×0.10**; within
  ±6h → **PARTIAL ×0.55**; before → **NO ×1.00**; unknown → 1.00 + warning.
- **RULE 23 model-overlap** (the 135 features already consume these
  channels): form ×0.70 · referee ×0.60 · fatigue ×0.85 · H2H ×0.80 ·
  script ×0.90 · everything else ×1.00. `model_incremental: true` waives it
  (warned, logged, measured by the ledger).
- **RULE 10 cap ladder** (MEASURED on 126k real Pinnacle morning→close line
  moves; full-apply = p60, band-2 = p90, band-3 = p99): tier A 1.5/4.0/7.5pp,
  tier B 2.0/5.0/9.0pp, tier C 2.5/6.0/10.5pp. Band-2 claims apply in full
  only with 2+ fresh (≤72h) credible sources, else capped to band-1.
  Band-3 needs Phase-6 SURVIVES (then 50%) — beyond band-3 the entry is
  REJECTED (`DIVERGENCE_UNRESOLVED`) and the market goes NO_BET.
- **Collisions**: same causal family claiming the same market:target keeps
  only the largest delta (self-reported families ignored).
- **RULE 22 cross-market coherence** (double-counting guards over the
  extended pair set): (a) a family's DC delta is mechanically capped at its
  own 1X2-implied mix + 0.3pp; (b) same-family same-sign deltas on a
  correlated pair take a ×0.50 haircut on the SMALLER side. Pair set:
  ou25↔btts (measured pair-stats lift 1.50–1.62), nested totals
  ou15↔ou25↔ou35, team-totals↔ou25, x12↔dnb, x12↔ht13, tt_a05:under↔wtn_h
  (win-to-nil mechanics), x12↔ah covers.
- **Total drift clip** at the tier's band-3 max; **observed line-move
  netting** subtracts same-direction market moves from the audited total;
  **transfer beta** ships at the measured parity ceiling 1.0 (the soft layer
  may never out-transfer sharp money; refittable from the ledger).

### PART S.4 — application (full-delta delivery, guarded)

Audited deltas are applied additively to the model probabilities with
full-delta delivery: a moved target receives the **whole** audited delta
(no `(1+d)` renormalization dilution) and the complementary mass is removed
proportionally, with an overflow guard so the vector always stays on the
simplex. AH deltas move `cover_home` (push mass kept). Every multiplier is
logged per entry (`multipliers` block) so the ledger can refit all of them,
not just beta.

### PART S.5 — flags, integrity gates, shadow ledger

- **FSC flags**: severity 0–5 clamped; unattributed flags clamped to 2;
  sev ≥ 4 requires a sourced URL (else clamped to 3); VARIANCE counts
  half (rounded up); scoping per market. FSC ≥ 10 on a market or any sourced
  sev-5 flag → **NO_PICK** for that market (JSON field `no_bet_markets`; no
  pick is emitted from it).
- **Data-integrity gates**: self-reported verification rate < 50% →
  DEVILS_VETO; unknown points > max(5, 5%) → DEVILS_VETO; engine-measured
  register verification rate < 50% (≥ 4 entries) → DEVILS_VETO; 50–70% →
  confidence penalty.
- **Shadow ledger** (`--soft-log shadow.jsonl`): one JSONL record per match
  with the full per-target baseline/delta/adjusted-probability breakdown.
  Backfill `won: true/false` per target after the matches; the 0pp bucket is
  the control group and every bucket's lift over it is the measured value of
  the soft layer.

## Final picks, gates and confidence ratings — what is emitted

1. **Final picks are emitted for every modeled market** (21) on the **final
   probability** (best-calibrated 1X2 ensemble variant + audited soft-factor
   deltas for x12; model + audited deltas on every market the soft-factor
   catalog covers; raw model probabilities for HT/FT and correct score — the
   `final_basis` / `probability_basis` field names it). Every pick always
   shows:
   - `p_final` — the final probability of the picked side;
   - `gate` — the market's **own headline gate**: 0.60 for the decision
     markets; HT/FT 0.25 and correct score 0.12 keep the walk-forward
     base-rate gates for their long-shot class spaces (a 0.60 bar there
     could never be met and would be a dishonest label);
   - `gate_label` — STRONG when p_final ≥ that market's gate, else ADVISORY;
   - `confidence` — the **HIGH / MEDIUM / LOW rating**: HIGH at or above the
     market's gate, MEDIUM from 0.50 up to it, LOW below 0.50. Nothing is
     suppressed: every pick is always computed and shown with its honest
     label.
2. **Pick sides.** Two-sided binary markets (O/U goals, team totals, corners,
   cards, BTTS, DNB, AH) take the higher-probability side. Yes-only markets —
   double chance 1X/X2/12 and win-to-nil — always sit on the YES/WIN side
   (the "no" side is not a standalone market) and are shown honestly even
   when that probability is below 0.50 (LOW / ADVISORY).
3. **Odds are reference-only.** When odds were supplied (1X2, O/U 2.5, BTTS,
   AH), each pick shows them beside the probability. No expected value is
   computed and no staking is suggested — the decision is yours.
4. `--min-conf` filters the market *display* only; picks are unaffected.
5. Soft-factor integrity: a market blocked by the FSC / DEVILS_VETO gates
   emits NO pick (listed as BLOCKED in the report), never a suppressed one.
6. AH picks additionally show `line` and `push_prob` — the settlement mass at
   the quoted line (cover + push + lose = 1).

## XAI — why each pick exists

Every pick carries an exact, model-native **TreeSHAP** decomposition
(CatBoost `ShapValues` for the CatBoost models, LightGBM `pred_contrib` for
HT/FT and correct score — no approximation, no extra dependency):

- `xai.top_support` / `xai.top_against` — the top features pushing the pick
  and the main counter-evidence, in margin points, with human-readable
  feature names (all 135 columns are labeled); binary UNDER/NO picks are
  explained on the negated margin so the summary always describes the
  *picked* side;
- the **soft-factor part** of the story: which verified modules moved this
  market, how much survived the caps, and under which priced-in state;
- the **evidence part**: the market's own walk-forward numbers (e.g. ou25
  @g60: 63.1% hit on 35.5% coverage);
- the **gate part**: p_final vs the market's own headline gate and the
  confidence rating.

All of it is composed into a one-paragraph `xai_summary` per pick — the
readable "why this pick" — and the structured `xai` block keeps the numbers
for programmatic use.

## Output format

`--out report.json` is **pure JSON**. A readable companion `report.txt` is
written next to it, and each match embeds the same text as `report_text`
inside the JSON.

Per match (JSON keys):

```jsonc
{
  "date": "...", "home": "...", "away": "...", "competition": "...",
  "in_sample_warning": false, "result": "H?",              // when played
  "model_fails": {},                                        // never silent
  "markets": { "x12": {"status":"ok","classes":["A","D","H"],"p":[...],
               "pick":"H","confidence":0.61,"gate":0.6,"gate_label":"...",
               "evidence":{...}}, "...": {"status":"FAIL","reason":"..."} },
  "soft_layer": { "blend_wf_iso": {"status":"ok","p":[...],"pick":"H"}, ... },
  "soft_factors": {
    "applied": true, "league_tier": "A", "entries_total": 3,
    "entries_verified": 2, "verification": [...], "adjustments": [...],
    "audited_delta_pp": {"x12:home_win": 0.2}, "coherence": [...],
    "collisions_removed": [...], "warnings": [...], "fsc": {...},
    "integrity_gates": {...},
    "final_probabilities": {...}, "final_confidence": {...},
    "final_pick": {...}, "final_basis": {...},
    "uncapped_delta_pp": {...}, "shadow_log": {...}
  },
  "final_picks": [
    {"market":"ou25","pick":"yes","pick_label":"OVER 2.5","odds":2.05,
     "p_raw":0.6787,"p_final":0.6787,"soft_total_pp":0.0,
     "gate":0.6,"gate_label":"STRONG","confidence":"HIGH",
     "basis":"final (calibrated ensemble + audited soft factors)",
     "evidence":"walk-forward @g60: ...","xai":{...},"xai_summary":"..."},
    {"market":"ah","pick":"away","pick_label":"AH AWAY +0.75","odds":1.80,
     "line":-0.75,"push_prob":0.08,"p_raw":0.7344,"p_final":0.7344,
     "soft_total_pp":0.0,"gate":0.6,"gate_label":"STRONG",
     "confidence":"HIGH","xai":{...},"xai_summary":"..."},
    "...": "21 picks in total - one per modeled market"
  ],
  "ai_intuition": {
    "kind": "AI INTUITION - a reasoning lean, NOT a probability-qualified pick",
    "engine_lean": {
      "side":"HOME","net_pp":2.0,"strength":"MODERATE",
      "basis":"net audited register narrative on the 1X2 targets",
      "net_by_target": {"home_win":2.0},
      "drivers":[{"module":"M3","target":"home_win","claim_pp":2.0,
                  "audited_pp":2.0,"status":"APPLIED_FULL","priced_in":false}],
      "contra":{"rejected_claims":1,"rejected_detail":[...],
                "priced_in_claims":0,"no_bet_markets":[],
                "fsc_veto":false,"artifact_fails":0}
    },
    "operator_pick": {
      "pick": null, "reasoning": null,
      "instruction": "The running AI completes this slot from its OWN reading
                      of the sourced evidence - NOT from the model
                      probabilities..."
    }
  },
  "input_coverage": {"n_filled":113,"n_total":135,"pct":83.7,
                     "groups":{...},"history_core_filled":21,
                     "notes":["..."]},                      // single-match mode
  "report_text": "..."                                      // readable
}
```

The readable companion (`report.txt`) is a fixed five-section template per
match — `[1] MARKET PROBABILITIES` (p_raw → p_final per market, the soft-pp
column and the confidence rating; plus the input-coverage block in
single-match mode), `[2] SOFT-FACTOR LAYER — AUDITED ADJUSTMENTS` (claim →
effective → audited with the full multiplier chain), `[3] FINAL PICKS`
(each of the 21 picks with probability, its market's gate, confidence rating
and XAI summary), `[4] AI INTUITION PICK` (the register narrative lean + the
`AI operator slot` the running AI completes with its own reasoning-based
pick), `[5] NOTES` — so the JSON and the human report can never disagree
about what was emitted.

The run envelope additionally carries **`"seal"`** (SEAL-V1): `run_id` plus
the sha256 of the engine file, the bundles, the inputs, the gate table and
the generated results — see *The genuine-run seal* above. `report.txt` ends
with the same `SEAL::V1::<run_id>` line. A report (or envelope) without a
valid seal is not engine output; `--check report.json` is the arbiter.

## The AI-intuition pick (reasoning, not probabilities)

The engine deliberately does NOT compute the intuition pick — it computes
probabilities, and an intuition must come from somewhere else. Instead it
emits an `ai_intuition` block per match:

- `engine_lean` — the net direction of the **audited register narrative** on
  the 1X2 targets (which sourced claims survived verification, their
  audited pp, which were priced-in or rejected) plus the contra-evidence.
  This is an aggregate of the *claims*, not of the model probabilities — the
  code never reads `p_final` while building it. With no register (or no net
  direction ≥ 1pp) the lean is NONE.
- `operator_pick` — the reserved slot the **running AI completes**: one pick
  that follows from its own reading of the sourced evidence (news,
  motivation, tactics, conditions, matchup), with the reasoning written out.
  It is labeled INTUITION, carries no probability, no gate and no staking,
  and may legitimately differ from the model's picks. Write "no lean" when
  nothing justifies one — an empty intuition is a valid answer.

In the readable report this is section `[4]`: the narrative lean, its
drivers, and the `pick = ______ / reasoning = ______` slot to fill.

**Where to complete it:** in the **chat** (or a separate file) — never by
editing `report.json` or `report.txt`. Editing the report breaks the
SEAL-V1 outputs digest and voids the whole report. The intuition is an
operator annotation on top of a sealed result, not part of it.

## Honest caveats

- The production models have seen every match through 2026-10-14 (champion)
  / 2026-10-16 (markets) — use them for fixtures AFTER that; earlier matches
  get `in_sample_warning: true`.
- The soft-FACTOR layer's multipliers and the M24–M31 ceilings are measured
  constants plus hand-set, evidence-anchored values — they are NOT yet
  measured on this engine's own outcomes. The shadow ledger (`--soft-log`)
  is the instrument that fixes that; treat the audited deltas as
  disciplined, capped opinions until the ledger has ~500 scored targets.
- The 0.60 headline gate is the walk-forward accuracy-lift point on
  champion probabilities; the picks' probability basis is the calibrated
  `blend_wf_iso` (equal-or-better out-of-sample: LL 0.92887 vs 0.93042) —
  a documented transfer of the gate, not a re-validation. Both bases are
  always printed.
- Single-match mode: without `--history-csv` the rolling-history features
  (form10 / venue / h2h / xG5 / elo_delta) are NaN. The models tolerate
  missing values, but predictions are materially better with the history
  CSV — check the input-coverage block before trusting a pick.
- AH cover probabilities inherit the model's censored margin support
  (classes −5..+5): margins beyond ±5 are not expressible by this model, so
  at `|line| ≥ 5` the settlement treats the ±5 bucket as push-or-lose
  (favorite side) per the documented censored-bucket compromise.
- High-confidence rows on thin markets (wtn, AH at extreme lines) are
  small-n — lean evidence.
- HT/FT and correct score remain genuinely hard (29.7% / 11.8% general) —
  their value is the distribution, not the argmax; they are flag-only for
  the soft-factor layer, and their picks sit at the base-rate gates
  (0.25 / 0.12) with the model-only basis note.
- The synthetic `--demo` row is NaN-heavy by design; ensemble members can
  disagree more on such inputs than on real feature rows.

## Kit versions

Engine `1.3.0` (kit `parlay_engine_v1`, plain-file format, probability +
confidence output, SEAL-V1 anti-fabrication seal), models `models_v1`. Soft-factor constants measured on
real line-move data (2026-09-17: 127,317 fixtures / 125,960 move legs; RULE
10 ladder from tier A n=27,423 / tier B n=41,095 Pinnacle moves). Models
fitted under catboost 1.2.8 / lightgbm 4.5.0 / xgboost 2.1.3 / scikit-learn
1.7.2 / numpy 2.1.3 / pandas 2.2.3 (Python 3.12).
