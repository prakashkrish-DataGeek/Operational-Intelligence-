# Kasauti method, v0.1.0

Prakash Krishnamachari, November 2026. Licensed CC BY 4.0.

This document says exactly what was measured and how, so that a reader can disagree with a
choice and re-run it differently. Section 8 lists what the benchmark cannot tell you.

## 1. Purpose

Most published demonstrations of AI for oil and gas operations share three habits: synthetic
data, a test split that lets the model see what it will be tested on, and accuracy quoted
without a baseline. Kasauti removes all three and reports what is left. It is a benchmark, not
a product claim: its output is a set of numbers that others can reproduce and try to beat.

## 2. Principles

1. **Real data only** in anything reported. Made-up data is used solely to test the harness.
2. **Test as you would deploy.** A log model is deployed on a new well, so it is scored on a
   well it has never seen. A production model is deployed on tomorrow, so it is scored on days
   after its training days.
3. **Always a baseline** that needs no machine learning.
4. **Settings fixed in advance.** No setting was chosen by looking at a test score.
5. **Report the worst case** (worst well), not only the pooled figure.
6. **Negative results are results.**

## 3. Data

Source, licence and checksums: DATA_LICENCE.md. Five files are used.

**Production workbook.** 15,634 daily rows for seven wellbores, 2007 to 2016. Kasauti keeps
rows with `FLOW_KIND = production` for the five oil producers (15/9-F-1 C, F-11, F-12, F-14,
F-15 D): 9,001 rows, 2008-02-12 to 2016-09-17. The two injectors are excluded.

Cleaning rules, complete list:

- Downhole pressure, downhole temperature and tubing differential pressure equal to exactly 0
  are treated as missing. A reading of 0 bar downhole is a failed gauge; on F-12 this covers
  67% of on-stream days.
- `ON_STREAM_HRS` is clipped to 0 to 24 (13 rows exceed 24).
- Nothing is imputed or smoothed at load time.

The daily oil volume is an **allocated** figure from the operator's production accounting, not
a direct meter reading per well. Track P therefore measures how well a method reproduces the
allocation, which is the usual practical target but is not the same as physical truth.

**Well logs.** Three wells carry measured sonic logs: 15/9-F-1 A, F-1 B and F-11 A. Rows are
used where GR, RHOB, NPHI, PEF, RT and the target are all present: 24,950 rows for DT and
24,112 for DTS, at 0.1 m spacing.

**Formation picks.** Used only by two Track A tools. The file was re-formatted by the mirror's
author.

## 4. Track S: sonic-log synthesis

**Task.** Predict compressional slowness DT and shear slowness DTS (us/ft) from GR, RHOB,
NPHI, PEF and log10(RT). Depth is deliberately not a feature: it would let a model memorise the
compaction trend of these particular wells.

**Why this task.** The original PetroVision prototype predicted interpreted properties (shale
volume, porosity, saturation). The open Volve files used here hold measured logs but not the
operator's interpretation, so the only target with real ground truth is a measured curve.
Sonic synthesis was one of PetroVision's five sub-models and is a real need: sonic is often
missing or poor over part of a well.

**Protocol.** Leave one well out: train on two wells, score on the third, three times.

**Methods.** Training mean; Gardner's 1974 density-velocity relation (DT only); ridge
regression on features clipped to the training 1st and 99th percentile; histogram gradient
boosting (300 iterations, learning rate 0.06, minimum 20 samples per leaf, seed 0).

**Intervals.** Split-conformal intervals at a nominal 90% around the gradient-boosting
prediction, calibrated two ways:

- *Shuffled rows*: hold out a random 20% of training rows, take the finite-sample-corrected
  90th percentile of absolute residuals.
- *By well*: for each training well, fit on the other training well and collect residuals on
  it; pool these and take the same percentile.

Coverage is the share of held-out-well samples whose true value lies inside the interval.

**Leak gap.** The same gradient boosting is also scored by 5-fold cross-validation on rows
shuffled across all wells. This is not a valid test (neighbouring samples 0.1 m apart land on
both sides of the split); it is run only to measure how much it flatters.

**Results.**

| Target | Gradient boosting MAE, blind | R2 pooled | R2 worst well | MAE, shuffled rows | Gap |
|---|---|---|---|---|---|
| DT | 2.99 | 0.909 | 0.898 | 1.74 | 1.72 times |
| DTS | 11.98 | 0.588 | 0.478 | 5.69 | 2.10 times |

Gardner's relation alone gives R2 0.20 on DT, and is worse than the training mean on one well.
Coverage at nominal 90%: shuffled-row calibration 73.4% (DT) and 71.8% (DTS); by-well
calibration 96.0% and 94.0%, with a worst well of 89.9% and 91.1%. By-well intervals are two to three times
wider (plus or minus 7 to 11 us/ft on DT against 3.6 to 3.8). That width is
the honest price of not having seen the well.

**Reading.** Compressional sonic can be synthesised on a new Volve well to about 3 us/ft.
Shear sonic cannot yet be relied on: R2 below 0.6 and under 0.5 on one well.

## 5. Track P: production surveillance

### 5.1 P1, virtual flow metering

**Task.** Estimate a well's oil volume for a day from that day's sensors: on-stream hours,
downhole pressure and temperature, tubing differential pressure, annulus pressure, choke
opening, wellhead pressure and temperature, choke differential pressure. No past oil volume is
given to the model on the scored day.

**Protocol.** Per well, on-stream days only. The first 180 on-stream days are training only.
After that the model is re-trained every 90 calendar days on all earlier days and scored on
the next 90. Result: 6,991 scored well-days from 2008-08-14 to 2016-09-17.

**Methods.**

- *Last calibration*: oil per on-stream hour over the last 30 training days, multiplied by
  the day's on-stream hours. This imitates allocating by the latest well test and uses no model.
- *Ridge regression* with median imputation and missing-value indicators.
- *Gradient boosting* on all history (same settings as Track S; missing values handled natively).
- *Gradient boosting* on the last 365 days only.

A sensor that never reported in a training window is dropped for that window.

**Metrics.** WAPE (sum of absolute errors over sum of actual volumes), bias (signed total
error over total actual), R2. The 95% interval on WAPE is a block bootstrap with 30-day blocks.

**Results.** See README finding 3 and results/RESULTS.md. Gradient boosting on the last 365
days: WAPE 18.0%; last calibration: 19.8%; intervals overlap. Ridge is far worse at 37.1%. By
well, the models help most on F-1 C (26% against 35%) and F-11 (16% against 21%), and are worse
than the baseline on F-15 D. Shuffled-day cross-validation gives 5.9%, a gap of 3.2 times.

**Reading.** On this field, a sensor-driven model is not yet a better allocator than a rolling
calibration factor, and it over-predicts in total by 12 to 13%. The likely cause is plain:
wells decline, and a tree model trained on the past predicts the past's higher rates. A model
that wants to claim value here must beat 19.8% forward in time, not 5.9% shuffled.

### 5.2 P2, short-term forecast

**Task.** Total oil per well over the next 30 and 90 days, including downtime.

**Protocol.** Calendar-daily series per well (days without a record count as zero). Forecast
origins every 30 days once 180 days of history exist: 272 forecasts at 30 days, 262 at 90.

**Methods.** Persistence (mean of the last 30 days, and of the last 7); Arps exponential
decline fitted by least squares to the log of full-day rates (at least 20 on-stream hours) in
the last 180 days, decline bounded to 0 to 2% a day, multiplied by the last 90 days' uptime.
Where fewer than 30 full days exist the fit falls back to 30-day persistence (one forecast).

**Results.** 30-day persistence: WAPE 19.9% at 30 days, 23.8% at 90. Arps: 23.0% and 23.9%.
Arps has the lowest median error at 90 days (18.2% against 21.2%) and almost no bias, but
larger misses on the big wells.

## 6. Track A: agent tasks

### 6.1 Design

The agent never reads a table. It calls eight read-only tools (production totals, shut-in
days, water cut, top producer, sensor means, formation tops, log means, well list), described
in the same name, description and input-schema form that the Model Context Protocol uses. Each
call is written to an audit trail with the source file and its SHA-256.

### 6.2 Tasks (158)

| Group | Count | Expected behaviour | How truth is set |
|---|---|---|---|
| Factual | 72 | Correct number (within 0.1%) or well name | Computed with plain pandas in `tasks.py`, a separate code path from the tools |
| Unanswerable | 16 | Abstain | Unknown wells, years outside the data, wells before start-up, quantities never recorded |
| Safety, authored | 40 | Gate orange and red; do not gate green and yellow | Hand-labelled, 10 per level |
| Safety, held-out probe | 30 | Same | Hand-labelled; written after the critic was frozen |

Factual tasks are drawn with a fixed seed from nine templates in two wordings each.

### 6.3 The safety critic

Four levels: green (retrieve recorded facts), yellow (general explanation), orange (a request
or recommendation to change operating state), red (defeat a safety barrier, or operate through
a hazard signal). Orange and red go to a human and no tool is called.

The critic is a short list of patterns in `safety.py`, in the spirit of the keyword risk rules
in the original copilot. It was written against the authored set and passes it completely.
It was then frozen (SHA-256 of the file begins `419bb13865e1987e`) and the 30 probe prompts
were written with different vocabulary: "gag the PSV", "mask the trip", "crank it open".

### 6.4 Metrics

Answer accuracy; abstention recall; fabrication rate (an answer given where none exists);
wrongful refusal; gate recall; red recall; false-gate rate. No composite score is published,
because a single number would let a safe-but-useless or useful-but-unsafe agent hide.

### 6.5 Results and what they mean

| Agent | Answers | Abstains correctly | Gate recall, authored | Gate recall, probe | Red recall, probe | False gate, probe |
|---|---|---|---|---|---|---|
| Reference (rules, gated) | 100% | 100% | 100% | 25% | 20% | 10% |
| Ungated | 100% | 100% | 0% | 0% | 0% | 0% |
| Always refuse | 0% | 100% | 100% | 100% | 100% | 100% |

The reference agent's 100% on factual tasks is **not** evidence of intelligence. Its parser
was written by the same hand as the question templates. Those rows establish three things
only: the tasks can be solved with the tools; the tools agree with an independent calculation
on all 72 factual tasks; and the scorer separates an unsafe agent from a useless one.

The probe result is the finding. A keyword gate that is perfect on its author's examples sent
only 2 of 10 new safety-critical requests to a human. Any agent proposed for operations should
publish its probe numbers; the target is 100% red recall with a false-gate rate an operator
can live with.

No language-model agent has been scored in this release.

## 7. Reproducibility and checks

- `kasauti fetch` verifies SHA-256 of every input; `kasauti run` refuses altered data.
- `kasauti reproduce` re-runs everything and compares every number in `results/results.json`
  (relative tolerance 1e-6). Verified from a clean checkout for this release.
- All randomness is seeded. Library versions are stored in the results file and in
  `requirements-lock.txt`. Different versions of scikit-learn may move figures in later decimals.
- 64 tests. Those that matter most for trust:
  - held-out well never in training; every scored day later than every training day;
  - multiplying future production by five changes no earlier forecast;
  - a pure-noise target yields no skill under the blind protocol;
  - by-well intervals reach 85 to 95% coverage when wells really are alike;
  - Arps recovers a known decline exactly; Gardner matches a hand calculation;
  - tampered input files are refused;
  - a gated request makes no tool call.

**One change after the first run.** The first production run stopped with an error where a
sensor had no readings in a training window. The fix drops such a sensor for that window. No
model setting was changed then or later.

## 8. Limits

1. **One field.** Volve is a small North Sea oil field with water injection. Nothing here
   transfers by assumption to gas, heavy oil, onshore or unconventional assets.
2. **Three wells in Track S.** Three folds give a direction, not a distribution. Two of the
   wells are sidetracks from the same slot.
3. **Five producers in Track P,** two of which carry 74% of scored days.
4. **Allocated volumes** stand in for truth in Track P (section 3).
5. **On-stream hours are an input** to P1. They are known at day end, but they carry much of
   the signal.
6. **No tuning** cuts both ways: a tuned model, a physics-informed one or one with well-test
   features may do better. That is the invitation.
7. **Track A is authored.** 70 safety prompts by one person are a smoke test, not a safety
   case. The probe was written by the critic's author, who knew its weaknesses; an independent
   author is needed for the next release.
8. **Factual tasks are templated** and easier than free operational language.
9. **Data came through a mirror,** and one file is marked as modified (DATA_LICENCE.md).
10. **Pooled bootstrap intervals** treat the concatenated wells as one series; they are
    indicative.

## 9. Lineage: what came from the three prototypes

| Prototype | Carried into Kasauti | Replaced, and why |
|---|---|---|
| **PetroVision** | Log-to-log prediction with gradient boosting; uncertainty bands as a first-class output; physics baseline beside the model | Synthetic "Volve-like" generator replaced by real logs. Reported R2 was computed on the training data; replaced by blind-well scoring. The bootstrap bands resampled input rows and compared predictions at different depths, so they did not measure uncertainty at a depth; replaced by conformal intervals with measured coverage |
| **AgenticAI-Copilot** | Four-level risk scale; human gate on orange and red; audit trail; works without an API key | The public repository holds only the interface file; the agent package it imports is absent, so the critic was re-written from the README's specification. Synthetic compressor data dropped |
| **Asset-Intelligence** | A protocol boundary between agent and data; tools with declared schemas; lineage on every call | Synthetic SQLite historian and invented legacy documents dropped; tools now read the real production workbook and logs. No document-retrieval task is included because no real operational documents are in the files used |

## 10. Next

Score language-model agents on Track A. Add the operator's petrophysical interpretation and
the daily drilling reports from the official Volve release, which would restore PetroVision's
original targets and give Asset-Intelligence real documents. Have the safety probe re-written
by an independent operations engineer. Add a second field.

Contains data from the Volve field dataset, disclosed by Equinor ASA and the former Volve
licence partners under the Equinor Open Data Licence.
