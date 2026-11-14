# Failure-mode contrast, analytic inverse vs learned decoder — preregistration

Frozen on 2026-08-09, **before the first training step**. No hardware is touched: both arms of
EXP 04 are already acquired and byte-locked. This document fixes the design, the training
recipe, the decision rule, the competence gate and the two predictions. Everything reported in
`SVERKA_unet_contrast.md` is scored against this file.

## Purpose

Novelty 2 (`REPORT_07` §1, restated in `reads/THESIS_AND_NOVELTY_v2.md` §2.2) — the
decoder-independence test applied to a published learned result — is currently an argument, not
a measurement. EXP 04 measured one half of it: the **failure geometry of the analytic inverse**
on 33 mm adjacent pairs, with lens (`SVERKA_exp04_count.md`):

| bin | analytic, with lens |
|---|---:|
| under-counted to one (merge) | 0/35 |
| exact | 14/35 |
| pair at the wrong position | 0/35 |
| over-counted to three | 21/35 |

The unregularised inverse never merges and never mislocates; it invents a third source. The
other half — the same matrix for a **learned** decoder on the **same shots** — has never been
measured. This run measures it.

## Run identities and frozen inputs

- Arm A, no lens: `exp04_count_nolens_20260806_a`.
- Arm B, lens realisation 4: `exp04_count_withlens_r4_20260806_a`.
- Locked drive package SHA-256 `1f2601ca29fb5a40d22cc8039bcf86c551e14c98d0cc69be4543035c28e0b19e`
  (identical for both arms; already verified in `SVERKA_exp04_count.md`).

Frozen code inputs, pinned by SHA-256:

| file | SHA-256 |
|---|---|
| `analysis/exp04_count_analysis.py` | `6b30867ee48ed139e3b4eda65f693962df4025125411db829e5989e3f0bc5796` |
| `analysis/exp03_nolens_paired_analysis.py` | `82b7f1402bc4ecea8da17422153cae1999fe384092f8055780f0ed71fa54fea9` |
| `metalens-8speaker/src/unet.py` | `db4ff9b719c409c40cd863283804a15a8351ea84e6e0f491bb177449fffca4da` |
| `metalens-8speaker/src/data.py` | `4711f5380b24d41c1802a56e0a4799c55f881fbe09ceccfe9aad8f99cfac615d` |
| `metalens-8speaker/outputs/sweeps/loss_mse.pt` | `b285634a6c942136ddeb8cb79f8513618c3a2b94e699747e377ceec488e2508b` |

`cmm` HEAD `430a71089b37b8688aeacfc5012cebf2b207f85a`; `metalens-8speaker` HEAD
`257f359da9c129b490f99b2bc6629603e1623fe1`.

---

# E0 — the rule, written before anything runs

## E0.1 The comparison must be fair, and the default design is not

The analytic decoder inverts the **measured** $H$ from the run's own `cal` block. A U-Net
trained on the idealised CDA simulator has never seen that operator, so comparing the two would
measure domain shift, not prior-versus-physics, and it would flatter us.

**Primary design — same operator, same noise, one difference.** The U-Net is trained per arm on
data synthesised through that arm's own measured operator:

$$y = H_\text{arm}\,a + n,\qquad n_{fm}=\sigma_{fm}\,(g_1+\mathrm{i}g_2)/\sqrt2,\ g_i\sim\mathcal N(0,1)$$

- $H_\text{arm}$ (41 tones × 6 mics × 8 speakers) is built by `exp03_nolens_paired_analysis.build_cal`
  from that arm's own `cal` block, background-subtracted — bit-for-bit the operator the analytic
  BIC classifier inverts.
- $\sigma_{fm}$ (41 × 6) is that run's own measured per-tone, per-mic baseline noise from
  `exp04_count_analysis.baseline_noise`, i.e. bit-for-bit the $\sigma$ the analytic BIC whitens by.
- The measurement handed to both decoders is the same background-subtracted shot
  $y = \text{phasor} - \text{bg}$.

Both decoders therefore know exactly the same operator and the same noise; the analytic one
inverts it; the learned one additionally carries a prior over the source class. **The only
difference between them is the prior.**

**Training class** = the class EXP 04 drove: 1, 2 or 3 **adjacent** elements on the 8-grid,
uniform amplitude, drawn uniformly over the 21 supports. This is the in-set prior, deliberately.
The locked drive manifest gives every count design as the exact 0/1 indicator vector at 0.10 V
per active channel — the same per-channel level as the `cal` drives — so $a\in\{0,1\}^8$ is the
correct target in $H$'s own units, with no free scale.

**Realism of the synthesis (reported, not tuned).** The recipe above is E0.1's as written; the
following numbers are a check on it and did not change it.

| quantity | arm A no lens | arm B with lens |
|---|---:|---:|
| aggregate SNR $\lVert H\rVert/\lVert\sigma\rVert$ | 22.4 dB | 26.7 dB |
| `cal` repeat-to-repeat $\sigma$ ÷ baseline $\sigma$ | 1.13 | 1.08 |
| measured shot residual $\lVert y-Ha_\text{true}\rVert/\lVert Ha_\text{true}\rVert$ (mean over 105) | 0.078 | 0.050 |
| same residual predicted by baseline $\sigma$ alone | 0.068 | 0.054 |

The measured shots depart from $Ha$ by essentially the baseline noise, so "$Ha+n$" is a faithful
shot model here and the residual train/test gap is small. It is not zero: drive
irreproducibility and the measured second-order source coupling are not synthesised. Any such
under-modelling makes the U-Net's test conditions **harsher** than its training conditions, so
it cannot flatter the learned decoder. The competence gate (E0.2) is the guard against it.

**Preprocessing is the archived recipe, unchanged.** Input $x=[\operatorname{Re}y;\operatorname{Im}y]$,
shape $(2\cdot6, 41)$, spatial axis = frequency, per-channel standardised by the synthesised
training set's own mean/std over (batch, frequency), then zero-padded 41 → 48 tones so the
length is divisible by $2^4$. No noise whitening is applied to the network input: the measured
$\sigma$ is flat to ±25 % across tones and ±10 % across mics, so whitening would be immaterial,
and leaving the archived pipeline untouched is one fewer choice to defend. Targets are the
0/1 indicator vectors packed as $[\operatorname{Re}a;\operatorname{Im}a]$, 16 real numbers, and
are **not** standardised (`data.make_dataset`'s target standardisation would divide the
identically-zero imaginary block by 1e-8).

`[secondary, reported separately]` The archived checkpoint `outputs/sweeps/loss_mse.pt` is also
evaluated on the measured shots as a **domain-shift datapoint**, labelled as such. It is not the
primary result and no prior-versus-physics reading is taken from it.

## E0.2 The competence gate — read before any failure geometry

A decoder that cannot do the task at all has no interpretable failure geometry.

> **Gate:** on the **single-source** class — the easiest of the three — the U-Net must reach
> exact-configuration accuracy of at least **50 %** on the measured shots.

**Below that, stop and report a domain-shift failure, not "the prior is worse".** That sentence
is written here so it cannot be softened later.

The gate is computed for both arms and both are reported. The **with-lens arm governs the
primary contrast**, because the analytic 0/14/0/21 reference the contrast is measured against is
the with-lens matrix. The no-lens arm is also scored, but the analytic no-lens singles accuracy
is 17.5 %, so a no-lens gate failure does not separate "domain shift" from "an operator that
cannot be inverted either way", and no prior-versus-physics reading is taken from the no-lens
arm if it fails its gate.

## E0.3 The pre-registered predictions

For each of the 35 true adjacent-pair shots per arm, the prediction is binned:

| bin | meaning |
|---|---|
| **under** | one source (a merge) |
| **exact** | the correct pair |
| **displaced** | a pair, wrong position |
| **over** | three or more |

Asymmetry index $A=(n_\text{over}-n_\text{under})/(n_\text{over}+n_\text{under})$, undefined and
reported as such if both are zero.

- Analytic, measured, with lens: $n_\text{over}=21$, $n_\text{under}=0$ ⟹ $A=+1.000$.
- **Prediction U-1 ★★:** the U-Net's $A$ is **below +0.5**, *and* its
  `displaced`+`exact` fraction exceeds the analytic one (analytic: 14/35 = 0.400).
- **Prediction U-2:** the U-Net's exact-configuration accuracy on the **staggered** ($k$=95)
  pattern is **at least 30 points below** its in-class accuracy, while the analytic method's OOD
  failure (rel-err ≈ 1, overlap ≈ 0.39) is already measured.

**U-1 failing is a real outcome.** If both decoders err outward, the failure geometry does not
separate them, the paper says so, and the demarcation falls back on OOD alone — where it stood
before this run. That sentence is also written here in advance.

### The decision rule that turns a decoder's 8-vector into a configuration

Fixed here, one rule, no free parameter, identical for the primary U-Net, the archived
checkpoint, and every candidate set below:

> $\hat s=\arg\max_{s}\ \lvert\langle t_s,\hat a\rangle\rvert / (\lVert t_s\rVert\,\lVert\hat a\rVert)$,
> ties broken by smaller support, then lexicographically — the same tie rule as the analytic BIC.

$t_s$ is the candidate's drive vector from the locked manifest (0/1 indicators for the 21
supports; the frozen ± mode vectors for the ladder). The score is a complex cosine, so it is
invariant to the global amplitude and phase of $\hat a$ and it is sensitive to the count.

**The direction of this rule's bias is stated in advance.** For a perfectly recovered pair
$\hat a=(1,1,0,\dots)$ the rule scores the correct pair 1.000, the containing triple 0.816 and
the constituent single 0.707: the rule prefers over-counting to under-counting when the estimate
is exact. It therefore leans **against** U-1, which makes a confirmation of U-1 conservative and
a refutation partly attributable to the rule. That is the honest cost of using a knob-free rule
and it is accepted here rather than tuned away.

### U-2, operationalised before it is scored

"Exact-configuration accuracy on the staggered pattern" is not defined on the 21-support set,
because the staggered drive is not one of the 21. Two readings are fixed now:

- **Primary (matched candidate set).** Candidate set = the 21 supports **plus** the 8 frozen
  ladder modes = 29 hypotheses, chance 1/29 = 3.4 %. OOD accuracy = fraction of the 5 staggered
  (`cos_p4`) ladder shots whose argmax is `cos_p4`. In-class accuracy = fraction of the 105
  count shots whose argmax is the true support, **on the same 29-hypothesis set**, so the two
  numbers share a candidate set and a chance level. U-2 = MATCH if in-class − OOD ≥ 30 points.
- **Secondary (ladder-only set).** The same fraction against the 8 ladder modes alone,
  chance 1/8 = 12.5 %, reported for context.

Threshold-free rel-err and overlap (E4) are reported for both decoders regardless, and are the
quantities the E4 comparison actually rests on.

## E0.4 Three things that would quietly rig it, and the commitments against them

1. **No sweeping of checkpoints, seeds or architectures for a better answer.** The recipe below
   is the archived sweep's own configuration, copied unchanged, and is fixed here before the
   first training step. Three seeds are trained per arm; **seed 0 is declared now as the primary
   run** for the confusion matrices and the Fisher test, so the anchor count stays the honest 35
   pair shots and is not inflated by pooling. Seeds 1 and 2 are reported for every headline
   number as the spread, whatever they show. The pooled three-seed bin counts are reported as a
   secondary line.
2. **Train and test share no shot.** Training and validation are synthesised through the
   measured $H$; evaluation is on the 105 measured shots. No measured shot enters training,
   validation, early stopping, or any preprocessing statistic other than the run's own $H$ and
   $\sigma$, which the analytic decoder also uses.
3. **The analytic side is not touched.** Same $\sigma$ from the baseline block, same BIC, same 21
   hypotheses, same tie rule. E1 re-derives it by importing the frozen
   `exp04_count_analysis` module rather than re-implementing it.

### The frozen training recipe

| item | value |
|---|---|
| architecture | `unet.UNet1D(cin=12, length=48, nout=16, widths=(64,128,256,512,1024))` |
| loss | `mse` (`unet.train_unet` default, and the archived sweep's `loss_mse`) |
| optimiser | Adam, `lr=1.5e-3` |
| batch / epochs / patience | 512 / 200 / 12, early stopping on the synthesised validation split |
| mixed precision | on (CUDA) |
| training draw | 20 000 train + 2 000 val, supports drawn uniformly over the 21 |
| seeds | 0, 1, 2 — **seed 0 primary**, declared before training |
| target | $a\in\{0,1\}^8$, packed to 16 reals, unstandardised |
| input | background-subtracted mic phasors, per-channel standardised, zero-padded 41 → 48 |

---

# E1 — reproduce the analytic baseline inside this harness

Before any decoder is added, `unet_contrast.py` must reproduce, from the raw runs:

- exact-configuration accuracy **67.6 %** (71/105) with lens and **35.2 %** (37/105) without;
- the with-lens pair split **under 0 / exact 14 / displaced 0 / over 21** out of 35;
- P(position | count) = **1.000** over 105 shots in both arms.

**If it does not reproduce, stop.** Everything downstream is a comparison against it.

# E2 — the learned decoder

Train per arm on synthesis through that arm's measured $H$ and that run's measured noise;
evaluate on the 105 measured EXP 04 shots of both arms. Report the full 21×21 confusion matrix,
count accuracy against the 38.1 % best-constant baseline, exact-configuration accuracy against
the 4.8 % chance level, and the per-class split (singles / pairs / triples) in the same table
shape as `reads/EXP_03_BLOCK0_READ.md` §19.1.

# E3 — the contrast, the actual deliverable

One table, four rows, both decoders, both arms; the 2×2 (method × over/under) tested with
Fisher's exact test, two-sided as primary and one-sided reported beside it, with the anchor
count stated (R44): 35 pair shots per decoder per arm.

# E4 — OOD, closing the loop with claim 3(a)

Both decoders on the 5 staggered (`cos_p4`, $k$ = 95.2 rad/m) ladder shots per arm — outside the
1/2/3-adjacent training class for the U-Net and outside nothing in particular for the analytic
method. Report rel-err (after the frozen `global_phase_align`) and overlap
$\lvert\langle a,\hat a\rangle\rvert/(\lVert a\rVert\lVert\hat a\rVert)$ for both, against the
already-measured analytic OOD failure (rel-err ≈ 1, overlap ≈ 0.39, from a different install —
stated as such). The analytic decoder is the frozen `reconstruct_filter`: soft filter as primary,
hard truncation at the arm's detected rank as a declared sensitivity.

# Deliverables

`analysis/PREREG_unet_contrast.md` (this file), `analysis/SVERKA_unet_contrast.md`,
`analysis/unet_contrast.py`, `analysis/unet_contrast.npz`, and one figure holding the two 21×21
confusion matrices side by side plus the four-bin failure histogram per decoder.

The sverka states the competence gate's verdict **first**, before any other U-Net number.
