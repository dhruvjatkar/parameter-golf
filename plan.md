# Parameter Golf: Research Directions

## Context

**Target to beat**: 1.0781 BPB (PR #672, TTT_EPOCHS=30 Cosine TTT, unmerged)
**Merged SOTA**: 1.1194 BPB (LeakyReLU² + Legal TTT + Parallel Muon, built on PR #414 stack)
**Our baseline**: 1.2304 BPB (8×H200, 10 min, stock train_gpt.py)
**Dream target**: 0.6 BPB — nearly halving current SOTA. Incremental tweaks won't get us there. We need paradigm shifts.

PR #672 achieves 1.0781 with TTT_EPOCHS=30 and a cosine TTT schedule. Since TTT is now maxed out (30 epochs using 590s of the 600s eval budget), future improvements must be ORTHOGONAL to TTT.

The SOTA already uses: 11 layers, 3x MLP, Parallel Muon, int6 GPTQ-lite + lzma, EMA+SWA, XSA4, Partial RoPE, LN Scale, BigramHash, SmearGate, VE128, LeakyReLU(0.5)², sliding window eval (stride=64), legal TTT, TTT_EPOCHS=30.

**Mindset**: At 0.6 BPB we need ~2x the effective model quality within the same 16MB/10min constraints. Think: architectural breakthroughs, novel training paradigms, distillation from larger models, radically different representations. Don't be afraid to try wild ideas.

---

## Top 8 Directions (Ranked by Expected Impact × Feasibility)

**Key constraint**: PR #672 maxes out TTT at 30 epochs (590s of 600s eval budget). All directions below must be ORTHOGONAL to TTT to stack on top of 1.0781.

### 1. XSA on all 11 layers — CONFIRMED POSITIVE, MUST INCLUDE
**What**: Set `XSA_LAST_N=11` (was 4). Apply cross-sequence attention to all layers instead of just the last 4.

**Why**: CONFIRMED delta of -0.0255 BPB on 1xH200 clean A/B test. One-line config change, zero risk.

**Evidence**: PRs #609, #634, #639 all use XSA-all. Our own clean H200 test confirmed it (Round 3.4).

**Implementation**: `XSA_LAST_N=11` in launcher. Already done.

**Expected gain**: -0.025 BPB (CONFIRMED). Must be in every future experiment.

---

### 2. Full Hessian GPTQ — HIGH IMPACT, ORTHOGONAL TO TTT
**What**: Replace GPTQ-lite with full Hessian-based GPTQ using Cholesky error compensation and calibration data.

**Why**: Halves the quantization gap from ~0.008 to ~0.004 BPB. Every frontier submission (PRs #609, #634, #639) uses this. Orthogonal to TTT.

**Evidence**: All submissions at 1.115x or better use Full GPTQ. Currently awaiting cluster results.

**Implementation**: Cholesky-based error compensation with training data calibration during export. Must be done at training time (not eval time) for legality.

**Expected gain**: -0.004 BPB (from halved quant gap)

---

### 3. SwiGLU MLP — PARAMETER-NEUTRAL, ORTHOGONAL TO TTT
**What**: Replace LeakyReLU(0.5)^2 activation with SwiGLU gated MLP. Parameter-neutral by adjusting hidden dimension.

**Why**: SwiGLU is proven better than other activations at this scale. Upstream PR #676 proposes this. Orthogonal to both TTT and quantization.

**Evidence**: PR #676 (opened 2026-03-25). Clean architectural change. Awaiting cluster results.

**Implementation**: Gate + SiLU in MLP, halve hidden dim to stay parameter-neutral. Env-gated via `USE_SWIGLU`.

**Expected gain**: -0.002 to -0.005 BPB (speculative, awaiting screen)

---

### 4. Muon-VS (Variance-Adaptive Muon) — LOW EFFORT, ORTHOGONAL TO TTT
**What**: Apply variance-based scaling to the momentum buffer before Newton-Schulz orthogonalization.

**Why**: Muon-VS shows 1.36x reduction in iterations to target loss. Faster convergence = lower final loss in fixed 10-minute budget. Orthogonal to TTT.

**Evidence**: Arxiv 2601.14603. Currently awaiting cluster results.

**Implementation**: Track running variance of gradients, scale momentum by inverse noise-to-signal ratio before NS5. ~20 lines of code.

**Expected gain**: -0.003 to -0.008 BPB

---

### 5. More Aggressive Quantization (Int4/Int5) — HIGH IMPACT, ORTHOGONAL TO TTT
**What**: Push quantization below int6 to free parameter budget for wider/deeper model.

**Why**: Int6 enabled 3x MLP. Int4 could enable 4x MLP or 13+ layers. Freed bytes = pure model capacity. Fully orthogonal to TTT.

**Evidence**: GPTQ-lite showed free -0.0006 BPB. PR #606 explores int5 (31 levels, 33.6M params under 16MB).

**Implementation**: Start with mixed int4/int6 (MLP at int4, attention at int6). Careful QAT schedule needed.

**Expected gain**: -0.005 to -0.015 BPB (from freed parameter budget)

---

### 6. Cross-Layer Weight Sharing (MASA) — HIGH IMPACT, ORTHOGONAL TO TTT
**What**: Share weight matrix components across layers. MASA decomposes Q/K/V/O into shared dictionary atoms.

**Why**: 66.7% parameter reduction in attention while maintaining performance. Directly frees parameter budget. Orthogonal to TTT.

**Evidence**: AAAI 2026 paper shows MASA outperforms GQA. ICLR 2025 Basis Sharing paper confirms cross-layer SVD sharing works.

**Implementation**: Decompose attention weight matrices into shared basis + per-layer coefficients.

**Expected gain**: -0.005 to -0.01 BPB (from effective model capacity increase)

---

### 7. Depth Recurrence (Looped Transformer) — WARNING: KNOWN INT6 RISK
**What**: Reuse weight blocks across multiple forward passes. 4-6 unique layers looped 2-3 times.

**Why**: Trades compute (excess) for parameters (scarce). But quantization error amplifies ~900x over cycles at int6.

**Evidence**: Recent papers show 25-55% parameter reduction. BUT competition experiment showed 1 block x 9 passes only reached 1.1454 BPB due to int6 error amplification.

**Implementation**: Currently awaiting clean test results from `DepthRecurrence6L2X_Clean`.

**Expected gain**: -0.01 to -0.03 BPB IF quant error is managed; likely NEGATIVE otherwise

---

### 8. AdEMAMix (Dual-EMA Momentum) — MEDIUM EFFORT, ORTHOGONAL TO TTT
**What**: Replace Adam's single EMA with a mixture of fast and slow EMAs for better data efficiency.

**Why**: 95% data efficiency improvement at scale. In our time-constrained setting, better data efficiency = lower loss per step. Orthogonal to TTT.

**Evidence**: ICLR 2025 paper. Clean A100 screen staged (`AdEMAMix_A100_Clean`).

**Implementation**: Apply AdEMAMix to Adam optimizer groups (embeddings, scalars). Keep Muon for matrix params.

**Expected gain**: -0.002 to -0.005 BPB

---

### Deprioritized (already exploited by PR #672 or low signal)
- **TTT epoch tuning**: 30 epochs already maxed (PR #672)
- **TTT LR scheduling**: Cosine already in PR #672
- **TTT per-layer LR**: May have marginal gains on top of 30 epochs, low priority
- **Curriculum learning**: Speculative, no signal from upstream
- **Larger tokenizer**: High risk, no evidence at 16MB scale

---

## Implementation Queue (updated 2026-03-25)

**XSA-all is confirmed. Stack orthogonal wins on top of the 1.0781 baseline.**

### Now running (collect results before launching new work)
1. **Full GPTQ** (`FullGPTQ_H200_Clean`) — jobs 5408981/5408982 on gpu-short
2. **SwiGLU MLP** (`SwiGLU_A100_Clean`) — jobs 5409004/5409005 on gpu-short
3. **Muon-VS** (`MuonVS_A100_Clean`) — jobs 5408902/5408903 on gpu
4. **Depth Recurrence 6Lx2** (`DepthRecurrence6L2X_Clean`) — jobs 5408935/5408936 on gpu
5. **TTT 30 epochs** (`TTT30_Clean`) — jobs 5408918/5408919 on gpu

### Next up (after collecting results)
- Stack confirmed positives: XSA-all + Full GPTQ + best optimizer/architecture winner
- **Selective magnitude pruning** — post-GPTQ, zero least-impactful quantized values. Frees capacity.
- **Value Residual Learning (VRL)** — arXiv:2410.17897. May conflict with VE128 (PR #609 found this).
- **Int5 quantization** — 31 levels, 33.6M params under 16MB (PR #606). High risk, high reward.

### Competition Intelligence (updated 2026-03-25)
- **PR #672 (OPEN): 1.0781 BPB** — THIS IS THE NEW TARGET. TTT_EPOCHS=30 Cosine TTT. 3-seed (std=0.0041). Eval: 494s TTT + 96s sliding = 590s total. TTT is maxed out here.
- **PR #609 (OPEN): 1.1154 BPB** — XSA all 11 layers + Full GPTQ. 3-seed.
- **PR #638 (OPEN): 1.1164 BPB** — XSA-all + Value Residual + Gated Attention. 1-seed.
- **PR #676 (OPEN): SwiGLU MLP** — parameter-neutral gated MLP replacing LeakyReLU(0.5)^2. No BPB posted yet.
- Merged SOTA: **1.1194 BPB** (PR #549)
- PR #659 closed (illegal hindsight selection)
- PR #668: 1.0920 BPB, unlimited compute (non-record)

### Legality notes
- PR #672's code performs AdamW TTT over validation tokens before final sliding eval. Compatibility with competition rule 5 is unclear.
- AdamW for TTT is catastrophic on GPTQ models (+0.077 BPB). Only SGD TTT is safe.
- Full GPTQ calibration must be done at training time (not eval time) for legality.

---

## Results & Learnings

_Agents: update this section after every experiment round._

### Historical Summary (March 24-25, 2026)

**69 attempt folders created, 1 usable result.** The primary cause was infrastructure chaos:
- Multiple competing agents cancelling each other's SLURM jobs
- flash_attn_interface missing from cluster environment
- SSH hangs blocking rsync/sbatch
- SLURM queue saturation (8-job limit)
- Crontab-based backlog system non-functional (PAM blocks crontab)

**Key findings:**
- XSA on all 11 layers: CONFIRMED POSITIVE (delta -0.0255 on 1xH200)
- Depth recurrence: KNOWN RISK at int6 quantization (error amplifies ~900x over cycles)
- Pre-EMA validation snapshots can be misleading vs final int6 sliding-window BPB
- StackedWins (combining all techniques): catastrophic when QAT fires at step 1 on A100

**Operational lesson:** Single-agent protocol is mandatory. One agent, one cluster, sequential experiments.

**Key negative results from upstream (avoid these):**
- **TrigramHash**: hurts compression (+0.0049 BPB from compression penalty)
- **BigramHash(8192)**: great BPB but 0.52MB over artifact budget
- **AdamW for TTT**: catastrophic on GPTQ models (+0.077 BPB). Only SGD TTT is safe.
- **Catalytic Residuals**: redundant with existing resid_mix
- **Hadamard rotation + GPTQ**: hurts lzma compressibility
- **Multi-agent cluster access**: Produces zero results. Never again.
- **StackedWins with QAT at step 1**: Catastrophic on A100 (warmdown threshold always active)

### Round 1 (2026-03-25) — Baseline Reset

**New target**: 1.0781 BPB (PR #672)
**Strategy**: Since TTT is maxed at 30 epochs, stack orthogonal improvements:
- XSA-all (confirmed -0.025 delta)
- Full GPTQ (expected -0.004 BPB from halved quant gap)
- SwiGLU MLP (parameter-neutral, awaiting screen)
- Muon-VS optimizer (awaiting screen)

**5 experiments currently submitted** -- collect results before launching new work:
- `MuonVS_A100_Clean` (jobs 5408902/5408903)
- `TTT30_Clean` (jobs 5408918/5408919)
- `DepthRecurrence6L2X_Clean` (jobs 5408935/5408936)
- `FullGPTQ_H200_Clean` (jobs 5408981/5408982)
- `SwiGLU_A100_Clean` (jobs 5409004/5409005)
