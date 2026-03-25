# PLANS.md — Operational Queue

Last updated: 2026-03-25

## Active Agent
<!-- Set your agent ID and timestamp when starting a session. Clear when done. -->
(none)

## Baseline

- **Target to beat**: 1.0781 BPB (PR #672, TTT_EPOCHS=30 Cosine TTT, 3-seed mean std=0.0041)
- **Merged SOTA**: 1.1194 BPB (March 23, LeakyReLU² + Legal TTT + Parallel Muon)
- **Base script**: `records/track_10min_16mb/PR672_CosineTTT30_1.0781/train_gpt.py`
- **Stock baseline**: 1.2304 BPB (8xH200, 10 min)

## Experiment Status

### KEEP (validated results)
| Name | BPB | Baseline BPB | Delta | GPU | Notes |
|------|-----|-------------|-------|-----|-------|
| XSAAll_H200_Clean | 2.4259 | 2.4514 | -0.0255 | 1xH200 | XSA on all 11 layers confirmed positive. Only 860 steps (1xH200), not leaderboard-scale. |

### SUBMITTED (awaiting results — DO NOT resubmit or cancel)
| Name | Jobs | GPU | Technique |
|------|------|-----|-----------|
| MuonVS_A100_Clean | 5408902/5408903 | 1xA100 | Variance-scaled Muon optimizer |
| TTT30_Clean | 5408918/5408919 | 1xH200 | Legal TTT with 30 epochs |
| DepthRecurrence6L2X_Clean | 5408935/5408936 | 1xA100 | 6-layer U-Net with 2x repetition |
| FullGPTQ_H200_Clean | 5408981/5408982 | 1xH200 | Full Hessian GPTQ quantization |
| SwiGLU_A100_Clean | 5409004/5409005 | 1xA100 | SwiGLU gated MLPs (parameter-neutral) |

### PENDING (clean scripts, re-evaluate against PR672 baseline before running)
- TTTPerLayerLR_Clean — Per-layer TTT LR groups (3x output, 0.5x input)
- TTTGradClip08_Clean — TTT grad clip 0.8 vs 1.0
- CurriculumSeqLen_A100_Clean — Curriculum seq len 512→2048
- TTT5ep_Clean — TTT 5 epochs + skip non-TTT eval
- AdEMAMix_A100_Clean — AdEMAMix on Adam-side groups
- XSAAll_Warmdown4000_A100 — XSA-all + warmdown 4000 iters
- XSA11_VR_GA_Clean — XSA-all + value residual + gated attention

### ARCHIVED
- ~50 stale attempt folders moved to `attempts/archive/`
- ~55 stale backlog scripts moved to `job_backlog/archive/`
- See `attempts/results.tsv` for the full historical record

## Next Directions

Since PR #672 already maxes TTT at 30 epochs (590s eval), the highest-leverage improvements are **orthogonal** to TTT:

1. **XSA-all-11** — Confirmed positive (delta -0.0255 on 1xH200). Stack into every new experiment.
2. **Full Hessian GPTQ** — Halves quantization gap. Awaiting results from FullGPTQ_H200_Clean.
3. **SwiGLU MLP** — Parameter-neutral gated MLP. Awaiting results from SwiGLU_A100_Clean.
4. **Muon-VS** — Variance-adaptive optimizer. Awaiting results from MuonVS_A100_Clean.
5. **Architecture search** — Deeper/wider within 16MB constraint. Depth recurrence test running.

See `plan.md` for detailed research priorities and technique analysis.

## Session Handoff Protocol

Before ending any session:
1. Collect results from submitted jobs (rsync logs from Explorer)
2. Update `attempts/results.tsv` with any new results
3. Update this file: move completed experiments to KEEP/ARCHIVED, add new PENDING items
4. Clear the Active Agent field
