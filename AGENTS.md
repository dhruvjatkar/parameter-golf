# AGENTS.md

## Codex Agent Instructions

This file is the Codex-specific operating manual for this repo.

- `PLANS.md` is the active execution queue and status board.
- `plan.md` is the long-form research notebook and experiment history.
- `attempts/results.tsv` is the append-only run log.

Start every session by reading all three.

## Mission

Beat the current 10-minute / 16MB SOTA BPB and keep iterating until you do.

- Target to beat: **1.0781 BPB** (PR #672, TTT_EPOCHS=30 Cosine TTT)
- Current merged SOTA on `2026-03-24`: **1.1194 BPB**
- Stock baseline: **1.2304 BPB**
- Stretch target: **0.6 BPB**

Do not stop at one clean experiment. Keep stacking wins, validating them, and updating the plan.

## Critical Repo Rules

- Work only in the fork `dhruvjatkar/parameter-golf`.
- Never open PRs, push, or otherwise target `openai/parameter-golf`.
- Never modify repo-root `train_gpt.py` or `train_gpt_mlx.py`.
- Always copy the best current training script into a new attempt folder before editing.
- Always run a concurrent baseline alongside every experiment on the same GPU type for the same duration using the unmodified SOTA script.
- Always write `hypothesis.md` before launching a run.
- Always append results to `attempts/results.tsv`, including failures.
- Never delete failed attempts. Mark them `DISCARDED` and move on.
- On Explorer, never cancel jobs you did not start. Only manage or cancel job IDs submitted by the current Codex session.

## Single-Agent Protocol

- Only ONE agent may operate against the Explorer cluster at a time.
- Before starting work, check PLANS.md for an active agent session marker.
- If another agent is active, coordinate with the user before proceeding.
- At session start, write your agent ID and start time to PLANS.md.
- At session end, clear the active agent marker and update handoff notes.

## Competition Rules

Check these before implementation, after implementation, and before submission.

1. Artifact must be `<= 16,000,000` bytes total.
2. Training must finish within 10 minutes on `8xH100 SXM`.
3. Evaluation must finish within 10 minutes on `8xH100 SXM`.
4. No external downloads or network calls during evaluation.
5. No training on validation data before evaluating it. Legal TTT only on tokens already scored.
6. Do not smuggle extra compute through custom libraries.
7. Do not brute-force seeds or otherwise game variance.
8. Record claims must beat SOTA by at least `0.005` nats with statistical significance, typically 3 seeds.
9. Tokenizer changes require proof that `val_bpb` is still computed correctly.
10. Final competition code must live in a single `train_gpt.py` that runs from the records folder.

## Codex Workflow

1. Read `PLANS.md`, `plan.md`, and `attempts/results.tsv`.
2. Pick the highest-priority untried direction that is not blocked or illegal.
3. Create `attempts/YYYY-MM-DD_ShortName/`.
4. Copy the current best legal SOTA script into that folder.
5. Implement the change in the copied file only.
6. Make every new feature toggleable through an env var for clean A/B tests.
7. Write `hypothesis.md` before any launch script is submitted.
8. Run an adversarial code review before submission.
9. Launch experiment and baseline concurrently.
10. Collect logs, record BPB and delta, and update `attempts/results.tsv`.
11. Update both `plan.md` and `PLANS.md` with results, blockers, and next steps.

Submission uses the job backlog (see below). Write SLURM scripts, place them in
`job_backlog/pending/`, rsync to Explorer, then immediately start the next research direction.
Never wait for the queue to clear before beginning new work.

## Code Review Standard

Every experiment must be reviewed as if it is broken until proven otherwise.

Check for:

- Wrong shapes, wrong axes, or silent broadcasting mistakes
- Bad gradient flow, missing `.detach()`, or dead code paths
- Dtype and numerical stability issues
- Flash-attention / SDPA fallback mismatches
- Env vars that do not match the code path they are supposed to toggle
- Evaluation-time legality issues, especially TTT and GPTQ calibration
- Baselines accidentally pointing at modified attempt copies
- Mismatch between `hypothesis.md` and the actual implementation

If an experiment underperforms, review it again before discarding the idea.

## Cluster Defaults

- SSH target: `ssh explorer` as user `d.jatkar`
- Cluster repo: `/projects/Sontag_Lab_Storage/parameter-golf/`
- Environment: `/projects/Sontag_Lab_Storage/parameter-golf-env/`
- Dataset: `./data/datasets/fineweb10B_sp1024/`
- Never write caches to `~/`

Every SLURM script should include:

```bash
source /etc/profile.d/modules.sh
module load python/3.13.5
module load cuda/12.8.0
export TRITON_CACHE_DIR=/projects/Sontag_Lab_Storage/.triton_cache
export HF_HOME=/projects/Sontag_Lab_Storage/.hf_cache
export TORCH_HOME=/projects/Sontag_Lab_Storage/.torch_cache
export XDG_CACHE_HOME=/projects/Sontag_Lab_Storage/.xdg_cache
source /projects/Sontag_Lab_Storage/parameter-golf-env/bin/activate
```

## GPU Escalation Path

1. `1xA100` for quick screens
2. `1xH200` to confirm wins
3. `8xH200` for leaderboard-equivalent validation

Only move to `8xH200` after a clear 1-GPU improvement.

## Attempt Protocol

Use this layout:

```text
attempts/
  results.tsv
  YYYY-MM-DD_ShortName/
    train_gpt.py
    hypothesis.md
    run_experiment.sh
    run_baseline.sh
    experiment.log
    baseline.log
    submission.json
```

Preferred source script at the moment:

`records/track_10min_16mb/PR672_CosineTTT30_1.0781/train_gpt.py`

Record each run in `attempts/results.tsv` as:

```text
date	name	bpb	baseline_bpb	delta	gpu	status	description
```

Status values:

- `PENDING` for staged work that is not yet `sbatch`-submitted
- `SUBMITTED` for work with live Slurm job IDs that has been submitted but not finished
- `RUNNING` for jobs currently executing
- `FAILED` / `TIMEOUT` for infrastructure or runtime failures
- `KEEP` for real wins worth stacking
- `DISCARDED` for confirmed non-wins
- `VALIDATING` for promising multi-seed or 8xH200 follow-up
- `RECORD` for validated SOTA-beating runs

## Search Policy

Search online when:

- `PLANS.md` and `plan.md` are exhausted or stalling
- A referenced paper or PR needs implementation details
- You need to verify the live upstream leaderboard
- A technique is promising but underspecified locally

## Job Backlog System

**WARNING: The crontab auto-submitter does NOT work** (PAM blocks crontab on Explorer). Use manual `sbatch` for all job submissions.

`job_backlog/` is a self-service SLURM queue directory structure.

```
job_backlog/
  submit_backlog.sh   # cron script on Explorer
  pending/            # agents drop .slurm scripts here, then rsync
  submitted/          # moved here after sbatch succeeds
  failed/             # moved here if sbatch fails
  submit.log          # full submission history
```

### One-time cron setup [NON-FUNCTIONAL — PAM blocks crontab]

```bash
ssh explorer "(crontab -l 2>/dev/null; echo '*/5 * * * * /projects/Sontag_Lab_Storage/parameter-golf/job_backlog/submit_backlog.sh') | crontab -"
```

Verify: `ssh explorer "crontab -l | grep submit_backlog"`

### SLURM script naming convention

Always prefix with a timestamp so the submitter processes scripts in order:
```
YYYY-MM-DD_HH-MM-SS_<attempt_name>_experiment.slurm
YYYY-MM-DD_HH-MM-SS_<attempt_name>_baseline.slurm
```

### Submission workflow

1. Write SLURM scripts into `job_backlog/pending/`
2. Rsync to Explorer:
   ```bash
   rsync -avz job_backlog/pending/ explorer:/projects/Sontag_Lab_Storage/parameter-golf/job_backlog/pending/
   ```
3. **Immediately pivot to a completely different research direction.** Do not wait.
   - Read `attempts/results.tsv` — what has worked best so far?
   - Search online for new approaches not yet in `plan.md`
   - Pick something architecturally distinct from what is now queued
   - Implement, review, queue that too
4. Collect results later by rsyncing `job_backlog/submitted/` and attempt `.log` files

### The pivot rule

Once scripts are in the backlog, treat them as running and move on. Multiple agents can all
rsync to `pending/` independently — the submitter serialises submission safely.

### Session handoff

Before ending any session, update:
- `attempts/results.tsv` — all newly completed results
- `plan.md` — research findings, new hypotheses, priorities
- `PLANS.md` — which experiments are staged/running and what direction comes next

**Goal**: the next agent must be able to read PLANS.md and start new work within 2 minutes.

## Operational Lessons

- Manual sbatch is the only reliable submission method on Explorer.
- Use gpu-short partition for 1-GPU screens under 2 hours.
- Always use absolute paths for DATA_PATH and TOKENIZER_PATH in SLURM scripts.
- PYTHONPATH must include attempts/_compat/ for the flash_attn_interface shim.
- Only final int6 sliding-window BPB is authoritative — pre-EMA snapshots are misleading.
- Non-interactive SSH: wrap commands in `/bin/bash --noprofile --norc -lc '...'`.

## Remotes

- `origin`: `https://github.com/dhruvjatkar/parameter-golf.git`
- `upstream`: `https://github.com/openai/parameter-golf.git`

Never target `upstream` for pushes or PRs.
