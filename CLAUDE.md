# CLAUDE.md

## Critical Rules

- **NEVER open PRs on the upstream repo (openai/parameter-golf)**. Always work in the fork (dhruvjatkar/parameter-golf).
- Never modify `train_gpt.py` or `train_gpt_mlx.py` in the repo root — copy to an attempt folder first.
- **ALWAYS run a concurrent baseline** alongside every experiment (same GPU, same duration, unmodified SOTA script). Only the delta matters.
- **NEVER cancel SLURM jobs on Explorer that you did not start.** Only cancel jobs you submitted yourself. Other jobs may belong to other agents or the user.

## Single-Agent Protocol

- Only ONE agent may operate against the Explorer cluster at a time.
- Before starting work, check PLANS.md for an active agent session marker.
- If another agent is active, coordinate with the user before proceeding.
- At session start, write your agent ID and start time to PLANS.md.
- At session end, clear the active agent marker and update handoff notes.
- Competing agents produce zero results. 69 attempt folders with 1 usable result proves this.

## COMPETITION RULES — READ BEFORE WRITING ANY CODE

**You MUST check your code against these rules repeatedly. Before implementing, after implementing, and before submitting. Violations disqualify the run.**

1. **Artifact <= 16,000,000 bytes** (code bytes + compressed model bytes). Decimal 16MB, NOT 16 MiB.
2. **Training <= 10 minutes** on 8xH100 SXM. Wall-clock cap.
3. **Evaluation <= 10 minutes** on 8xH100 SXM (separate from training time).
4. **No external downloads or network calls** during evaluation. Artifact must be fully self-contained.
5. **No training on validation data** before evaluating on it. You may only TTT on val tokens you've ALREADY scored.
6. **No sneaking compute** through custom libraries. Importing FlashAttention etc. is fine, but libraries cannot add extra compute, capabilities, or massively increase effective code size.
7. **No brute-forcing seeds** or otherwise gaming variance.
8. **Beat SOTA by >= 0.005 nats** with statistical significance (p < 0.01, typically 3-seed validation).
9. **If you change the tokenizer**, you must prove val_bpb is correctly calculated. Tokenizer changes get extra scrutiny.
10. **All code must live in `train_gpt.py`**. The script must compile and run from the records folder.

**CHECK THESE RULES EVERY TIME YOU:**
- Add a new feature or technique
- Modify the evaluation loop
- Change the quantization/compression pipeline
- Add or import a new package
- Implement test-time training or any eval-time computation

## Mission

**Beat the current target BPB (1.0781).** Iterate autonomously until you do. Do not ask the user for guidance unless you are truly stuck with no remaining directions. You have full authority to:
- Pick directions from `plan.md`, or research new ones online
- Implement, submit, collect results, analyze, and try the next thing
- Search the web for papers, blog posts, and code if you run out of ideas or need implementation details
- Stack multiple winning techniques together

**Stop condition**: Keep iterating. After each round of experiments, update `plan.md` with what worked, what didn't, and new ideas — then launch the next round. Report results back to the user but do NOT open PRs. Just keep pushing toward 0.6 BPB.

**Milestones** — any of these is a major win, do NOT abandon an approach just because it hasn't reached 0.6:
- **< 1.0781** — beats PR #672 target. Validate with 3 seeds, keep going.
- **< 1.05** — significant improvement over PR #672. Validate and report immediately.
- **< 1.0** — excellent result. This is real progress.
- **< 0.9** — exceptional. This would be a landmark.
- **< 0.6** — the dream target.

## Project Overview

**Parameter Golf** is an OpenAI challenge: train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100 GPUs. Scored by BPB (bits per byte) on FineWeb validation set — lower is better.

- **Current target to beat**: 1.0781 BPB (PR #672, TTT_EPOCHS=30 Cosine TTT, 3-seed mean, std=0.0041)
- **Current merged SOTA**: 1.1194 BPB (LeakyReLU^2 + Legal TTT + Parallel Muon)
- **Baseline**: 1.2244 BPB (9L, 512d, Muon optimizer)
- **Artifact limit**: 16,000,000 bytes (code + compressed model)
- **Target**: 0.6 BPB — we need radical, not incremental improvements. Think architectural breakthroughs, novel training paradigms, and ideas from outside the usual playbook.
- **But**: Beating the target at all (< 1.0781) is excellent. Even 1.0 or 0.9 would be outstanding. Don't throw away a winning approach because it's "only" 1.05.
- PR #672 achieves 1.0781 with TTT_EPOCHS=30 cosine schedule. This is the algorithm to beat.

## Autonomous Iteration Loop

```
REPEAT:
  1. Read attempts/results.tsv — what's been tried, what worked
  2. Read plan.md — pick the highest-priority untried direction
     - If all directions exhausted or stalling: search online for new ideas
  3. Create attempt folder, copy SOTA train_gpt.py, implement changes
  4. CODE REVIEW (MANDATORY — see "Code Judge" section below):
     - Spawn a subagent to adversarially review the code BEFORE submitting
     - Fix all issues found. Do NOT skip this step.
  5. Write hypothesis.md BEFORE running
  6. Submit experiment + baseline to Explorer via manual sbatch
  7. IMMEDIATELY start the next research direction — do not wait for results
     - Pick something architecturally different from what is now queued
     - While that experiment runs you should be implementing the next one
     - Collect results by rsyncing attempt logs when convenient
  8. Log to attempts/results.tsv (ALWAYS, even failures)
  9. Analyze:
     - If BPB improved: stack this win into next attempt, try on 1xH200 to confirm
     - If BPB same/worse: BEFORE discarding, spawn a code review subagent to check
       if the implementation has bugs. Many "failed" ideas fail due to code errors,
       not bad theory. Only mark DISCARDED after confirming the code is correct.
     - If promising on 1xA100/1xH200: validate on 8xH200 (leaderboard-equivalent)
  10. Update plan.md:
     - Add results to the "Results & Learnings" section
     - Prune dead ends, promote promising directions
     - Add new hypotheses inspired by what you learned
  11. Launch next round of experiments informed by updated plan
UNTIL 0.6 BPB is reached
```

## Job Backlog System

**WARNING: The crontab-based auto-submitter does NOT work** (PAM blocks crontab on Explorer for d.jatkar). Use manual `sbatch` for all submissions.

`job_backlog/` contains a directory structure for organizing SLURM scripts.

```
job_backlog/
  submit_backlog.sh   # NON-FUNCTIONAL — crontab blocked by PAM
  pending/            # agents drop .slurm scripts here
  submitted/          # moved here after sbatch succeeds
  failed/             # moved here if sbatch fails
  submit.log          # full submission history
```

### One-time cron setup (NON-FUNCTIONAL)

**This does not work on Explorer.** PAM blocks crontab for d.jatkar. Use manual `sbatch` instead.

```bash
# DOES NOT WORK — kept for reference only
ssh explorer "(crontab -l 2>/dev/null; echo '*/5 * * * * /projects/Sontag_Lab_Storage/parameter-golf/job_backlog/submit_backlog.sh') | crontab -"
```

### How agents use the backlog

**Name every SLURM script with a timestamp prefix** so they are processed in order:
```
YYYY-MM-DD_HH-MM-SS_<attempt_name>_experiment.slurm
YYYY-MM-DD_HH-MM-SS_<attempt_name>_baseline.slurm
```

**To submit jobs:**
1. Write the SLURM scripts into `job_backlog/pending/`
2. `rsync` to Explorer:
   ```bash
   rsync -avz job_backlog/pending/ explorer:/projects/Sontag_Lab_Storage/parameter-golf/job_backlog/pending/
   ```
3. SSH to Explorer and manually `sbatch` each script:
   ```bash
   ssh explorer "/bin/bash --noprofile --norc -lc 'cd /projects/Sontag_Lab_Storage/parameter-golf && sbatch job_backlog/pending/<script>.slurm'"
   ```
4. Move submitted scripts to `job_backlog/submitted/` for bookkeeping
5. **Immediately pivot** — do NOT wait for results. Start a completely different research direction.

### The pivot rule

**Once your scripts are submitted, treat them as "running" and move on.**
Do not poll `squeue` excessively. Do not idle.
Your job is to generate the next good experiment, not to babysit the queue.

### Session handoff

Before ending any session, update:
- `attempts/results.tsv` — all newly completed results
- `plan.md` — research findings, new hypotheses, updated priorities
- `PLANS.md` — which experiments are staged/running, what direction comes next

**Goal**: the next agent session must read PLANS.md and either start a new direction or collect
pending results within 2 minutes.

## Operational Lessons (March 2026)

- The crontab-based backlog submitter does NOT work on Explorer (PAM blocks crontab for d.jatkar). Manual sbatch is the only reliable submission method.
- Multiple agents fighting over the same cluster produce zero results. Enforce single-agent protocol.
- The gpu-short partition (2h walltime) is valid and preferred for 1-GPU screens.
- Always use absolute paths for DATA_PATH and TOKENIZER_PATH in SLURM scripts.
- PYTHONPATH must include attempts/_compat/ for the flash_attn_interface shim.
- Pre-EMA/pre-quant validation snapshots can be misleading. Only final int6 sliding-window BPB is the authoritative metric.
- SSH to Explorer works reliably only with non-interactive commands wrapped in `/bin/bash --noprofile --norc -lc '...'`.

## Code Judge — MANDATORY

**Every experiment's train_gpt.py MUST be adversarially reviewed before submitting to the cluster.** Speed creates bugs. Bugs mask wins. A theoretically sound idea can look like it "doesn't work" because of an off-by-one error, a wrong dimension, a missing gradient, or a misplaced detach().

### Before every cluster submission

Spawn a subagent (Agent tool) with this prompt template:

```
You are a harsh, adversarial code reviewer for a competitive ML training script.
Your job is to find BUGS, CORRECTNESS ISSUES, and PERFORMANCE PROBLEMS.

Review the file: attempts/<attempt_name>/train_gpt.py

Compare it against the base it was copied from:
records/track_10min_16mb/PR672_CosineTTT30_1.0781/train_gpt.py

Focus on:
1. CORRECTNESS: Wrong dimensions, broadcasting errors, off-by-one, wrong axis
   in reductions, missing .detach(), gradient flow where there shouldn't be,
   no gradient flow where there should be, incorrect loss computation
2. COMPETITION RULES: Check against the 10 competition rules in CLAUDE.md.
   Is the artifact under 16MB? Is there any data leakage? Any illegal eval
   tricks? Any network calls?
3. NUMERICAL ISSUES: NaN/inf risks, dtype mismatches (bf16 vs fp32),
   unnecessary precision loss, unstable operations (log of near-zero, etc.)
4. PERFORMANCE: Unnecessary recomputation, missing torch.compile compatibility,
   operations that break fusion, gratuitous memory copies, inefficient indexing
5. LOGIC: Does the code actually implement what the hypothesis.md claims?
   Read both files and verify the implementation matches the intent.

Be harsh. Assume there ARE bugs until proven otherwise.
List every issue found with file path, line number, and severity (CRITICAL/HIGH/MEDIUM/LOW).
For each CRITICAL or HIGH issue, provide the exact fix.
```

### After a "failed" experiment (BPB same or worse than baseline)

Before discarding the idea, spawn a second review subagent:

```
An experiment was expected to improve BPB but didn't.
Review attempts/<attempt_name>/train_gpt.py for bugs that could explain the failure.
The hypothesis was: <paste from hypothesis.md>
The result was: <BPB> vs baseline <baseline_BPB> (delta: <delta>)

Look specifically for:
1. Is the new code actually being executed? (Dead code, wrong branch, feature flag off)
2. Is the new code correct? (Dimensions, dtypes, gradient flow)
3. Are hyperparameters reasonable? (LR too high/low, wrong schedule, etc.)
4. Could there be a subtle interaction with existing code? (Compilation, DDP, autocast)
5. Is the change measured correctly? (Same eval, same data, same seed handling)

If you find a likely bug, provide the fix. If the code looks correct,
say so explicitly — the idea may genuinely not work.
```

### Code quality standards
- Every tensor operation must have the right dimensions. When in doubt, add asserts.
- Every new feature must be toggleable via an env var (so it can be A/B tested).
- No silent failures — if something goes wrong, it should crash, not silently degrade.
- Match the existing code style exactly (the codebase uses specific patterns for a reason).

### Escalation path (GPU usage)
1. **1xA100** — quick A/B tests (~700ms/step, cheapest)
2. **1xH200** — confirm wins (~305ms/step)
3. **8xH200** — leaderboard-equivalent (multigpu partition, ~42ms/step)

Only escalate to 8xH200 when you have a technique that clearly improves BPB on 1-GPU runs.

### When to search online
- All plan.md directions tried or stalling
- You need implementation details for a specific technique (e.g., "how to implement AOL preconditioning")
- A new paper or approach is referenced in a submission README that you want to understand
- You want to check if the competition leaderboard has moved (new PRs on upstream)

## Cluster (Explorer @ Northeastern)

- **SSH**: `ssh explorer` (user: d.jatkar)
- **Env**: `/projects/Sontag_Lab_Storage/parameter-golf-env/` (Python 3.13.5, torch 2.10+cu128)
- **Repo on cluster**: `/projects/Sontag_Lab_Storage/parameter-golf/`
- **Data on cluster**: `./data/datasets/fineweb10B_sp1024/`
- **All SLURM scripts must include**:
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
- **Home directory is over quota** — never write caches to `~/`. Always redirect to `/projects/Sontag_Lab_Storage/`.

### GPU inventory

| GPU | Nodes | Partition | Use for |
|---|---|---|---|
| H200 x8 | d4052-4055 | gpu / multigpu | Final leaderboard runs |
| A100 x3-4 | d1026, d1028-1029 | gpu | Quick experiments |
| V100 | many nodes | --- | **Do NOT use** --- no bf16, no flash attention |

### Performance benchmarks (stock baseline)

| Config | ms/step | Steps in 10 min | BPB |
|---|---|---|---|
| 1xA100 | 701 | 856 | 1.3779 |
| 1xH200 | 305 | 1,971 | 1.2987 |
| 8xH200 | 42 | 14,402 | 1.2304 |

## Experiment Protocol

### Attempt folder structure
```
attempts/
  results.tsv                    # Append-only log (TSV, see format below)
  2026-03-24_TurboMuon/
    train_gpt.py                 # Modified copy (NEVER edit repo root)
    hypothesis.md                # What, why, expected impact — BEFORE running
    experiment.log               # Training log for the experiment
    baseline.log                 # Concurrent baseline log (REQUIRED)
    submission.json              # Metadata (fill in after run if promising)
```

### Step-by-step for each experiment
```bash
# 1. Create attempt folder
mkdir -p attempts/YYYY-MM-DD_ShortName

# 2. Copy SOTA script (best known train_gpt.py, NOT repo root baseline)
cp records/track_10min_16mb/PR672_CosineTTT30_1.0781/train_gpt.py attempts/YYYY-MM-DD_ShortName/

# 3. Make your changes to the copy

# 4. Write hypothesis.md

# 5. Sync to cluster
rsync -avz attempts/YYYY-MM-DD_ShortName/ explorer:/projects/Sontag_Lab_Storage/parameter-golf/attempts/YYYY-MM-DD_ShortName/

# 6. Submit BOTH experiment and baseline (same GPU type, same duration)
# Write a SLURM script for each and sbatch them manually

# 7. Wait for completion, collect logs back
rsync -avz explorer:/projects/Sontag_Lab_Storage/parameter-golf/attempts/YYYY-MM-DD_ShortName/*.log attempts/YYYY-MM-DD_ShortName/

# 8. Log results to attempts/results.tsv (ALWAYS, even failures)
```

### results.tsv format
```
date	name	bpb	baseline_bpb	delta	gpu	status	description
2026-03-24	TurboMuon	1.2250	1.2304	-0.0054	1xH200	KEEP	Turbo-Muon AOL preconditioning
2026-03-24	CurriculumLR	1.2310	1.2304	+0.0006	1xH200	DISCARDED	Growing seq_len 512->2048, no gain
```

### Status values
- **KEEP** — BPB improved. Stack this into next attempts.
- **DISCARDED** — No improvement. Keep folder for reference, never delete.
- **VALIDATING** — Promising, running 3-seed validation on 8xH200.
- **RECORD** — Beats SOTA with 3-seed validation. Ready to submit.

## Remotes

- `origin`: `https://github.com/dhruvjatkar/parameter-golf.git` (fork — push here)
- `upstream`: `https://github.com/openai/parameter-golf.git` (upstream — never push here)
