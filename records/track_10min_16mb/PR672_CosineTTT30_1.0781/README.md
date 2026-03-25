# PR #672: Cosine TTT (TTT_EPOCHS=30)

- **BPB**: 1.0781 (3-seed mean, std=0.0041)
- **Source**: https://github.com/openai/parameter-golf/pull/672
- **Status**: Open PR (not yet merged upstream)
- **Key change**: TTT_EPOCHS=30 with cosine TTT schedule (up from 3 in the March 23 merged SOTA)
- **Eval time**: 494s TTT + 96s sliding = 590s total (under 10min cap)
- **Base**: LeakyReLU² + Legal TTT + Parallel Muon (PR #549 stack)

This directory serves as the reference baseline for all new experiments.
Every experiment must compare against this algorithm to be meaningful.
