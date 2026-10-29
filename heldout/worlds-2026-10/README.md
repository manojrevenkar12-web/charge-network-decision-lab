# Held-out worlds: worlds-2026-10

Written blind from `heldout/WRITING_WORLDS.md`, `config/world.yaml` and `src/cndl/sim/world.py` only.

- `worlds/trend-targeted-increase.yaml`: 2025 price increase very strongly targeted on unseen local growth, slightly larger increase, more spread in trends.
- `worlds/random-increase.yaml`: price increase given at random and smaller; checks that a selection correction does not push away from a simple, correct estimate.
- `worlds/busy-sites-utilisation-targeted.yaml`: saturated network (more demand per point, lower capacity); increase given to the busiest sites whose growth was slowing, so the bias runs the other way.
- `worlds/segment-elasticity-spread.yaml`: price sensitivity very different by segment and site, slower driver adjustment to the new price.
- `worlds/weak-trial-carryover.yaml`: small, short off-peak trial (40 sites, 6 weeks) with small effects and heavy carryover to the next day.
- `worlds/deep-discount-short-window.yaml`: large, long trial (120 sites, 10 weeks), deep discount over a shorter night window; behaviour shifts clearly but revenue falls.
- `worlds/competition-wave.yaml`: heavy competition bunched around the price waves, larger and more varied effects, slower phase-in, urban hubs hit harder.
- `worlds/noisy-unreliable.yaml`: overdispersed demand, larger persistent country shocks, stronger seasonality, more quality spread, outages far more frequent and longer.
- `worlds/mature-slow-market.yaml`: slow BEV growth with weaker pass-through, faster ramp-ups, tight rollout economics and a smaller budget.
- `worlds/fast-growth-young-network.yaml`: large network weighted to fast-growing eastern markets with slow, deep ramp-ups; bigger, cheaper rollout with more grid-constrained candidates.
- `worlds/small-network-faint-signal.yaml`: small operator, small two-wave increase, few competitors, small trial; all effects faint against the noise.
- `worlds/dense-cannibalisation-rollout.yaml`: stronger, wider cannibalisation, larger site options, costlier grid and batteries, higher partner share and discount rate, longer horizon.
