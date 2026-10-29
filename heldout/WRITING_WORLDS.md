# Writing a held-out set of worlds

The decision lab answers five questions about a simulated charge-point network and is judged
against the simulator's truth. On the development world (`config/world.yaml`) its author could
see the truth while building, so a good score there proves little. A held-out set is a handful of
other worlds, written by someone who has **not read the analysis code**, run once; that first run
is the result that counts.

## What you may read

- this file,
- `config/world.yaml` (every parameter of the simulated world, with comments),
- `src/cndl/sim/world.py` (how the parameters turn into data),
- `README.md` if it exists.

Do not open `src/cndl/analysis/`, `src/cndl/evaluation.py`, `src/cndl/briefs.py`, `config/analysis.yaml`,
`config/release.yaml`, `tests/`, `reports/` or other held-out sets. The point is that the worlds
are not shaped around what the analyses happen to handle.

## What to write

Write 8 to 12 files `heldout/<set>/worlds/<name>.yaml`. Each is an override of
`config/world.yaml`: only the keys you change, nested the same way. Every world must still be a
plausible European charging network. Vary what a real network could plausibly be like, and
include some worlds that should be hard for an analyst, for example:

- the 2025 price increase given to sites chosen differently (more or less on growth, on
  utilisation, at random), larger or smaller, with stronger or weaker price sensitivity, or
  sensitivity that differs a lot between segments;
- more or less competition, larger or smaller competitor effects;
- a night-discount trial with smaller effects, more carryover, fewer sites or weeks;
- faster or slower growth, more saturation (busier sites), noisier demand, more outages;
- different candidate economics and cannibalisation for next year's rollout;
- a different random seed in every world.

Keep `start`, `end` and `future_weeks` unchanged, keep `network.sites` between 600 and 1500 and
`rollout.candidates` between 600 and 1600, and keep every number within what the comments in
`config/world.yaml` make plausible. Each file starts with a comment saying what the world is
meant to test.

When done, write `heldout/<set>/README.md`: one line per world (file name and what it tests).
