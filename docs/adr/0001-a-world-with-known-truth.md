# 1. Build the evidence on a simulated network whose truth is known

Date: 2026-10-29. Status: accepted.

## Context

Causal estimates, forecasts and plans cannot be checked on real data: nobody knows what sessions
the raised sites would have had without the increase. A method that looks reasonable can be
badly wrong, and the usual check (does the code run, does the number look plausible) cannot
tell.

## Decision

The data comes from a structural simulator of a pan-European network (`src/cndl/sim`) with the
quantities the business cares about as explicit parameters: price elasticities by segment,
site-specific growth the analyst cannot see, charger capacity, competitor effects, trial effects,
cannibalisation. The simulator writes potential outcomes, so every estimand has a true value.
Only `cndl.sim` and `cndl.evaluation` read it; a test fails if any analysis module imports them
or names the truth.

The world is built to contain the traps a real network contains, deliberately: managers raised
prices where demand was growing, busy chargers saturate, outages hide demand, sites compete.

## Consequences

- Every answer is graded: interval coverage, bias, decision correctness, regret of the rollout.
- A world written by the person who wrote the analyses proves little, so held-out worlds are
  written blind from `heldout/WRITING_WORLDS.md` and their first run is kept.
- Simulated data is a limit to state plainly: the methods are what transfers, not the numbers.
