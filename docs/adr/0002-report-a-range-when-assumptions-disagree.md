# 2. When credible assumptions disagree, report the range and decide on it

Date: 2026-10-29. Status: accepted.

## Context

The 2025 price increase was not randomised. The event study shows the raised sites were already
growing faster, so parallel trends fails. The methods that relax it rest on assumptions the
data cannot test: synthetic DiD assumes reweighted controls would have tracked the raised sites;
trend imputation assumes each site would have kept its own growth path. They give different
answers, in a known direction.

## Decision

Report the range the credible estimates span (with their bootstrap intervals), state which
assumption gives which end, and check whether the decision changes anywhere in the range. If it
does not (here: revenue rises at every effect size), decide; if it does, recommend the experiment
that would settle it.

## Consequences

- Executives see a range and a robust decision rather than a precise number that may be wrong.
- The usual estimates are shown too, with why they are wrong: before/after and two-way fixed
  effects say sessions went up.
