# 3. Plan the rollout against posterior scenarios, not a point forecast

Date: 2026-10-29. Status: accepted.

## Context

Choosing 1,000 points from 1,400 candidates by ranking a point forecast ignores two things: most
of the uncertainty is site-to-site variation the features cannot explain, and new sites close to
each other take each other's customers.

## Decision

A Bayesian hierarchical model (NumPyro, country effects drawn from a common distribution) gives
posterior predictive demand scenarios for every candidate. Every scenario, candidate and size
gets a ten-year NPV from one shared economics module. A mixed-integer programme (HiGHS) chooses
sites and sizes to maximise expected NPV with about 1,000 points, within budget, country
commitments and a cap per country, charging for pairs of new sites that would compete, with a
CVaR constraint on the worst tenth of scenarios.

Price is kept out of the demand model: in the cross-section, prices were raised where demand
grew, so price "predicts" more demand. Its effect comes from the causal analysis instead.

## Consequences

- The plan is judged against the simulator's true demand, including true cannibalisation, and
  against the plan that knowing the truth allows (regret).
- The expected NPV the plan reports is optimistic when the demand model is (the evaluation shows
  by how much); the ranking of plans is what the decision needs.
