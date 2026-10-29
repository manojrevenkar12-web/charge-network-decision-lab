"""Site economics: what a high-power site with a given demand earns over its life.

Used by the rollout (with predicted demand) and by the evaluation (with true demand), so the plan
and its judgement use the same arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# share of mature demand reached in each of the first years after opening
RAMP = (0.75, 0.95)


@dataclass(frozen=True)
class Costs:
    capex_per_point: float
    site_fixed: float
    grid_kw_per_point: float
    battery_per_point: float
    opex_per_point_year: float
    partner_share: float
    energy_cost_eur_kwh: float
    fees_share: float
    discount_rate: float
    years: int
    kwh_per_session: float
    capacity_per_point: float

    @classmethod
    def from_config(cls, c: dict, capacity_per_point: float) -> Costs:
        keys = cls.__dataclass_fields__.keys() - {"capacity_per_point"}
        return cls(
            **{k: float(c[k]) if k != "years" else int(c[k]) for k in keys}, capacity_per_point=capacity_per_point
        )


def capex(points: np.ndarray, grid_kw: np.ndarray, c: Costs) -> np.ndarray:
    needs_battery = points * c.grid_kw_per_point > grid_kw
    return c.site_fixed + points * c.capex_per_point + np.where(needs_battery, points * c.battery_per_point, 0.0)


def npv(
    z_mature: np.ndarray, points: np.ndarray, price: np.ndarray, grid_kw: np.ndarray, growth: np.ndarray, c: Costs
) -> np.ndarray:
    """NPV in EUR.

    z_mature: demand per point on the capacity-linearised scale in the first mature year
    (any leading shape, broadcast with the site arrays); growth: yearly demand growth per site,
    applied on that scale for three years and then held flat (a planning assumption)."""
    total = -capex(points, grid_kw, c)
    cap = c.capacity_per_point
    for y in range(1, c.years + 1):
        z = z_mature + np.log1p(growth) * min(max(y - 1, 0), 3)
        spd = cap * (1 - np.exp(-np.exp(z) / cap))
        ramp = RAMP[y - 1] if y <= len(RAMP) else 1.0
        kwh = spd * ramp * points * 365 * c.kwh_per_session
        revenue = kwh * price
        margin = (
            revenue * (1 - c.fees_share - c.partner_share)
            - kwh * c.energy_cost_eur_kwh
            - points * c.opex_per_point_year
        )
        total = total + margin / (1 + c.discount_rate) ** y
    return total


def to_z(sessions_per_point_day: np.ndarray, cap: float) -> np.ndarray:
    s = np.minimum(sessions_per_point_day, 0.97 * cap)
    return np.log(-cap * np.log(1 - s / cap))
