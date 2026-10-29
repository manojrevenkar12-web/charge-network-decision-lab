const nf = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

export const pct = (x: number, digits = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(digits)}%`;
export const eur = (x: number) => {
  const a = Math.abs(x);
  const s = a >= 1e6 ? `€${nf.format(a / 1e6)}m` : `€${nf.format(a)}`;
  return x < 0 ? `−${s}` : s;
};
export const gwh = (x: number) => `${nf.format(x)} GWh`;
export const fmt = (unit: "pct" | "eur" | "gwh", x: number) => (unit === "pct" ? pct(x) : unit === "eur" ? eur(x) : gwh(x));
/** log effect to a percentage change */
export const logPct = (l: number) => Math.expm1(l);
export const METHOD_NAMES: Record<string, string> = {
  before_after: "Before and after",
  twfe: "Two-way fixed effects",
  imputation_parallel_trends: "Imputation, parallel trends",
  synthetic_did: "Synthetic DiD",
  trend_imputation: "Each site's own trend",
  highest_traffic: "Highest traffic first",
  expected_npv_greedy: "Best expected NPV, one by one",
  optimised: "Optimised",
  optimised_risk: "Optimised, worst tenth protected",
};
