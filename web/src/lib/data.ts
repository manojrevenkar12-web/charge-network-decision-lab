import { useQuery } from "@tanstack/react-query";

/** Everything the page shows comes from the pipeline's result files, copied by `cndl export-web`. */
export class LoadError extends Error {
  constructor(
    readonly file: string,
    readonly status: number,
  ) {
    super(`${file} could not be loaded (${status || "network error"})`);
  }
}

export const load = async <T>(file: string): Promise<T> => {
  let res: Response;
  try {
    res = await fetch(`./data/${file}.json`);
  } catch {
    throw new LoadError(file, 0);
  }
  if (!res.ok) throw new LoadError(file, res.status);
  return (await res.json()) as T;
};

export const useData = <T>(file: string, enabled = true) =>
  useQuery({ queryKey: ["data", file], queryFn: () => load<T>(file), staleTime: Number.POSITIVE_INFINITY, enabled });

export interface Brief {
  id: string;
  title: string;
  question: string;
  answer: string;
  number?: { label: string; value: number; low: number; high: number; unit: "pct" | "eur" | "gwh" };
  decision?: string;
  /** holds: the decision is the same across the uncertainty; flips: somewhere it would change */
  tone?: "holds" | "flips";
  rests_on: string[];
  would_change: string[];
  method?: string;
}

export interface Estimate {
  method: string;
  estimate: number;
  ci: [number, number];
}

export interface PriceChange {
  usual_estimates: Estimate[];
  credible_estimates: Estimate[];
  range_log: [number, number];
  price_increase_log: number;
  revenue_range_pct: [number, number];
  event_study: { coefficients: Array<{ event_week: number; estimate: number; lo: number; hi: number }>; pretrend_p: number };
}

export interface Trial {
  effects: Record<string, { label: string; estimate: number; ci: [number, number]; control_mean: number }>;
  net_value_per_site_day: [number, number, number];
  peak_sessions_relieved_per_site_day: [number, number, number];
  peak_session_value_eur: number;
  sequential: { stopped_on_day: number | null; path: Array<{ day: number; estimate: number; always_valid_p: number }> };
  decision: string;
}

export interface Competitor {
  available: boolean;
  site_id: string;
  competitor_opened: string;
  effect_log: number;
  ci_log: [number, number];
  pooled: { estimate_log: number; ci_log: [number, number]; events: number };
  series: Array<{ week: string; gap: number }>;
}

export interface Forecast {
  total: Array<{ month: string; kwh: number; lo: number; hi: number }>;
  history_total: Array<{ month: string; kwh: number }>;
  next_12_months_gwh: number;
  backtest: Record<string, { mape_country: number }>;
  chosen: string;
  interval: number;
}

export interface Plan {
  sites: number;
  points: number;
  capex_eur: number;
  expected_npv_eur: number;
  cvar_npv_eur: number;
  selection: Array<{ candidate_id: string; points: number }>;
}

export interface Rollout {
  plans: Record<string, Plan>;
  recommended: string;
  candidate_npv: Array<{ candidate_id: string; lat: number; lon: number; country: string; npv_mean: number }>;
}

export interface Evaluation {
  price_change: { truth_log: number; range_covers_truth: boolean; methods: Array<{ method: string; covers: boolean; error: number }>; decision_correct: boolean };
  trial: Record<string, { truth: number; estimate: number; covers: boolean }> & { true_net_value_per_site_day: number; decision_correct: boolean | null };
  competitor: { available: boolean; synthetic_control: { truth: number | null; covers: boolean | null }; pooled: { truth: number; covers: boolean } };
  forecast: { mape_country: number; coverage_country: number; nominal: number; true_total_gwh?: number; months: number };
  rollout: { oracle_npv_eur: number; plans: Record<string, { true_npv_eur: number; share_of_oracle: number }> };
}
