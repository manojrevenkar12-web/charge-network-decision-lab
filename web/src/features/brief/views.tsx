import type { ReactNode } from "react";
import { LineBand } from "../../components/LineBand";
import type { StripProps } from "../../components/RangeStrip";
import { SiteMap } from "../../components/SiteMap";
import type { Competitor, Evaluation, Forecast, PriceChange, Rollout, Trial } from "../../lib/data";
import { eur, gwh, logPct, METHOD_NAMES, pct } from "../../lib/format";

export interface View {
  strip: Omit<StripProps, "showTruth">;
  chart: ReactNode;
  chartTitle: string;
  chartLead: string;
  verdicts: string[];
}

const yes = (b: boolean | null | undefined) => (b ? "inside" : "outside");

export const priceChangeView = (r: PriceChange, ev?: Evaluation): View => {
  const truth = ev ? logPct(ev.price_change.truth_log) : undefined;
  const rows = [
    ...r.usual_estimates.map((e) => ({ ...e, kind: "usual" as const })),
    ...r.credible_estimates.map((e) => ({ ...e, kind: "credible" as const })),
  ].map((e) => ({ label: METHOD_NAMES[e.method] ?? e.method, estimate: logPct(e.estimate), lo: logPct(e.ci[0]), hi: logPct(e.ci[1]), kind: e.kind }));
  const breakEven = Math.expm1(-r.price_increase_log);
  const es = r.event_study.coefficients;
  return {
    strip: {
      title: "Effect of the price increase on sessions, by method",
      rows,
      format: (v) => pct(v, 0),
      band: [logPct(r.range_log[0]), logPct(r.range_log[1])],
      threshold: { value: breakEven, label: `revenue falls below ${pct(breakEven, 0)}` },
      truth,
    },
    chartTitle: "Before and after each site's increase",
    chartLead:
      "Under parallel trends the weeks before an increase should hover around zero. They do not: the raised sites were already pulling ahead, which is why the usual comparisons understate the loss.",
    chart: (
      <LineBand
        title="Event study: effect by week relative to the increase"
        series={es.map((c) => ({ x: c.event_week, y: c.estimate, lo: c.lo, hi: c.hi }))}
        marker={{ x: 0, label: "price raised" }}
        xLabel="Weeks relative to each site's increase; effect on log weekly sessions, 95% bootstrap band"
        yFormat={(v) => pct(v, 0)}
        xFormat={(v) => `${v}`}
      />
    ),
    verdicts: ev
      ? [
          `The truth (${pct(truth ?? 0)}) is ${yes(ev.price_change.range_covers_truth)} the reported range.`,
          ...ev.price_change.methods.map((m) => `${METHOD_NAMES[m.method] ?? m.method}: off by ${pct(m.error)}, interval ${m.covers ? "covers" : "misses"} the truth.`),
          `Keeping the price was ${ev.price_change.decision_correct ? "the right" : "the wrong"} decision.`,
        ]
      : [],
  };
};

export const trialView = (r: Trial, ev?: Evaluation): View => {
  const rev = r.effects.revenue_eur;
  const net = r.net_value_per_site_day;
  const relief = r.peak_sessions_relieved_per_site_day;
  const rows = [
    { label: "Revenue per site and day", estimate: rev?.estimate ?? 0, lo: rev?.ci[0] ?? 0, hi: rev?.ci[1] ?? 0, kind: "credible" as const, truth: ev?.trial.revenue_eur?.truth },
    {
      label: "Value of relieved daytime demand",
      estimate: relief[1] * r.peak_session_value_eur,
      lo: relief[0] * r.peak_session_value_eur,
      hi: relief[2] * r.peak_session_value_eur,
      kind: "credible" as const,
    },
    { label: "Net value", estimate: net[1], lo: net[0], hi: net[2], kind: "credible" as const, truth: ev?.trial.true_net_value_per_site_day },
  ];
  return {
    strip: {
      title: "Value of the night discount per site and day",
      rows,
      format: (v) => eur(v),
      threshold: { value: 0, label: "pays for itself" },
      band: [net[0], net[2]],
      legend: { credible: "estimate and 95% interval", usual: "" },
    },
    chartTitle: "The estimate as the trial ran",
    chartLead: "The revenue effect, recomputed each day of the trial. It settles early; an always-valid sequential test would have allowed stopping well before eight weeks.",
    chart: (
      <LineBand
        title="Running estimate of the revenue effect"
        series={r.sequential.path.map((p) => ({ x: p.day, y: p.estimate }))}
        marker={r.sequential.stopped_on_day ? { x: r.sequential.stopped_on_day, label: "could stop" } : undefined}
        xLabel="Day of the trial; within-site difference in daily revenue, EUR"
        yFormat={(v) => eur(v)}
        xFormat={(v) => `${v}`}
      />
    ),
    verdicts: ev
      ? [
          ...(
            [
              ["revenue_eur", (v: number) => eur(v)],
              ["offpeak_share", (v: number) => `${(v * 100).toFixed(1)} points`],
              ["log_peak_sessions", (v: number) => pct(Math.expm1(v))],
            ] as const
          ).map(([k, f]) => {
            const t = ev.trial[k];
            return `${r.effects[k]?.label ?? k}: truth ${f(t?.truth ?? 0)}, estimate ${f(t?.estimate ?? 0)}, ${t?.covers ? "inside" : "outside"} the interval.`;
          }),
          `True net value ${eur(ev.trial.true_net_value_per_site_day)} a site-day: the decision was ${ev.trial.decision_correct ? "right" : "wrong"}.`,
        ]
      : [],
  };
};

export const competitorView = (r: Competitor, ev?: Evaluation): View => {
  const sc = ev?.competitor.synthetic_control;
  const pooled = ev?.competitor.pooled;
  const opened = new Date(r.competitor_opened).getTime();
  return {
    strip: {
      title: "Sessions lost to a competitor next door",
      rows: [
        { label: `${r.site_id} (synthetic control)`, estimate: logPct(r.effect_log), lo: logPct(r.ci_log[0]), hi: logPct(r.ci_log[1]), kind: "credible", truth: sc?.truth != null ? logPct(sc.truth) : undefined },
        { label: `All ${r.pooled.events} openings`, estimate: logPct(r.pooled.estimate_log), lo: logPct(r.pooled.ci_log[0]), hi: logPct(r.pooled.ci_log[1]), kind: "credible", truth: pooled ? logPct(pooled.truth) : undefined },
      ],
      format: (v) => pct(v, 0),
    },
    chartTitle: `${r.site_id} against its synthetic twin`,
    chartLead: "The gap between the site and a weighted mix of comparable sites. Before the opening the two move together; afterwards the site falls behind.",
    chart: (
      <LineBand
        title="Gap between the site and its synthetic control"
        series={r.series.map((p) => ({ x: new Date(p.week).getTime(), y: p.gap }))}
        marker={{ x: opened, label: "competitor opens" }}
        xLabel="Week; difference in log weekly sessions"
        yFormat={(v) => pct(v, 0)}
        xFormat={(v) => new Date(v).toLocaleDateString("en-GB", { month: "short", year: "2-digit" })}
      />
    ),
    verdicts: ev
      ? [
          `At the site: truth ${pct(logPct(sc?.truth ?? 0))}, ${yes(sc?.covers)} the interval.`,
          `Across the network: truth ${pct(logPct(pooled?.truth ?? 0))}, ${yes(pooled?.covers)} the interval.`,
        ]
      : [],
  };
};

export const forecastView = (r: Forecast, ev?: Evaluation): View => {
  const t0 = new Date(r.total[0]?.month ?? 0).getTime();
  const lo = r.total.reduce((s, x) => s + x.lo, 0) / 1e6;
  const hi = r.total.reduce((s, x) => s + x.hi, 0) / 1e6;
  const truth = ev?.forecast.true_total_gwh;
  return {
    strip: {
      title: "Energy in the next twelve months",
      rows: [{ label: "Existing network", estimate: r.next_12_months_gwh, lo, hi, kind: "credible", truth }],
      format: (v) => gwh(v),
    },
    chartTitle: "Monthly energy, history and forecast",
    chartLead: `Grey: what the network sold. Blue: the forecast with its ${Math.round(r.interval * 100)}% interval, from the backtest's own errors.`,
    chart: (
      <LineBand
        title="Monthly energy sold by the existing network"
        series={r.total.map((p) => ({ x: new Date(p.month).getTime(), y: p.kwh / 1e6, lo: p.lo / 1e6, hi: p.hi / 1e6 }))}
        context={r.history_total.map((p) => ({ x: new Date(p.month).getTime(), y: p.kwh / 1e6 }))}
        marker={{ x: t0, label: "forecast" }}
        xLabel="Month; GWh"
        yFormat={(v) => `${v.toFixed(0)}`}
        xFormat={(v) => new Date(v).toLocaleDateString("en-GB", { month: "short", year: "2-digit" })}
      />
    ),
    verdicts: ev
      ? [
          `Countries missed by ${(ev.forecast.mape_country * 100).toFixed(1)}% a month on average over ${ev.forecast.months} months.`,
          `${Math.round(ev.forecast.coverage_country * 100)}% of country-months fell inside the ${Math.round(ev.forecast.nominal * 100)}% intervals.`,
        ]
      : [],
  };
};

export const rolloutView = (r: Rollout, ev?: Evaluation): View => {
  const order = ["highest_traffic", "expected_npv_greedy", "optimised", "optimised_risk"];
  const rows = order
    .filter((k) => r.plans[k])
    .map((k) => {
      const p = r.plans[k] as NonNullable<(typeof r.plans)[string]>;
      return {
        label: METHOD_NAMES[k] ?? k,
        estimate: p.expected_npv_eur,
        lo: p.cvar_npv_eur,
        hi: p.expected_npv_eur,
        kind: k === r.recommended ? ("credible" as const) : ("usual" as const),
        truth: ev?.rollout.plans[k]?.true_npv_eur,
      };
    });
  const chosen = new Map((r.plans[r.recommended]?.selection ?? []).map((s) => [s.candidate_id, s.points]));
  return {
    strip: {
      title: "Ten-year NPV of each plan: expected, and in the worst tenth of scenarios",
      rows,
      format: (v) => eur(v),
      truth: ev?.rollout.oracle_npv_eur,
      legend: { credible: "recommended plan: expected NPV, line to the worst tenth", usual: "other ways to choose" },
    },
    chartTitle: "Where the recommended sites are",
    chartLead: "Candidates across the eleven countries; the plan spreads across countries within the one-third cap and avoids sites that would take each other's customers.",
    chart: (
      <SiteMap
        title="Candidate locations and the recommended sites"
        sites={r.candidate_npv.map((c) => ({ id: c.candidate_id, lat: c.lat, lon: c.lon, chosen: chosen.has(c.candidate_id), points: chosen.get(c.candidate_id) ?? 0 }))}
      />
    ),
    verdicts: ev
      ? [
          `With perfect knowledge of demand the best plan earns ${eur(ev.rollout.oracle_npv_eur)} (the violet line); the diamonds are what each plan really earns.`,
          ...order
            .filter((k) => ev.rollout.plans[k])
            .map((k) => `${METHOD_NAMES[k] ?? k}: really earns ${eur(ev.rollout.plans[k]?.true_npv_eur ?? 0)}, ${Math.round((ev.rollout.plans[k]?.share_of_oracle ?? 0) * 100)}% of the best.`),
        ]
      : [],
  };
};
