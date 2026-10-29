import { motion, useReducedMotion } from "motion/react";
import { RangeStrip } from "../../components/RangeStrip";
import { ErrorState, Skeleton } from "../../components/States";
import { type Brief, type Evaluation, useData } from "../../lib/data";
import { DURATION, EASE } from "../../theme/motion";
import { competitorView, forecastView, priceChangeView, rolloutView, trialView, type View } from "./views";

const DETAIL: Record<string, string> = {
  "price-change": "price_change",
  "night-discount": "trial",
  competitor: "competitor",
  forecast: "forecast",
  rollout: "rollout",
};

const build = (id: string, detail: unknown, ev?: Evaluation): View | null => {
  switch (id) {
    case "price-change":
      return priceChangeView(detail as Parameters<typeof priceChangeView>[0], ev);
    case "night-discount":
      return trialView(detail as Parameters<typeof trialView>[0], ev);
    case "competitor":
      return (detail as { available?: boolean }).available ? competitorView(detail as Parameters<typeof competitorView>[0], ev) : null;
    case "forecast":
      return forecastView(detail as Parameters<typeof forecastView>[0], ev);
    case "rollout":
      return rolloutView(detail as Parameters<typeof rolloutView>[0], ev);
    default:
      return null;
  }
};

export const BriefView = ({ brief, index, showTruth }: { brief: Brief; index: number; showTruth: boolean }) => {
  const detail = useData<unknown>(DETAIL[brief.id] ?? brief.id);
  const evaluation = useData<Evaluation>("evaluation", showTruth);
  const reduce = useReducedMotion();
  let view: View | null = null;
  let shapeError: Error | null = null;
  if (detail.data) {
    try {
      view = build(brief.id, detail.data, showTruth ? evaluation.data : undefined);
    } catch (e) {
      // a result file from an older run, or a different shape: say so instead of breaking the page
      shapeError = new Error(`The ${DETAIL[brief.id] ?? brief.id} results do not have the expected shape (${(e as Error).message}). Re-run cndl run and cndl export-web.`);
    }
  }
  const holds = brief.tone !== "flips";

  return (
    <motion.article
      key={brief.id}
      className="brief"
      initial={reduce ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: DURATION.reveal, ease: EASE }}
      aria-labelledby={`brief-${brief.id}`}
    >
      <p className="kicker">
        Question {index + 1}: {brief.title.toLowerCase()}
      </p>
      <h2 id={`brief-${brief.id}`}>{brief.question}</h2>
      <p className="answer">{brief.answer}</p>

      {detail.isPending ? (
        <div className="section">
          <Skeleton lines={3} label="Loading the evidence" />
        </div>
      ) : detail.isError || shapeError ? (
        <div className="section">
          <ErrorState error={detail.error ?? shapeError} onRetry={() => void detail.refetch()} />
        </div>
      ) : view ? (
        <RangeStrip {...view.strip} showTruth={showTruth && !!evaluation.data} />
      ) : null}

      {brief.decision && (
        <div className={`decision decision--${holds ? "holds" : "flips"}`}>
          <b>{holds ? "Decision, robust to the uncertainty" : "Decision, sensitive to the uncertainty"}</b>
          <span>{brief.decision}</span>
        </div>
      )}

      {showTruth && view && (
        <section className="section" aria-labelledby={`truth-${brief.id}`}>
          <h3 id={`truth-${brief.id}`}>Checked against the simulator's truth</h3>
          {evaluation.isPending ? (
            <Skeleton lines={2} label="Loading the evaluation" />
          ) : evaluation.isError ? (
            <ErrorState error={evaluation.error} onRetry={() => void evaluation.refetch()} />
          ) : (
            <ul className="verdicts">
              {view.verdicts.map((v) => (
                <li key={v}>
                  <span className="mark">truth</span>
                  <span>{v}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {view && (
        <section className="section" aria-labelledby={`chart-${brief.id}`}>
          <h3 id={`chart-${brief.id}`}>{view.chartTitle}</h3>
          <p className="lead">{view.chartLead}</p>
          {view.chart}
        </section>
      )}

      <section className="section" aria-labelledby={`rests-${brief.id}`}>
        <h3 id={`rests-${brief.id}`}>What this rests on</h3>
        <ul className="points">
          {brief.rests_on.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </section>
      <section className="section" aria-labelledby={`change-${brief.id}`}>
        <h3 id={`change-${brief.id}`}>What would change it</h3>
        <ul className="points">
          {brief.would_change.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      </section>
      {brief.method && <p className="method">{brief.method}</p>}
    </motion.article>
  );
};
