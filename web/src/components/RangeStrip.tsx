import { scaleLinear } from "d3-scale";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useId } from "react";
import { useWidth } from "../lib/useWidth";
import { DURATION, EASE, STAGGER } from "../theme/motion";

export interface StripRow {
  label: string;
  estimate: number;
  lo: number;
  hi: number;
  kind: "usual" | "credible";
  truth?: number | undefined;
}

export interface StripProps {
  title: string;
  rows: StripRow[];
  format: (x: number) => string;
  /** the range the credible rows span */
  band?: [number, number] | undefined;
  /** where the decision would change */
  threshold?: { value: number; label: string } | undefined;
  /** one truth for the whole strip (drawn across all rows) */
  truth?: number | undefined;
  showTruth: boolean;
  /** what the filled and the open markers stand for */
  legend?: { credible: string; usual: string } | undefined;
}

const L = { label: 214, right: 16, row: 34, top: 30, axis: 26, dot: 5 } as const;

/**
 * The page's central picture: every estimate with its interval on one scale, the range the
 * credible ones span, and the value at which the decision would change. If the threshold is
 * outside the range, the decision holds whatever the exact number is.
 */
export const RangeStrip = ({ title, rows, format, band, threshold, truth, showTruth, legend }: StripProps) => {
  const { ref, w } = useWidth(720);
  const reduce = useReducedMotion();
  const id = useId();
  const narrow = w < 520;
  const labelW = narrow ? 0 : L.label;
  const rowH = narrow ? L.row + 16 : L.row;
  const values = rows.flatMap((r) => [r.lo, r.hi, r.estimate, ...(r.truth !== undefined ? [r.truth] : [])]);
  if (threshold) values.push(threshold.value);
  if (truth !== undefined) values.push(truth);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
  const x = scaleLinear().domain([lo - pad, hi + pad]).range([labelW, w - L.right]).nice();
  const height = L.top + rows.length * rowH + L.axis;
  const y = (i: number) => L.top + i * rowH + rowH / 2 + (narrow ? 8 : 0);
  const holds = threshold && band ? threshold.value < band[0] || threshold.value > band[1] : undefined;
  const tone = holds === undefined ? "holds" : holds ? "holds" : "flips";

  return (
    <figure className="strip" ref={ref} aria-labelledby={`${id}-t`}>
      <figcaption id={`${id}-t`} className="sr-only">
        {title}
      </figcaption>
      <svg width={w} height={height} viewBox={`0 0 ${w} ${height}`} role="img" aria-labelledby={`${id}-t ${id}-d`}>
        <desc id={`${id}-d`}>
          {rows.map((r) => `${r.label}: ${format(r.estimate)} (${format(r.lo)} to ${format(r.hi)})`).join(". ")}
          {threshold ? `. The decision changes at ${format(threshold.value)}.` : ""}
        </desc>
        {x.ticks(6).map((t) => (
          <g key={t} className="tick">
            <line x1={x(t)} x2={x(t)} y1={L.top - 6} y2={height - L.axis + 4} />
            <text x={x(t)} y={height - 6} textAnchor="middle">
              {format(t)}
            </text>
          </g>
        ))}
        {band && <rect className="band band--credible" x={x(band[0])} y={L.top - 6} width={Math.max(1, x(band[1]) - x(band[0]))} height={rows.length * rowH + 8 + (narrow ? 8 : 0)} opacity={0.45} rx={4} />}
        {threshold && (
          <g>
            <line className={`threshold threshold--${tone}`} x1={x(threshold.value)} x2={x(threshold.value)} y1={L.top - 14} y2={height - L.axis + 4} />
            <text className={`threshold-label--${tone}`} x={x(threshold.value)} y={L.top - 18} textAnchor={x(threshold.value) > w * 0.7 ? "end" : "start"} dx={x(threshold.value) > w * 0.7 ? -4 : 4}>
              {threshold.label}
            </text>
          </g>
        )}
        {rows.map((r, i) => (
          <motion.g
            key={r.label}
            initial={reduce ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: DURATION.base, delay: i * STAGGER, ease: EASE }}
          >
            <text className={`row-label row-label--${r.kind}`} x={narrow ? x.range()[0] : 0} y={narrow ? y(i) - 12 : y(i) + 4}>
              {r.label}
            </text>
            <line className={`ci ${r.kind === "usual" ? "ci--usual" : ""}`} x1={x(r.lo)} x2={x(r.hi)} y1={y(i)} y2={y(i)} />
            <circle className={r.kind === "usual" ? "dot dot--usual" : "dot"} cx={x(r.estimate)} cy={y(i)} r={L.dot} />
            <AnimatePresence>
              {showTruth && r.truth !== undefined && (
                <motion.path
                  key="truth"
                  className="truth"
                  d={`M ${x(r.truth)} ${y(i) - 6} L ${x(r.truth) + 6} ${y(i)} L ${x(r.truth)} ${y(i) + 6} L ${x(r.truth) - 6} ${y(i)} Z`}
                  initial={reduce ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: DURATION.base, ease: EASE }}
                />
              )}
            </AnimatePresence>
          </motion.g>
        ))}
        <AnimatePresence>
          {showTruth && truth !== undefined && (
            <motion.g key="truth-line" initial={reduce ? false : { opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: DURATION.base }}>
              <line className="truth-line" x1={x(truth)} x2={x(truth)} y1={L.top - 6} y2={height - L.axis + 4} />
              <text className="truth-label" x={x(truth) + 5} y={height - L.axis - 4}>
                truth {format(truth)}
              </text>
            </motion.g>
          )}
        </AnimatePresence>
        <line className="axis" x1={x.range()[0]} x2={x.range()[1]} y1={height - L.axis + 4} y2={height - L.axis + 4} />
      </svg>
      <div className="strip-caption">
        <span>
          <i className="key key--estimate" /> {legend?.credible ?? "credible estimate and 95% interval"}
        </span>
        {rows.some((r) => r.kind === "usual") && (
          <span>
            <i className="key key--usual" /> {legend?.usual ?? "the usual comparison"}
          </span>
        )}
        {band && (
          <span>
            <i className="key key--range" /> range the credible estimates span
          </span>
        )}
        {showTruth && (
          <span>
            <i className="key key--truth" /> the simulator's truth
          </span>
        )}
      </div>
    </figure>
  );
};
