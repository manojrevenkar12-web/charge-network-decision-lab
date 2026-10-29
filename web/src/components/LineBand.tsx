import { scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";
import { useId } from "react";
import { useWidth } from "../lib/useWidth";

export interface Point {
  x: number;
  y: number;
  lo?: number | undefined;
  hi?: number | undefined;
}

const M = { top: 14, right: 16, bottom: 26, left: 56, h: 240 } as const;

/** A line with an uncertainty band, a zero line and an optional vertical marker. */
export const LineBand = ({
  title,
  series,
  context,
  marker,
  xLabel,
  yFormat,
  xFormat,
}: {
  title: string;
  series: Point[];
  context?: Point[] | undefined;
  marker?: { x: number; label: string } | undefined;
  xLabel: string;
  yFormat: (v: number) => string;
  xFormat: (v: number) => string;
}) => {
  const { ref, w } = useWidth(720);
  const id = useId();
  const all = [...series, ...(context ?? [])];
  const ys = all.flatMap((p) => [p.y, p.lo ?? p.y, p.hi ?? p.y]);
  const x = scaleLinear()
    .domain([Math.min(...all.map((p) => p.x)), Math.max(...all.map((p) => p.x))])
    .range([M.left, w - M.right]);
  const y = scaleLinear()
    .domain([Math.min(0, ...ys), Math.max(0, ...ys)])
    .range([M.h - M.bottom, M.top])
    .nice();
  const band = area<Point>()
    .x((p) => x(p.x))
    .y0((p) => y(p.lo ?? p.y))
    .y1((p) => y(p.hi ?? p.y));
  const path = line<Point>()
    .x((p) => x(p.x))
    .y((p) => y(p.y));
  const hasBand = series.some((p) => p.lo !== undefined);
  return (
    <figure className="chart" ref={ref}>
      <svg width={w} height={M.h} viewBox={`0 0 ${w} ${M.h}`} role="img" aria-labelledby={`${id}-t`}>
        <title id={`${id}-t`}>{title}</title>
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line className={t === 0 ? "zero" : "grid"} x1={M.left} x2={w - M.right} y1={y(t)} y2={y(t)} />
            <text x={M.left - 8} y={y(t) + 4} textAnchor="end">
              {yFormat(t)}
            </text>
          </g>
        ))}
        {x.ticks(Math.max(3, Math.floor(w / 110))).map((t) => (
          <text key={t} x={x(t)} y={M.h - 6} textAnchor="middle">
            {xFormat(t)}
          </text>
        ))}
        {hasBand && <path className="band" d={band(series) ?? ""} />}
        {context && <path className="line line--muted" d={path(context) ?? ""} />}
        <path className="line" d={path(series) ?? ""} />
        {marker && (
          <g>
            <line className="marker" x1={x(marker.x)} x2={x(marker.x)} y1={M.top} y2={M.h - M.bottom} />
            <text className="marker-label" x={x(marker.x) + 4} y={M.top + 10}>
              {marker.label}
            </text>
          </g>
        )}
      </svg>
      <figcaption className="chart-caption">{xLabel}</figcaption>
    </figure>
  );
};
