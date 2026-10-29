import { scaleLinear, scaleSqrt } from "d3-scale";
import { useId } from "react";
import { useWidth } from "../lib/useWidth";

export interface MapSite {
  id: string;
  lat: number;
  lon: number;
  chosen: boolean;
  points: number;
}

/** Candidates on a plain longitude/latitude plane (no map tiles: nothing leaves the page). */
export const SiteMap = ({ sites, title }: { sites: MapSite[]; title: string }) => {
  const { ref, w } = useWidth(720);
  const id = useId();
  const lon = [Math.min(...sites.map((s) => s.lon)), Math.max(...sites.map((s) => s.lon))] as const;
  const lat = [Math.min(...sites.map((s) => s.lat)), Math.max(...sites.map((s) => s.lat))] as const;
  // keep the shape of Europe: degrees of longitude are shorter in the north
  const k = Math.cos(((lat[0] + lat[1]) / 2) * (Math.PI / 180));
  const h = Math.round(Math.min(560, (w * (lat[1] - lat[0])) / ((lon[1] - lon[0]) * k)));
  const x = scaleLinear().domain(lon).range([8, w - 8]);
  const y = scaleLinear().domain(lat).range([h - 8, 8]);
  const r = scaleSqrt().domain([0, 12]).range([0, 7]);
  const chosen = sites.filter((s) => s.chosen);
  return (
    <figure className="chart" ref={ref}>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-labelledby={`${id}-t`}>
        <title id={`${id}-t`}>{title}</title>
        {sites
          .filter((s) => !s.chosen)
          .map((s) => (
            <circle key={s.id} className="site" cx={x(s.lon)} cy={y(s.lat)} r={1.8} />
          ))}
        {chosen.map((s) => (
          <circle key={s.id} className="site site--chosen" cx={x(s.lon)} cy={y(s.lat)} r={r(s.points)} />
        ))}
      </svg>
      <figcaption className="chart-caption">
        Grey: all candidate locations. Blue: the recommended sites, sized by the number of points.
      </figcaption>
    </figure>
  );
};
