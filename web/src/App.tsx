import { useEffect, useState } from "react";
import { BriefView } from "./features/brief/BriefView";
import { EmptyState, ErrorState, Skeleton } from "./components/States";
import { type Brief, useData } from "./lib/data";

const fromHash = () => (typeof window === "undefined" ? "" : decodeURIComponent(window.location.hash.slice(1)));

export const App = () => {
  const briefs = useData<Brief[]>("briefs");
  const [current, setCurrent] = useState(fromHash());
  const [showTruth, setShowTruth] = useState(false);

  useEffect(() => {
    const on = () => setCurrent(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const list = briefs.data ?? [];
  const index = Math.max(0, list.findIndex((b) => b.id === current));
  const brief = list[index];

  return (
    <div className="room">
      <aside className="rail" aria-label="Questions">
        <div>
          <h1>Decision room</h1>
          <p>Five decisions for a pan-European charging network, each with how sure we are.</p>
        </div>
        {briefs.isPending ? (
          <Skeleton lines={5} label="Loading the questions" />
        ) : (
          <nav aria-label="Briefs">
            <ul className="index">
              {list.map((b, i) => (
                <li key={b.id}>
                  <button
                    type="button"
                    aria-current={b.id === brief?.id ? "page" : undefined}
                    onClick={() => {
                      window.location.hash = b.id;
                      setCurrent(b.id);
                    }}
                  >
                    <span className="q">Q{i + 1}</span>
                    <span className="t">{b.title}</span>
                    <span className="d">{b.number?.label ?? ""}</span>
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        )}
        <div className="truth-switch">
          <label>
            <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
            Show the simulator's truth
          </label>
          <span>The network is simulated, so every answer can be checked. The analyses never see the truth.</span>
        </div>
      </aside>
      <main className="main">
        {briefs.isPending ? (
          <Skeleton lines={6} label="Loading the brief" />
        ) : briefs.isError ? (
          <ErrorState error={briefs.error} onRetry={() => void briefs.refetch()} />
        ) : !brief ? (
          <EmptyState title="No briefs yet">Run cndl run and cndl export-web to produce the decision briefs.</EmptyState>
        ) : (
          <BriefView key={brief.id} brief={brief} index={index} showTruth={showTruth} />
        )}
      </main>
    </div>
  );
};
