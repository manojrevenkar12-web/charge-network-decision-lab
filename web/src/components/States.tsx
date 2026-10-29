import { LoadError } from "../lib/data";

export const Skeleton = ({ lines = 4, label = "Loading" }: { lines?: number; label?: string }) => (
  <div role="status" aria-label={label} className="brief">
    {Array.from({ length: lines }, (_, i) => (
      <span
        // biome-ignore lint/suspicious/noArrayIndexKey: placeholder lines have no identity
        key={i}
        className="skeleton"
        style={{ height: i === 0 ? "var(--s-7)" : "var(--s-4)", width: i === lines - 1 ? "60%" : "100%", marginTop: i ? "var(--s-3)" : 0 }}
      />
    ))}
  </div>
);

export const ErrorState = ({ error, onRetry }: { error: unknown; onRetry: () => void }) => (
  <div role="alert" className="state state--error">
    <h2>The results could not be loaded</h2>
    <p>
      {error instanceof LoadError
        ? `${error.message}. Run cndl run and cndl export-web to produce them.`
        : error instanceof Error
          ? error.message
          : String(error)}
    </p>
    <button type="button" className="btn" onClick={onRetry}>
      Try again
    </button>
  </div>
);

export const EmptyState = ({ title, children }: { title: string; children: string }) => (
  <div className="state">
    <h2>{title}</h2>
    <p>{children}</p>
  </div>
);
