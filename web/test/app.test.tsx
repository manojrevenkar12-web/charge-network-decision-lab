import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/App";

const wrap = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <App />
    </QueryClientProvider>,
  );

const brief = (id: string, title: string) => ({
  id,
  title,
  question: `${title}?`,
  answer: `Answer to ${title}.`,
  decision: "Keep it.",
  tone: "holds",
  number: { label: "label", value: 0, low: 0, high: 0, unit: "pct" },
  rests_on: ["a reason"],
  would_change: ["a change"],
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.location.hash = "";
});

describe("the decision room", () => {
  it("says what to do when the results are missing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 404 })));
    wrap();
    expect(await screen.findByRole("alert")).toHaveTextContent("briefs could not be loaded (404)");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("shows an empty state when there are no briefs", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("[]", { status: 200 })));
    wrap();
    expect(await screen.findByText("No briefs yet")).toBeInTheDocument();
  });

  it("lists the questions and opens the one chosen", async () => {
    const briefs = [brief("forecast", "Energy next year"), brief("competitor", "A competitor next door")];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.includes("briefs") ? new Response(JSON.stringify(briefs)) : new Response(JSON.stringify({ available: false })),
      ),
    );
    wrap();
    expect(await screen.findByRole("heading", { level: 2, name: "Energy next year?" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /A competitor next door/ }));
    expect(await screen.findByRole("heading", { level: 2, name: "A competitor next door?" })).toBeInTheDocument();
    expect(screen.getByText("Decision, robust to the uncertainty")).toBeInTheDocument();
  });

  it("explains a result file it cannot read instead of breaking", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => (url.includes("briefs") ? new Response(JSON.stringify([brief("forecast", "Energy")])) : new Response("{}"))),
    );
    wrap();
    expect(await screen.findByRole("alert")).toHaveTextContent("do not have the expected shape");
  });
});
