import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RangeStrip, type StripRow } from "../src/components/RangeStrip";

const rows: StripRow[] = [
  { label: "Two-way fixed effects", estimate: 0.015, lo: 0.0, hi: 0.03, kind: "usual", truth: -0.08 },
  { label: "Each site's own trend", estimate: -0.09, lo: -0.1, hi: -0.08, kind: "credible", truth: -0.08 },
];
const fmt = (v: number) => `${(v * 100).toFixed(0)}%`;

describe("the range strip", () => {
  it("describes every estimate in words for screen readers", () => {
    render(<RangeStrip title="Effect" rows={rows} format={fmt} showTruth={false} />);
    const desc = screen.getByRole("img").querySelector("desc")?.textContent ?? "";
    expect(desc).toContain("Two-way fixed effects: 2% (0% to 3%)");
    expect(desc).toContain("Each site's own trend: -9% (-10% to -8%)");
  });

  it("marks a threshold outside the credible range as a decision that holds", () => {
    const { container } = render(
      <RangeStrip title="Effect" rows={rows} format={fmt} band={[-0.1, -0.08]} threshold={{ value: -0.16, label: "break-even" }} showTruth={false} />,
    );
    expect(container.querySelector(".threshold--holds")).not.toBeNull();
    expect(container.querySelector(".threshold--flips")).toBeNull();
  });

  it("marks a threshold inside the credible range as a decision that could flip", () => {
    const { container } = render(
      <RangeStrip title="Effect" rows={rows} format={fmt} band={[-0.1, -0.08]} threshold={{ value: -0.09, label: "break-even" }} showTruth={false} />,
    );
    expect(container.querySelector(".threshold--flips")).not.toBeNull();
  });

  it("shows the truth only when asked", () => {
    const { container, rerender } = render(<RangeStrip title="Effect" rows={rows} format={fmt} showTruth={false} />);
    expect(container.querySelectorAll("path.truth")).toHaveLength(0);
    rerender(<RangeStrip title="Effect" rows={rows} format={fmt} showTruth />);
    expect(container.querySelectorAll("path.truth")).toHaveLength(2);
    expect(screen.getByText("the simulator's truth")).toBeInTheDocument();
  });
});
