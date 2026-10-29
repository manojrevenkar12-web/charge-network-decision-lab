import { useEffect, useRef, useState } from "react";

/** The container's width in CSS pixels, so SVG text keeps its size at any width. */
export const useWidth = (fallback = 640, min = 300) => {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => {
      if (e) setW(Math.max(min, Math.round(e.contentRect.width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return { ref, w };
};
