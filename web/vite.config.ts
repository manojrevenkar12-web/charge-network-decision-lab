import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
import { CSP } from "./csp";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5180, host: "127.0.0.1" },
  preview: { port: 4180, host: "127.0.0.1", headers: { "Content-Security-Policy": CSP } },
  test: { environment: "jsdom", include: ["test/**/*.test.tsx", "test/**/*.test.ts"], setupFiles: ["test/setup.ts"] },
});
