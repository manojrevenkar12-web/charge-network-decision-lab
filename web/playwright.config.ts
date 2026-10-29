import { defineConfig, devices } from "@playwright/test";

/** End-to-end against the production build, served with the production CSP. */
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  reporter: [["list"]],
  use: { baseURL: "http://127.0.0.1:4180" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1360, height: 900 } }, testIgnore: /phone/ },
    { name: "phone", use: { ...devices["Pixel 7"] }, testMatch: /phone/ },
  ],
  webServer: { command: "npx vite build && npx vite preview --port 4180 --host 127.0.0.1 --strictPort", url: "http://127.0.0.1:4180", reuseExistingServer: true },
});
