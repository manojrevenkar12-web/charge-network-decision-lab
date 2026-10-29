import "@fontsource-variable/instrument-sans/wdth.css";
import "./theme/tokens.css";
import "./theme/app.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MotionConfig } from "motion/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });
const el = document.getElementById("root");
if (el)
  createRoot(el).render(
    <StrictMode>
      <QueryClientProvider client={client}>
        <MotionConfig reducedMotion="user">
          <App />
        </MotionConfig>
      </QueryClientProvider>
    </StrictMode>,
  );
