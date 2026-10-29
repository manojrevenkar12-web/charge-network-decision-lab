import { expect, type Page, test } from "@playwright/test";

const shoot = async (page: Page, name: string) => {
  if (!process.env.SCREENSHOTS) return;
  await page.waitForTimeout(900);
  await page.screenshot({ path: `../docs/screenshots/${name}.png`, fullPage: true });
};

const watch = (page: Page) => {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(m.text());
  });
  return problems;
};

test("is served with a strict content security policy", async ({ page }) => {
  const res = await page.goto("/");
  const csp = res?.headers()["content-security-policy"] ?? "";
  expect(csp).toContain("script-src 'self'");
  expect(csp).not.toContain("unsafe-inline");
});

test("every brief opens with its answer, its range and its decision", async ({ page }) => {
  const problems = watch(page);
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Briefs" });
  await expect(nav.getByRole("button")).toHaveCount(5);
  const names = ["price-increase", "night-discount", "competitor", "forecast", "rollout"];
  for (let i = 0; i < 5; i++) {
    await nav.getByRole("button").nth(i).click();
    await expect(page.getByRole("heading", { level: 2 })).toBeVisible();
    await expect(page.locator(".strip svg")).toBeVisible();
    await expect(page.locator(".decision")).toBeVisible();
    await shoot(page, `0${i + 1}-${names[i]}`);
  }
  expect(problems).toEqual([]);
});

test("the simulator's truth appears only when asked for", async ({ page }) => {
  await page.goto("/#price-change");
  await expect(page.locator(".strip svg")).toBeVisible();
  await expect(page.locator("path.truth")).toHaveCount(0);
  await page.getByLabel("Show the simulator's truth").check();
  await expect(page.getByRole("heading", { name: "Checked against the simulator's truth" })).toBeVisible();
  await expect(page.locator(".verdicts li").first()).toContainText("reported range");
  await shoot(page, "06-truth");
});

test("a link to a brief opens that brief", async ({ page }) => {
  await page.goto("/#rollout");
  await expect(page.getByRole("heading", { level: 2 })).toContainText("1,000 high-power points");
  await expect(page.getByRole("img", { name: "Candidate locations and the recommended sites" })).toBeVisible();
});
