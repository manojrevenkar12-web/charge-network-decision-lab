import { expect, test } from "@playwright/test";

test("reads on a phone without sideways scrolling", async ({ page }) => {
  await page.goto("/#price-change");
  await expect(page.locator(".strip svg")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  if (process.env.SCREENSHOTS) await page.screenshot({ path: "../docs/screenshots/07-phone.png", fullPage: true });
});
