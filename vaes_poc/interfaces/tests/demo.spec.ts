import { expect, test } from "@playwright/test";
import rawData from "../public/data/data.json";
import type { VaeData } from "../lib/types";
const data = rawData as unknown as VaeData;

test.beforeEach(async ({ page }) => {
  await page.goto("/#queue");
  await expect(
    page.getByRole("tab", { name: "Fila de triagem" }),
  ).toHaveAttribute("aria-selected", "true");
});

test("busca, filtro e seleção mantêm o visualizador consistente", async ({
  page,
  isMobile,
}) => {
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  const rows = panel.locator(".rows > li");
  await expect(rows).toHaveCount(data.runs[data.main].queue.length);
  await panel.getByRole("button", { name: /^Alertas/ }).click();
  await expect(rows).toHaveCount(
    data.runs[data.main].queue.filter((exam) => exam.flagged).length,
  );
  const search = page.getByRole("searchbox", {
    name: "Buscar exame pelo código",
  });
  await search.fill("INEXISTENTE");
  await expect(rows).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Visualizador de exames" }),
  ).toBeVisible();
  await search.fill("");
  await panel.getByRole("button", { name: "Todos", exact: true }).click();
  const second = rows.nth(1).getByRole("button");
  const code = await rows.nth(1).locator(".exam > span").innerText();
  await second.focus();
  await page.keyboard.press("Enter");
  if (isMobile) {
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByRole("button", { name: "Fechar análise" }).click();
  }
  await expect(panel.locator(".viewer h2")).toContainText(code);
  await expect(second).toHaveAttribute("aria-pressed", "true");
});

test("simulação pausa, retoma e restaura sem mostrar exames futuros", async ({
  page,
}) => {
  await page.clock.install();
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  const rows = panel.locator(".rows > li");
  await panel.getByRole("button", { name: "Simular chegada" }).click();
  await expect(rows).toHaveCount(0);
  await expect(panel.locator(".viewer-empty")).toBeVisible();
  await page.clock.runFor(1300);
  await expect(rows).toHaveCount(2);
  await panel.getByRole("button", { name: "Pausar simulação" }).click();
  await page.clock.runFor(2000);
  await expect(rows).toHaveCount(2);
  await panel.getByRole("button", { name: "Continuar simulação" }).click();
  await page.clock.runFor(650);
  await expect(rows).toHaveCount(3);
  await panel.getByRole("button", { name: "Restaurar fila" }).click();
  await expect(rows).toHaveCount(data.runs[data.main].queue.length);
  await expect(
    panel.getByRole("button", { name: "Simular chegada" }),
  ).toBeVisible();
});

test("troca de modelo e laudos controlam as anotações", async ({ page }) => {
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  const boxes = panel.getByRole("checkbox", { name: "Caixas do laudo" });
  await expect(boxes).toBeDisabled();
  await panel.getByRole("checkbox", { name: "Laudos", exact: true }).check();
  await boxes.check();
  await expect(boxes).toBeChecked();
  await panel.getByRole("checkbox", { name: "Laudos", exact: true }).uncheck();
  await expect(boxes).not.toBeChecked();
  await expect(boxes).toBeDisabled();
  if (!data.runs.ap_pa) {
    await expect(
      panel.getByRole("button", { name: "AP + PA", exact: true }),
    ).toBeDisabled();
    return;
  }
  await panel.getByRole("button", { name: "AP + PA", exact: true }).click();
  await expect(panel.locator(".rows > li")).toHaveCount(
    data.runs.ap_pa.queue.length,
  );
  await expect(
    panel.getByText("Atalho da incidência.", { exact: true }),
  ).toBeVisible();
  await expect(panel.locator(".viewer h2")).toContainText("EX-");
});

test("abas suportam teclado, histórico e preservam o desafio", async ({
  page,
}) => {
  const queueTab = page.getByRole("tab", { name: "Fila de triagem" });
  await queueTab.focus();
  await page.keyboard.press("ArrowDown");
  await expect(page).toHaveURL(/#inside$/);
  await expect(
    page.getByRole("tabpanel", { name: "Por dentro do VAE" }),
  ).toBeVisible();
  await page.keyboard.press("End");
  const game = page.getByRole("tabpanel", { name: "Você × VAE" });
  await expect(game).toBeVisible();
  await game.locator(".game-card").first().click();
  await queueTab.click();
  await page.goBack();
  await expect(game).toBeVisible();
  await expect(game.locator(".game-card").first()).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  for (let index = 1; index < data.game.length; index++)
    await game.locator(".game-card").nth(index).click();
  await game.getByRole("button", { name: "Revelar", exact: true }).click();
  await expect(
    game.getByRole("heading", { name: /Opacidades nas/ }),
  ).toBeVisible();
  await game.getByRole("button", { name: "Limpar", exact: true }).click();
  await expect(
    game.getByRole("button", { name: "Revelar", exact: true }),
  ).toBeDisabled();
});

test("layout cabe na tela e os recursos visuais funcionam", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  await panel.getByRole("button", { name: "Comparar", exact: true }).click();
  await expect(
    panel.getByRole("slider", { name: "Comparar original e reconstrução" }),
  ).toBeVisible();
  await panel
    .getByRole("checkbox", { name: "Mapa de calor", exact: true })
    .check();
  await expect(
    panel.getByRole("slider", { name: "Opacidade do mapa" }),
  ).toBeEnabled();
  await panel
    .getByRole("button", { name: "Próximo exame", exact: true })
    .click();
  await expect(panel.locator(".viewer-navigation")).toContainText(
    `2/${data.runs[data.main].queue.length}`,
  );
  for (const name of ["Fila de triagem", "Por dentro do VAE", "Você × VAE"]) {
    await page.getByRole("tab", { name, exact: true }).click();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  }
  await page.getByRole("tab", { name: "Fila de triagem", exact: true }).click();
  await panel.getByRole("button", { name: "Original", exact: true }).click();
  await panel
    .getByRole("checkbox", { name: "Mapa de calor", exact: true })
    .uncheck();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: testInfo.outputPath("workspace.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
