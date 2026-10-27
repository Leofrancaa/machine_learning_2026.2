import { expect, test } from "@playwright/test";
import rawData from "../public/data/data.json";
import type { VaeData } from "../lib/types";
const data = rawData as unknown as VaeData;
const run = data.runs[data.main];

test("laudos revelados explicam acertos e erros sem alterar a ordenação", async ({
  page,
  isMobile,
}) => {
  await page.goto("/#queue");
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  await expect(panel.locator(".decision-reading")).toHaveCount(0);
  const labels = panel.getByRole("checkbox", { name: "Laudos", exact: true });
  await labels.check();
  const cases = new Map(
    run.queue.map((exam) => [`${exam.label}:${exam.flagged}`, exam]),
  );
  for (const exam of cases.values()) {
    const position = run.queue.findIndex((item) => item.id === exam.id) + 1;
    await panel
      .getByRole("region", { name: "Comparação das filas" })
      .getByRole("button", {
        name: `Ordem de chegada: ${exam.id}, posição ${position}`,
        exact: true,
      })
      .click();
    const expected =
      exam.label === "opacidade"
        ? exam.flagged
          ? "Opacidade sinalizada"
          : "Falso negativo neste limiar"
        : exam.flagged
          ? "Falso alerta neste limiar"
          : "Normal abaixo do limiar";
    if (isMobile) {
      await expect(
        page.getByRole("dialog").locator(".decision-reading strong"),
      ).toHaveText(expected);
      await page.getByRole("button", { name: "Fechar análise" }).click();
    }
    await expect(panel.locator(".viewer .decision-reading strong")).toHaveText(
      expected,
    );
  }
  const explanation = panel.locator(".ranking-reading");
  if (await explanation.count()) {
    await explanation.locator("summary").click();
    await expect(explanation).toContainText(
      "Alterar somente o limiar não muda a ordem.",
    );
  }
  await labels.uncheck();
  await expect(panel.locator(".decision-reading")).toHaveCount(0);
});

test("bancada operacional alinha lista e visualizador e mantém rolagem interna", async ({
  page,
  isMobile,
}, testInfo) => {
  await page.goto("/#queue");
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  const worklist = panel.locator(".worklist");
  const viewer = panel.locator(".viewer");
  if (isMobile) {
    const listBox = await worklist.boundingBox();
    const viewerBox = await viewer.boundingBox();
    expect(viewerBox!.y).toBeGreaterThanOrEqual(listBox!.y + listBox!.height);
    return;
  }
  for (const width of [1440, 1920]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect
      .poll(async () => {
        const listBox = await worklist.boundingBox();
        const viewerBox = await viewer.boundingBox();
        return (
          Math.abs(listBox!.y - viewerBox!.y) +
          Math.abs(listBox!.height - viewerBox!.height)
        );
      })
      .toBeLessThan(2);
    const rows = worklist.locator(".rows");
    if (run.queue.length >= 16)
      expect(
        await rows.evaluate(
          (element) => element.scrollHeight > element.clientHeight,
        ),
      ).toBe(true);
  }
  const last = run.queue[run.queue.length - 1];
  await panel
    .getByRole("region", { name: "Comparação das filas" })
    .getByRole("button", {
      name: `Ordem de chegada: ${last.id}, posição ${run.queue.length}`,
      exact: true,
    })
    .click();
  await expect
    .poll(() =>
      worklist.locator(".rows").evaluate((list) => {
        const selected = list
          .querySelector(".selected")
          ?.getBoundingClientRect();
        const bounds = list.getBoundingClientRect();
        return (
          !!selected &&
          selected.top >= bounds.top - 1 &&
          selected.bottom <= bounds.bottom + 1
        );
      }),
    )
    .toBe(true);
  await worklist.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: testInfo.outputPath("bancada-operacional.png"),
    fullPage: true,
  });
});

test("velocidade, passo manual e acompanhamento usam as chegadas reais", async ({
  page,
  isMobile,
}) => {
  test.skip(run.queue.length < 3, "Este cenário precisa de três exames.");
  await page.goto("/#queue");
  await page.clock.install();
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  await panel
    .getByRole("combobox", { name: "Velocidade da simulação" })
    .selectOption("1300");
  await panel
    .getByRole("button", { name: "Simular chegada", exact: true })
    .click();
  await page.clock.runFor(1300);
  await expect(panel.locator(".rows > li")).toHaveCount(1);
  await expect(panel.locator(".arrival-narration strong")).toHaveText(
    run.queue[0].id,
  );
  await page.clock.runFor(1300);
  await expect(panel.locator(".viewer h2")).toContainText(run.queue[1].id);
  await panel
    .getByRole("button", { name: "Pausar simulação", exact: true })
    .click();
  await panel.getByRole("button", { name: "Adicionar próximo exame" }).click();
  await expect(panel.locator(".rows > li")).toHaveCount(3);
  await expect(panel.locator(".viewer h2")).toContainText(run.queue[2].id);
  await expect(
    panel.getByRole("progressbar", { name: "Progresso da chegada dos exames" }),
  ).toHaveAttribute("value", "3");
  const comparison = panel.getByRole("region", {
    name: "Comparação das filas",
  });
  await expect(comparison.locator(".latest-arrival")).toHaveCount(2);
  await page.clock.runFor(30);
  await expect(comparison.locator(".queue-connection > path")).toHaveCount(1);
  await expect(
    comparison.locator(".queue-connection > path"),
  ).not.toHaveAttribute("d", /NaN|Infinity/);
  const first = comparison.getByRole("button", {
    name: `Ordem de chegada: ${run.queue[0].id}, posição 1`,
    exact: true,
  });
  await first.focus();
  await first.press("Enter");
  if (isMobile) {
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.getByRole("button", { name: "Fechar análise" }).click();
  }
  await expect(
    panel.getByRole("checkbox", { name: "Seguir novas chegadas" }),
  ).not.toBeChecked();
});

test("modo apresentação preserva exame e mantém a simulação utilizável", async ({
  page,
}, testInfo) => {
  await page.goto("/#queue");
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  await panel.getByRole("button", { name: "Modo apresentação" }).click();
  await expect(panel.locator(".worklist")).toBeHidden();
  await expect(
    panel.getByRole("region", { name: "Comparação das filas" }),
  ).toBeVisible();
  await expect(
    panel.getByRole("region", { name: "Visualizador do exame selecionado" }),
  ).toBeVisible();
  await panel.getByRole("checkbox", { name: "Laudos", exact: true }).check();
  await expect(
    panel.getByRole("checkbox", { name: "Caixas do laudo" }),
  ).toBeEnabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: testInfo.outputPath("apresentacao.png"),
    fullPage: true,
  });
  await panel.getByRole("button", { name: "Voltar à exploração" }).click();
  await expect(panel.locator(".rows > li")).toHaveCount(run.queue.length);
  await expect(panel.locator(".run-provenance")).toContainText(run.model.run);
  await expect(panel.locator(".run-provenance")).toContainText(
    data.generated.split("-").reverse().join("/"),
  );
});
