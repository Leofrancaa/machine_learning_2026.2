import { expect, test } from "@playwright/test";
import rawData from "../public/data/data.json";
import type { VaeData } from "../lib/types";
const data = rawData as unknown as VaeData;

test("percurso conecta arquitetura, amostragem e resultados reais", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const lesson = page.getByRole("tabpanel", { name: "Entenda o VAE" });
  await expect(lesson).toBeVisible();
  await expect(
    lesson.getByRole("heading", { name: "Por que aprender o normal?" }),
  ).toBeVisible();
  await lesson.getByRole("button", { name: "Próxima etapa" }).click();
  await expect(
    lesson.getByRole("heading", { name: "Da imagem ao código e de volta" }),
  ).toBeVisible();
  await expect(
    lesson.getByText(
      `${data.runs[data.main].model.latent_dim} médias + ${data.runs[data.main].model.latent_dim} log-variâncias`,
      { exact: true },
    ),
  ).toBeVisible();
  await lesson.getByRole("button", { name: "Próxima etapa" }).click();
  const equation = lesson.locator(".sample-equation strong");
  const before = await equation.innerText();
  const mean = lesson.getByRole("slider", { name: "Média μ" });
  await mean.focus();
  await page.keyboard.press("ArrowRight");
  await expect(equation).not.toHaveText(before);
  await expect(
    lesson.getByText(/Os valores deste gráfico são didáticos/),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("amostragem.png"),
    fullPage: true,
  });
  await lesson.getByRole("button", { name: "Próxima etapa" }).click();
  const sample = [...data.runs[data.main].queue].sort(
    (a, b) => b.score - a.score,
  )[0];
  await expect(
    lesson.getByRole("img", {
      name: `Radiografia original ${sample.id}`,
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    lesson.getByRole("img", {
      name: `Reconstrução pelo VAE de ${sample.id}`,
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    lesson.getByRole("img", {
      name: `Mapa de diferenças de ${sample.id}`,
      exact: true,
    }),
  ).toBeVisible();
  await lesson.getByRole("button", { name: "Próxima etapa" }).click();
  await expect(
    lesson.getByRole("region", { name: "Comparação das filas" }),
  ).toBeVisible();
  await lesson
    .getByRole("button", { name: "Explorar a fila", exact: true })
    .click();
  await expect(page).toHaveURL(/#queue$/);
  await expect(
    page.getByRole("tabpanel", { name: "Fila de triagem" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("comparação acompanha a posição real e limpa filtros incompatíveis", async ({
  page,
  isMobile,
}) => {
  await page.goto("/#queue");
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  const comparison = panel.getByRole("region", {
    name: "Comparação das filas",
  });
  const exam = data.runs[data.main].queue.find((item) => !item.flagged)!;
  const arrival =
    data.runs[data.main].queue.findIndex((item) => item.id === exam.id) + 1;
  const ranked = [...data.runs[data.main].queue].sort(
    (a, b) => b.score - a.score,
  );
  const position = ranked.findIndex((item) => item.id === exam.id) + 1;
  await panel.getByRole("button", { name: /^Alertas/ }).click();
  await comparison
    .getByRole("button", {
      name: `Ordem de chegada: ${exam.id}, posição ${arrival}`,
      exact: true,
    })
    .click();
  if (isMobile)
    await page.getByRole("button", { name: "Fechar análise" }).click();
  await expect(panel.locator(".rows > li")).toHaveCount(
    data.runs[data.main].queue.length,
  );
  await expect(comparison.locator(".movement-explanation")).toContainText(
    `${exam.id}: posição ${arrival} na chegada → posição ${position} pelo VAE.`,
  );
  await expect(
    comparison.getByRole("button", {
      name: `Ordem pelo VAE: ${exam.id}, posição ${position}`,
      exact: true,
    }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(panel.locator(".viewer h2")).toContainText(exam.id);
});

test("análise ampliada permite comparar, navegar, ampliar e fechar com foco restaurado", async ({
  page,
}, testInfo) => {
  await page.goto("/#queue");
  const panel = page.getByRole("tabpanel", { name: "Fila de triagem" });
  const trigger = panel.getByRole("button", { name: "Analisar lado a lado" });
  await trigger.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".exam-triptych figure")).toHaveCount(3);
  const first = [...data.runs[data.main].queue].sort(
    (a, b) => b.score - a.score,
  )[0];
  await expect(dialog.getByRole("heading", { level: 2 })).toContainText(
    first.id,
  );
  await page
    .getByRole("button", { name: "Fechar análise" })
    .press("ArrowRight");
  await expect(dialog.locator(".inspection-navigation")).toContainText(
    `2 / ${data.runs[data.main].queue.length}`,
  );
  await dialog
    .getByRole("button", { name: "Imagem ampliada", exact: true })
    .click();
  await dialog.getByRole("button", { name: "Aumentar zoom" }).click();
  await expect(dialog.getByLabel("Nível de zoom")).toHaveText("125%");
  const title = await dialog.getByRole("heading", { level: 2 }).innerText();
  const image = dialog.getByRole("region", {
    name: "Imagem com zoom, use as barras de rolagem para explorar",
  });
  await image.focus();
  await image.press("ArrowRight");
  await expect(dialog.getByRole("heading", { level: 2 })).toHaveText(title, {
    useInnerText: true,
  });
  await dialog.getByRole("button", { name: "Ajustar", exact: true }).click();
  await expect(dialog.getByLabel("Nível de zoom")).toHaveText("100%");
  await dialog
    .getByRole("button", { name: "Comparação explicada", exact: true })
    .click();
  await page.screenshot({
    path: testInfo.outputPath("analise.png"),
    fullPage: true,
  });
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  expect(await page.evaluate(() => document.body.style.overflow)).not.toBe(
    "hidden",
  );
});

test("layout permanece utilizável em todas as etapas", async ({
  page,
}, testInfo) => {
  await page.goto("/#learn");
  const lesson = page.getByRole("tabpanel", { name: "Entenda o VAE" });
  const steps = lesson
    .getByRole("navigation", { name: "Etapas da explicação" })
    .getByRole("button");
  for (let index = 0; index < 5; index++) {
    await steps.nth(index).click();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  }
  await steps.first().click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: testInfo.outputPath("percurso.png"),
    fullPage: true,
  });
  if (testInfo.project.name === "mobile") {
    await page.setViewportSize({ width: 320, height: 740 });
    for (let index = 0; index < 5; index++) {
      await steps.nth(index).click();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBe(true);
    }
  }
});
