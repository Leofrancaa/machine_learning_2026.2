import { test, expect } from "@playwright/test";
import catalog from "../public/data/models.json";

const fmt = (n: number, digits = 3) =>
  n.toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
const latest = catalog.runs.filter((r) => r.scope === "pa").at(-1)!;

test("compara versões e populações usando os relatórios reais", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/#models");
  const panel = page.getByRole("tabpanel", { name: "Modelos e métricas" });
  await expect(panel).toBeVisible();
  await expect(panel.getByTestId("model-auc")).toHaveText(
    fmt(latest.metrics.roc_auc),
  );
  await panel.getByLabel("Métrica no gráfico").selectOption("sensitivity");
  const older = catalog.runs.find(
    (r) => r.scope === "pa" && r.version === "v2",
  )!;
  await panel
    .getByRole("button", {
      name: `V2, Sensibilidade ${fmt(older.metrics.sensitivity * 100, 1)}%`,
      exact: true,
    })
    .click();
  await expect(panel.getByTestId("model-auc")).toHaveText(
    fmt(older.metrics.roc_auc),
  );
  await expect(panel.getByLabel("Inspecionar versão")).toHaveValue("v2");
  await panel.getByRole("button", { name: "AP + PA", exact: true }).click();
  await expect(
    panel.getByText("Uma população diferente muda a comparação."),
  ).toBeVisible();
  const mixed = catalog.runs.find(
    (r) => r.scope === "ap_pa" && r.version === "v2",
  )!;
  await expect(panel.getByTestId("model-auc")).toHaveText(
    fmt(mixed.metrics.roc_auc),
  );
  await panel.getByRole("button", { name: "Somente PA", exact: true }).click();
  await panel.getByLabel("Inspecionar versão").selectOption(latest.version);
  await panel.getByLabel("Métrica no gráfico").selectOption("roc_auc");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("modelos.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("limiares exibem o custo dos falsos alarmes sem mudar a fila", async ({
  page,
}) => {
  await page.goto("/#models");
  const panel = page.getByRole("tabpanel", { name: "Modelos e métricas" });
  await expect(panel.getByTestId("true-positives")).toHaveText(
    fmt(latest.metrics.confusion_matrix.tp, 0),
  );
  const lower = latest.thresholds.find(
    (r) => r.split === "test" && String(r.threshold).startsWith("p80"),
  )!;
  await panel
    .getByLabel("Regra do limiar")
    .selectOption(String(lower.threshold));
  await expect(panel.getByTestId("true-positives")).toHaveText(
    fmt(Math.round(Number(lower.sensitivity) * latest.positives), 0),
  );
  await expect(panel.getByTestId("false-positives")).toHaveText(
    fmt(
      Math.round(
        (1 - Number(lower.specificity)) * (latest.n - latest.positives),
      ),
      0,
    ),
  );
  await panel
    .getByText("Como esses números foram produzidos e onde conferir")
    .click();
  await expect(
    panel.getByText(`reports/${latest.id}/metrics.json`, { exact: true }),
  ).toBeVisible();
  const download = page.waitForEvent("download");
  await panel.getByRole("link", { name: "Baixar dados da comparação" }).click();
  expect((await download).suggestedFilename()).toBe(
    "vae-modelos-relatorios.json",
  );
  await page.getByRole("tab", { name: "Fila de triagem", exact: true }).click();
  await expect(
    page
      .getByRole("tabpanel", { name: "Fila de triagem" })
      .getByText(latest.id, { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("tab", { name: "Modelos e métricas", exact: true })
    .click();
  await expect(panel.getByLabel("Regra do limiar")).toHaveValue(
    String(lower.threshold),
  );
});

test("métodos e ablações distinguem sementes, supervisão e validação", async ({
  page,
}, info) => {
  await page.goto("/#models");
  const panel = page.getByRole("tabpanel", { name: "Modelos e métricas" });
  await panel
    .getByRole("button", { name: "Comparar métodos", exact: true })
    .click();
  const table = panel.getByRole("region", { name: "Comparação de métodos" });
  const vae = latest.methods.find((r) => r.method === "VAE")!;
  await expect(
    table
      .getByRole("row")
      .filter({
        has: page.getByRole("rowheader", { name: "VAE", exact: true }),
      }),
  ).toContainText(fmt(Number(vae.auc_test)));
  await expect(
    table.getByRole("rowheader", { name: "CNN supervisionada", exact: true }),
  ).toBeVisible();
  await panel
    .getByRole("button", { name: "Ablações na validação", exact: true })
    .click();
  await expect(
    panel.getByText("Regra fixada antes dos resultados", { exact: true }),
  ).toBeVisible();
  const ablations = panel.getByRole("region", {
    name: "Resultados das ablações",
  });
  await expect(
    ablations.getByRole("row").filter({ hasText: "A · Recorte de bordas" }),
  ).toContainText("Rejeitada");
  await expect(
    ablations.getByRole("row").filter({ hasText: "B + C1 · Combinação" }),
  ).toContainText("Aceita");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("ablacoes.png"),
    fullPage: true,
  });
});
