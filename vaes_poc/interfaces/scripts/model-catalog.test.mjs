import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { buildCatalog, parseCsv } from "./model-catalog.mjs";

test("CSV preserva vírgulas, aspas e métricas ausentes", () => {
  assert.deepEqual(
    parseCsv(
      'method,sigma,auc_test,uses_labels\r\n"PCA, 8","4, 6",0.8,False\r\n"Um ""modelo""",,0.7,True\r\n',
    ),
    [
      { method: "PCA, 8", sigma: "4, 6", auc_test: 0.8, uses_labels: false },
      { method: 'Um "modelo"', sigma: null, auc_test: 0.7, uses_labels: true },
    ],
  );
});

test("catálogo reproduz relatórios e permanece coerente com a demo", async () => {
  // fileURLToPath também trata letras de unidade no Windows.
  const { fileURLToPath } = await import("node:url");
  const catalog = await buildCatalog(
    fileURLToPath(new URL("../../", import.meta.url)),
  );
  assert.ok(catalog.runs.length >= 2);
  for (const run of catalog.runs) {
    const metrics = JSON.parse(
      await readFile(
        new URL(`../../reports/${run.id}/metrics.json`, import.meta.url),
        "utf8",
      ),
    );
    assert.deepEqual(run.metrics, metrics);
    assert.equal(
      run.n,
      Object.values(metrics.confusion_matrix).reduce((a, b) => a + b, 0),
    );
    const p95 = run.thresholds.find(
      (r) => r.split === "test" && r.threshold.startsWith("p95"),
    );
    assert.equal(p95.sensitivity, metrics.sensitivity);
    assert.equal(p95.specificity, metrics.specificity);
    assert.ok(run.methods.some((m) => m.method === "VAE"));
  }
  const demo = JSON.parse(
    await readFile(
      new URL("../public/data/data.json", import.meta.url),
      "utf8",
    ),
  );
  for (const { model } of Object.values(demo.runs)) {
    const report = catalog.runs.find((r) => r.id === model.run);
    assert.ok(
      report,
      `A demo deve referenciar um relatório conhecido: ${model.run}`,
    );
    assert.equal(report.metadata.threshold, model.threshold);
    assert.equal(report.metadata.sigma, model.sigma);
    assert.equal(report.n, model.n_test);
  }
  assert.ok(catalog.ablations.some((a) => a.accepted === false));
});
