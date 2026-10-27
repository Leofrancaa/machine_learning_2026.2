import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
import { inspectDemoData } from "./data-contract.mjs";

const baseline = JSON.parse(
  readFileSync(new URL("../public/data/data.json", import.meta.url), "utf8"),
);

test("exportação atual mantém dados, decisões e sprites consistentes", () => {
  assert.deepEqual(inspectDemoData(baseline).errors, []);
});
test("aceita nova versão, contagem, limiar, score e dimensão latente", () => {
  const updated = structuredClone(baseline);
  const run = updated.runs[updated.main];
  run.model.run = "experimento_futuro";
  run.model.threshold = 0.5;
  run.model.latent_dim = 3;
  run.model.image_size = 256;
  run.queue = run.queue
    .slice(0, 3)
    .map((exam, i) => ({ ...exam, score: i / 2, flagged: i / 2 >= 0.5 }));
  updated.latent.dims = Array.from(
    { length: 3 },
    (_, i) => `/data/fixture/latent${i}.png`,
  );
  assert.deepEqual(inspectDemoData(updated).errors, []);
});
test("detecta alerta calculado com limiar de outra execução", () => {
  const updated = structuredClone(baseline);
  updated.runs[updated.main].queue[0].flagged =
    !updated.runs[updated.main].queue[0].flagged;
  assert.ok(
    inspectDemoData(updated).errors.some((error) =>
      error.includes("alerta não corresponde"),
    ),
  );
});
test("detecta sprites incompatíveis com um latente alterado", () => {
  const updated = structuredClone(baseline);
  updated.runs[updated.main].model.latent_dim += 1;
  assert.ok(
    inspectDemoData(updated).errors.some((error) =>
      error.startsWith("latent.dims"),
    ),
  );
});
test("rejeita métricas fora de escala, códigos repetidos e scores não finitos", () => {
  const updated = structuredClone(baseline);
  const run = updated.runs[updated.main];
  if (run.model.test) run.model.test.sensitivity = 93;
  run.queue[1].id = run.queue[0].id;
  run.queue[0].score = NaN;
  const errors = inspectDemoData(updated).errors;
  assert.ok(errors.some((error) => error.includes("duplicados")));
  assert.ok(errors.some((error) => error.includes("score deve ser finito")));
  if (run.model.test)
    assert.ok(errors.some((error) => error.includes("sensitivity")));
});
test("estrutura ausente gera diagnóstico em vez de falhar na navegação", () => {
  assert.ok(inspectDemoData({}).errors.length > 0);
  assert.ok(inspectDemoData(null).errors.length > 0);
});
