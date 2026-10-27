import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Os CSVs incluem campos entre aspas com vírgulas (por exemplo, vários σ).
export function parseCsv(text) {
  const rows = [];
  let row = [],
    field = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (c === "," && !quoted) {
      row.push(field);
      field = "";
    } else if (c === "\n" && !quoted) {
      row.push(field.replace(/\r$/, ""));
      if (row.some(Boolean)) rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  if (quoted) throw new Error("CSV com aspas não fechadas");
  const [headers, ...values] = rows;
  return values.map((values) =>
    Object.fromEntries(
      headers.map((h, i) => {
        const v = values[i] ?? "";
        return [
          h,
          v === ""
            ? null
            : /^(True|False)$/i.test(v)
              ? v.toLowerCase() === "true"
              : Number.isFinite(Number(v))
                ? Number(v)
                : v,
        ];
      }),
    ),
  );
}

export async function buildCatalog(root) {
  const json = async (file) =>
    JSON.parse(await readFile(path.join(root, file), "utf8"));
  const csv = async (file) =>
    parseCsv(await readFile(path.join(root, file), "utf8"));
  const names = (await readdir(path.join(root, "reports"))).filter((name) =>
    /^rsna_(?:pa|ap_pa)(?:_v\d+)?$/.test(name),
  );
  const runs = await Promise.all(
    names.map(async (id) => {
      const metadata = await json(`models/${id}/metadata.json`);
      const metrics = await json(`reports/${id}/metrics.json`);
      const cm = metrics.confusion_matrix;
      if (
        !cm ||
        !Number.isFinite(metrics.roc_auc) ||
        !Number.isFinite(metadata.sigma)
      )
        throw new Error(`Relatório incompleto: ${id}`);
      if (Math.abs(metrics.threshold - metadata.threshold) > 1e-10)
        throw new Error(`Limiar divergente entre relatório e modelo: ${id}`);
      const version = metadata.version || "v1";
      return {
        id,
        version,
        scope: metadata.views.length > 1 ? "ap_pa" : "pa",
        metadata,
        metrics,
        n: cm.tn + cm.fp + cm.fn + cm.tp,
        positives: cm.fn + cm.tp,
        thresholds: await csv(`reports/${id}/thresholds.csv`),
        methods: await csv(`reports/${id}/method_comparison.csv`),
        sources: [
          `models/${id}/metadata.json`,
          `reports/${id}/metrics.json`,
          `reports/${id}/thresholds.csv`,
          `reports/${id}/method_comparison.csv`,
        ],
      };
    }),
  );
  runs.sort(
    (a, b) =>
      a.scope.localeCompare(b.scope) ||
      Number(a.version.slice(1)) - Number(b.version.slice(1)),
  );
  const ablationSource = "reports/rsna_pa_ablations_v2/ablations.csv";
  return {
    schema: 1,
    runs,
    ablations: await csv(ablationSource),
    ablationSource,
  };
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const interfaces = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "..",
  );
  const catalog = await buildCatalog(path.resolve(interfaces, ".."));
  await writeFile(
    path.join(interfaces, "public/data/models.json"),
    JSON.stringify(catalog, null, 2) + "\n",
  );
  console.log(
    `Comparação atualizada: ${catalog.runs.length} modelos, ${catalog.ablations.length} configurações de ablação. Fontes: reports/ e models/.`,
  );
}
