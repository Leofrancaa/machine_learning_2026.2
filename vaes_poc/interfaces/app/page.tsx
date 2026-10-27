import { readFile } from "node:fs/promises";
import path from "node:path";

import Demo from "@/components/Demo";
import type { VaeData } from "@/lib/types";
import type { ModelCatalog } from "@/lib/models";

// Os dados são lidos no build: a página sai pronta, sem nenhuma chamada ao servidor.
async function loadData(): Promise<VaeData | null> {
  try {
    const file = path.join(process.cwd(), "public", "data", "data.json");
    return JSON.parse(await readFile(file, "utf-8"));
  } catch {
    return null;
  }
}

export default async function Page() {
  const data = await loadData();
  if (!data) {
    return (
      <main className="missing">
        <h1>Sem dados da demo</h1>
        <p>
          Rode <code>make export</code> na raiz do repositório para gerar{" "}
          <code>interfaces/public/data</code>.
        </p>
      </main>
    );
  }
  const catalog: ModelCatalog = JSON.parse(
    await readFile(path.join(process.cwd(), "public/data/models.json"), "utf8"),
  );
  // O export já existente pode ser anterior aos novos campos de metadados.
  // Só completar campos descritivos; scores e resultados da demo continuam intactos.
  for (const run of Object.values(data.runs)) {
    const report = catalog.runs.find((r) => r.id === run.model.run);
    if (!report) continue;
    if (Math.abs(report.metadata.threshold - run.model.threshold) > 1e-10)
      throw new Error(
        `Exportação e relatório divergentes: ${run.model.run}. Rode make export.`,
      );
    const {
      image_size,
      epochs,
      version,
      threshold_rule,
      normalize,
      lung_mask,
    } = report.metadata;
    Object.assign(run.model, {
      image_size,
      epochs,
      version: version ?? report.version,
      threshold_rule,
      normalize: normalize ?? false,
      lung_mask: lung_mask ?? false,
    });
  }
  return <Demo data={data} catalog={catalog} />;
}
