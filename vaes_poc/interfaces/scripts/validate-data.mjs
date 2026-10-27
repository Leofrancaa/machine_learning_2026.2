import { readFile, access } from "node:fs/promises";
import path from "node:path";
import { inspectDemoData } from "./data-contract.mjs";

try {
  const file = path.resolve(process.argv[2] ?? "public/data/data.json");
  const data = JSON.parse(await readFile(file, "utf8"));
  const result = inspectDemoData(data);
  await Promise.all(
    result.assets.map(async (asset) => {
      try {
        await access(path.resolve("public", `.${asset}`));
      } catch {
        result.errors.push(`Arquivo ausente: ${asset}`);
      }
    }),
  );
  if (result.errors.length) {
    console.error(
      `Exportação incompatível (${result.errors.length} problema(s)):\n${result.errors.map((error) => `- ${error}`).join("\n")}`,
    );
    process.exitCode = 1;
  } else {
    console.log(
      `Exportação válida: ${data.generated} · ${result.assets.length} arquivos conferidos.`,
    );
    for (const run of Object.values(data.runs))
      console.log(
        `  ${run.model.run}: ${run.queue.length} exames · limiar ${run.model.threshold} · latente ${run.model.latent_dim}D`,
      );
  }
} catch (error) {
  console.error(`Não foi possível validar a exportação: ${error.message}`);
  process.exitCode = 1;
}
