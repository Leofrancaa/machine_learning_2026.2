/** Valida o contrato da exportação sem fixar scores, contagens ou versão do modelo. */
export function inspectDemoData(data) {
  const errors = [];
  const assets = new Set();
  const object = (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value);
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const check = (condition, path, message) => {
    if (!condition) errors.push(`${path}: ${message}`);
  };
  const probability = (value, path) =>
    check(
      finite(value) && value >= 0 && value <= 1,
      path,
      "deve estar entre 0 e 1",
    );
  const positiveInteger = (value, path) =>
    check(
      Number.isInteger(value) && value > 0,
      path,
      "deve ser inteiro positivo",
    );
  function asset(value, path) {
    if (
      typeof value !== "string" ||
      !value.startsWith("/data/") ||
      value.includes("..") ||
      value.includes("\\")
    )
      return check(false, path, "deve apontar para um arquivo local em /data/");
    assets.add(value);
  }
  function exam(value, path, threshold) {
    if (!object(value)) return check(false, path, "exame inválido");
    check(
      typeof value.id === "string" && value.id.length > 0,
      `${path}.id`,
      "código obrigatório",
    );
    check(
      ["normal", "opacidade"].includes(value.label),
      `${path}.label`,
      "rótulo desconhecido",
    );
    check(
      typeof value.view === "string" && value.view.length > 0,
      `${path}.view`,
      "incidência obrigatória",
    );
    check(
      finite(value.score) && value.score >= 0,
      `${path}.score`,
      "score deve ser finito e não negativo",
    );
    probability(value.percentile, `${path}.percentile`);
    check(
      typeof value.flagged === "boolean",
      `${path}.flagged`,
      "deve ser booleano",
    );
    if (finite(threshold) && finite(value.score))
      check(
        value.flagged === value.score >= threshold,
        `${path}.flagged`,
        "alerta não corresponde ao score e limiar desta execução",
      );
    for (const field of ["img", "rec", "heat"])
      asset(value[field], `${path}.${field}`);
    check(
      Array.isArray(value.boxes) &&
        value.boxes.every(
          (box) =>
            Array.isArray(box) &&
            box.length === 4 &&
            box.every(finite) &&
            box.every((v) => v >= 0 && v <= 100),
        ),
      `${path}.boxes`,
      "caixas devem conter quatro coordenadas percentuais válidas",
    );
  }
  function exams(values, path, threshold) {
    if (!Array.isArray(values) || !values.length)
      return check(false, path, "deve conter ao menos um exame");
    values.forEach((value, i) => exam(value, `${path}[${i}]`, threshold));
    check(
      new Set(values.map((value) => value?.id)).size === values.length,
      path,
      "há códigos de exame duplicados",
    );
  }
  if (!object(data))
    return { errors: ["Exportação deve ser um objeto JSON"], assets: [] };
  check(
    typeof data.generated === "string" &&
      /^\d{4}-\d{2}-\d{2}$/.test(data.generated),
    "generated",
    "data deve estar no formato AAAA-MM-DD",
  );
  if (!object(data.runs))
    return { errors: [...errors, "runs: execuções ausentes"], assets: [] };
  check(
    typeof data.main === "string" && Object.hasOwn(data.runs, data.main),
    "main",
    "execução principal não encontrada em runs",
  );
  for (const [key, run] of Object.entries(data.runs)) {
    const path = `runs.${key}`;
    if (!object(run) || !object(run.model)) {
      check(false, path, "modelo ausente");
      continue;
    }
    const model = run.model;
    check(
      typeof model.run === "string" && model.run.length > 0,
      `${path}.model.run`,
      "nome da execução obrigatório",
    );
    check(
      Array.isArray(model.views) &&
        model.views.length > 0 &&
        model.views.every((view) => typeof view === "string"),
      `${path}.model.views`,
      "incidências ausentes",
    );
    for (const field of ["threshold", "sigma"])
      check(
        finite(model[field]) && model[field] >= 0,
        `${path}.model.${field}`,
        "deve ser finito e não negativo",
      );
    for (const field of ["latent_dim", "n_test"])
      positiveInteger(model[field], `${path}.model.${field}`);
    probability(model.auc_val, `${path}.model.auc_val`);
    for (const field of ["image_size", "original_image_size", "epochs"])
      if (model[field] !== undefined)
        positiveInteger(model[field], `${path}.model.${field}`);
    if (model.test !== undefined) {
      if (!object(model.test))
        check(false, `${path}.model.test`, "métricas inválidas");
      else {
        for (const field of ["roc_auc", "sensitivity", "specificity"])
          probability(model.test[field], `${path}.model.test.${field}`);
        check(
          object(model.test.auc_by_view),
          `${path}.model.test.auc_by_view`,
          "métricas por incidência ausentes",
        );
        if (object(model.test.auc_by_view))
          for (const [view, value] of Object.entries(model.test.auc_by_view))
            probability(value, `${path}.model.test.auc_by_view.${view}`);
      }
    }
    exams(run.queue, `${path}.queue`, model.threshold);
  }
  exams(data.game, "game");
  if (
    !object(data.sigma) ||
    !Array.isArray(data.sigma.steps) ||
    !data.sigma.steps.length
  )
    check(false, "sigma", "exemplo e etapas de suavização ausentes");
  else {
    exam(data.sigma.example, "sigma.example");
    data.sigma.steps.forEach((step, i) => {
      if (!object(step))
        return check(false, `sigma.steps[${i}]`, "etapa inválida");
      check(
        finite(step.sigma) && step.sigma >= 0,
        `sigma.steps[${i}].sigma`,
        "suavização inválida",
      );
      probability(step.auc_val, `sigma.steps[${i}].auc_val`);
      asset(step.heat, `sigma.steps[${i}].heat`);
    });
    check(
      data.sigma.steps.some((step) => step?.sigma === data.sigma.chosen),
      "sigma.chosen",
      "suavização escolhida não existe nas etapas",
    );
  }
  if (!object(data.latent)) check(false, "latent", "dados latentes ausentes");
  else {
    const latent = data.latent;
    check(
      Array.isArray(latent.values) &&
        latent.values.length >= 2 &&
        latent.values.every(finite),
      "latent.values",
      "ao menos dois valores finitos são necessários",
    );
    positiveInteger(latent.atlas_cols, "latent.atlas_cols");
    asset(latent.atlas, "latent.atlas");
    check(
      Array.isArray(latent.points) &&
        latent.points.every(
          (point) =>
            Array.isArray(point) &&
            point.length === 3 &&
            finite(point[0]) &&
            finite(point[1]) &&
            ["normal", "opacidade", "outra"].includes(point[2]),
        ),
      "latent.points",
      "pontos inválidos",
    );
    check(
      Array.isArray(latent.dims) &&
        latent.dims.length === data.runs[data.main]?.model?.latent_dim,
      "latent.dims",
      "quantidade de sprites difere da dimensão latente do modelo principal",
    );
    if (Array.isArray(latent.dims))
      latent.dims.forEach((value, i) => asset(value, `latent.dims[${i}]`));
    if (!object(latent.interpolation))
      check(false, "latent.interpolation", "interpolação ausente");
    else {
      check(
        Number.isInteger(latent.interpolation.steps) &&
          latent.interpolation.steps >= 2,
        "latent.interpolation.steps",
        "ao menos dois passos são necessários",
      );
      asset(latent.interpolation.sprite, "latent.interpolation.sprite");
      exam(latent.interpolation.from, "latent.interpolation.from");
      exam(latent.interpolation.to, "latent.interpolation.to");
    }
  }
  return { errors, assets: [...assets] };
}
