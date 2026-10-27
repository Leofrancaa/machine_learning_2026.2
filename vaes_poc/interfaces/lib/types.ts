// Formato do public/data/data.json, gerado por `make export` (module_vae/export.py).

export type Label = "normal" | "opacidade";

/** Caixa do laudo em % da imagem: x, y, largura, altura. */
export type Box = [number, number, number, number];

export interface Exam {
  id: string;
  label: Label;
  view: string; // "AP" ou "PA"
  score: number;
  percentile: number; // fração dos normais da validação com score menor
  flagged: boolean; // score acima do limiar
  boxes: Box[];
  img: string;
  rec: string;
  heat: string;
  arrival?: string; // só nos exames da fila
}

export interface ModelInfo {
  run: string;
  views: string[];
  sigma: number;
  threshold: number;
  latent_dim: number;
  auc_val: number;
  n_test: number;
  image_size?: number;
  original_image_size?: number;
  epochs?: number;
  version?: string;
  threshold_rule?: string;
  normalize?: boolean;
  lung_mask?: boolean;
  test?: {
    roc_auc: number;
    sensitivity: number;
    specificity: number;
    auc_by_view: Record<string, number>;
    auc_ap_vs_pa_normals?: number; // só no modelo com AP e PA
  };
}

export interface RunData {
  model: ModelInfo;
  queue: Exam[];
}

export interface SigmaStep {
  sigma: number;
  auc_val: number;
  heat: string;
}

export type PointClass = "normal" | "opacidade" | "outra";

/** Um exame no mapa latente: as 2 primeiras componentes do PCA dos μ e a classe. */
export type LatentPoint = [number, number, PointClass];

export interface LatentData {
  values: number[]; // valores de cada eixo de z nos sprites do gerador
  points: LatentPoint[];
  atlas: string;
  atlas_cols: number;
  dims: string[]; // um sprite por eixo de z
  interpolation: { sprite: string; steps: number; from: Exam; to: Exam };
}

export interface VaeData {
  generated: string;
  main: string; // chave do modelo principal em runs
  runs: Record<string, RunData>;
  game: Exam[];
  sigma: { example: Exam; chosen: number; steps: SigmaStep[] };
  latent: LatentData;
}
