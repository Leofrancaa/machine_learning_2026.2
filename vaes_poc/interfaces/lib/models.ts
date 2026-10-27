export interface ModelReport {
  id: string;
  version: string;
  scope: "pa" | "ap_pa";
  n: number;
  positives: number;
  metadata: {
    version?: string;
    image_size: number;
    latent_dim: number;
    epochs: number;
    seed?: number;
    sigma: number;
    threshold: number;
    threshold_rule: string;
    auc_val: number;
    normalize?: boolean;
    lung_mask?: boolean;
    crop_borders?: boolean;
    hygiene?: boolean;
  };
  metrics: {
    roc_auc: number;
    sensitivity: number;
    specificity: number;
    precision: number;
    average_precision: number;
    f1?: number;
    accuracy?: number;
    accuracy_all_normal?: number;
    confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
    localization: {
      pointing_game: number;
      box_area_fraction: number;
      box_fraction_in_mask?: number;
      extent_spearman: number;
    };
    ranking?: {
      precision_at_10: number;
      recall_at_10: number;
      precision_at_100: number;
    };
    queue_simulation: {
      mean_position_arrival: number;
      mean_position_model: number;
      top10_arrival: number;
      top10_model: number;
    };
    views: Record<string, number>;
  };
  thresholds: {
    threshold: string;
    split: string;
    sensitivity: number;
    specificity: number;
    precision: number;
    f1?: number;
    accuracy?: number;
  }[];
  methods: {
    method: string;
    uses_labels: boolean;
    auc_test: number;
    auc_test_std: number;
    n_runs: number;
    ap_test: number;
  }[];
  sources: string[];
}

export interface AblationReport {
  run: string;
  n_seeds: number;
  val_auc: number;
  val_auc_std: number;
  delta_auc: number;
  val_pointing: number;
  chance: number;
  pointing_above_chance: number;
  pointing_above_chance_std: number;
  delta_above_chance: number;
  accepted: boolean | null;
}

export interface ModelCatalog {
  schema: number;
  runs: ModelReport[];
  ablations: AblationReport[];
  ablationSource: string;
}

export function localizationGain(run: ModelReport) {
  const l = run.metrics.localization;
  return l.pointing_game - (l.box_fraction_in_mask ?? l.box_area_fraction);
}

export function preprocessing(run: ModelReport) {
  const m = run.metadata;
  return [
    m.hygiene ? "Higiene dos dados" : "Pipeline inicial",
    m.normalize && "Normalização",
    m.lung_mask && "Máscara pulmonar",
    m.crop_borders && "Recorte de bordas",
  ]
    .filter(Boolean)
    .join(" + ");
}
