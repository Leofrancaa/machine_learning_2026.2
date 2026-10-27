"use client";

import { useState } from "react";
import Segmented from "@/components/Segmented";
import Icon from "@/components/Icon";
import {
  localizationGain,
  preprocessing,
  type ModelCatalog,
  type ModelReport,
} from "@/lib/models";

const num = (v: number | undefined, digits = 3) =>
  v == null
    ? "—"
    : v.toLocaleString("pt-BR", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
const pct = (v: number | undefined) =>
  v == null ? "—" : `${num(v * 100, 1)}%`;
const pp = (v: number) => `${v >= 0 ? "+" : ""}${num(v * 100, 1)} p.p.`;
const count = (v: number) => num(v, 0);
const ablationNames: Record<string, string> = {
  "abl-A-recorte": "A · Recorte de bordas",
  "abl-B-normalizacao": "B · Normalização",
  "abl-C1-mascara": "C1 · Máscara pulmonar",
  "abl-B+C1": "B + C1 · Combinação",
};

export default function ModelsTab({
  catalog,
  activeRun,
}: {
  catalog: ModelCatalog;
  activeRun: string;
}) {
  const [scope, setScope] = useState<"pa" | "ap_pa">("pa");
  const [section, setSection] = useState<"evolution" | "methods" | "ablations">(
    "evolution",
  );
  const [version, setVersion] = useState(
    catalog.runs.filter((r) => r.scope === "pa").at(-1)?.version ?? "v3",
  );
  const [metric, setMetric] = useState<
    "roc_auc" | "sensitivity" | "specificity"
  >("roc_auc");
  const runs = catalog.runs.filter((r) => r.scope === scope);
  const run = runs.find((r) => r.version === version) ?? runs.at(-1)!;
  const m = run.metrics;
  const vae = run.methods.find((r) => r.method === "VAE");
  const metricNames = {
    roc_auc: "AUC ROC",
    sensitivity: "Sensibilidade",
    specificity: "Especificidade",
  };
  const chance =
    m.localization.box_fraction_in_mask ?? m.localization.box_area_fraction;

  return (
    <div className="models-experience">
      <div className="models-toolbar">
        <div>
          <span className="control-label">POPULAÇÃO AVALIADA</span>
          <Segmented
            label="População dos modelos"
            value={scope}
            onChange={setScope}
            options={[
              { value: "pa", label: "Somente PA" },
              { value: "ap_pa", label: "AP + PA" },
            ]}
          />
        </div>
        <label className="models-version">
          Inspecionar versão
          <select
            value={run.version}
            onChange={(e) => setVersion(e.target.value)}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.version}>
                {r.version.toUpperCase()}
                {r.id === activeRun ? " · na demo" : ""}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="models-hero">
        <div>
          <p className="eyebrow">EVIDÊNCIAS EXPERIMENTAIS · RSNA</p>
          <h2>
            O que mudou.
            <br />
            <span>O que os números sustentam.</span>
          </h2>
          <p>
            Compare as versões do pipeline e os métodos de referência. A fila
            mostra uma amostra; aqui você vê a avaliação completa.
          </p>
          <div className="model-pipeline">
            <span>{run.version.toUpperCase()}</span>
            {preprocessing(run)}
          </div>
          <div className="model-cohort">
            {count(run.n)} exames de teste <i>·</i> {count(run.positives)} com
            opacidade <i>·</i> prevalência {pct(run.positives / run.n)}
          </div>
        </div>
        <div className="model-auc">
          <span>AUC ROC · MODELO SALVO</span>
          <strong data-testid="model-auc">{num(m.roc_auc)}</strong>
          <p>
            Capacidade de ordenar opacidades acima de normais, considerando
            todos os limiares.
          </p>
          {vae && (
            <small>
              {vae.n_runs > 1
                ? `Média de ${vae.n_runs} treinos: ${num(vae.auc_test)} ± ${num(vae.auc_test_std)} (DP)`
                : "Uma execução de referência"}
            </small>
          )}
        </div>
      </div>

      {scope === "ap_pa" && (
        <aside className="model-caution">
          <Icon name="info" size={19} />
          <div>
            <strong>Uma população diferente muda a comparação.</strong>
            <p>
              A prevalência e as incidências são diferentes de PA. O score
              separa AP de PA até entre normais (AUC{" "}
              {num(m.views.roc_auc_ap_vs_pa_normals)}); parte do resultado
              agregado pode refletir aquisição. Dentro de AP:{" "}
              {num(m.views.roc_auc_AP)}. Dentro de PA: {num(m.views.roc_auc_PA)}
              .
            </p>
          </div>
        </aside>
      )}

      <div className="model-kpis">
        <Metric
          label="Sensibilidade"
          value={pct(m.sensitivity)}
          detail={`${m.confusion_matrix.tp} de ${run.positives} opacidades geraram alerta no p95.`}
        />
        <Metric
          label="Especificidade"
          value={pct(m.specificity)}
          detail={`${m.confusion_matrix.tn} de ${run.n - run.positives} normais ficaram sem alerta.`}
        />
        <Metric
          label="Precisão média · AP"
          value={num(m.average_precision)}
          detail={`Resumo da curva precisão × recall. AP significa average precision. Referência aleatória: ${num(run.positives / run.n)}.`}
        />
        <Metric
          label="Posição média · fila simulada"
          value={num(m.queue_simulation.mean_position_model, 1)}
          detail={`Por chegada: ${num(m.queue_simulation.mean_position_arrival, 1)}.${["v1", "v2", "v3"].includes(run.version) ? " Filas de 100 exames, com 10% de opacidades." : " Menor posição significa revisão mais cedo."}`}
        />
      </div>

      <Segmented
        label="Comparações disponíveis"
        value={section}
        onChange={setSection}
        options={[
          { value: "evolution", label: "Evolução das versões" },
          { value: "methods", label: "Comparar métodos" },
          { value: "ablations", label: "Ablações na validação" },
        ]}
      />

      {section === "evolution" && (
        <>
          <section
            className="panel model-evolution"
            aria-labelledby="evolution-title"
          >
            <div className="model-section-heading">
              <div>
                <p className="eyebrow">01 · EVOLUÇÃO DO PIPELINE</p>
                <h2 id="evolution-title">O mesmo objetivo, novas escolhas</h2>
                <p>Clique em uma versão para inspecionar seus resultados.</p>
              </div>
              <label>
                Métrica no gráfico
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value as typeof metric)}
                >
                  {Object.entries(metricNames).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div
              className="model-chart"
              role="group"
              aria-label={`${metricNames[metric]} por versão`}
            >
              <div className="model-chart-axis" aria-hidden="true">
                <span>0</span>
                <span>0,5</span>
                <span>1,0</span>
              </div>
              {runs.map((r) => (
                <button
                  key={r.id}
                  className={`model-chart-row ${r.id === run.id ? "selected" : ""}`}
                  aria-pressed={r.id === run.id}
                  onClick={() => setVersion(r.version)}
                  aria-label={`${r.version.toUpperCase()}, ${metricNames[metric]} ${metric === "roc_auc" ? num(r.metrics[metric]) : pct(r.metrics[metric])}`}
                >
                  <span>
                    <b>{r.version.toUpperCase()}</b>
                    <small>{preprocessing(r)}</small>
                  </span>
                  <span className="model-chart-track">
                    <i style={{ width: `${r.metrics[metric] * 100}%` }} />
                  </span>
                  <strong>
                    {metric === "roc_auc"
                      ? num(r.metrics[metric])
                      : pct(r.metrics[metric])}
                  </strong>
                </button>
              ))}
            </div>
            <div
              className="model-table-scroll"
              role="region"
              aria-label="Métricas por versão"
              tabIndex={0}
            >
              <table className="model-table">
                <caption>
                  Modelo salvo de cada versão · métricas de teste, exceto AUC
                  val
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Versão</th>
                    <th scope="col">Teste / opacidades</th>
                    <th scope="col">AUC val</th>
                    <th scope="col">AUC teste</th>
                    <th scope="col">Sensib.</th>
                    <th scope="col">Especif.</th>
                    <th scope="col">F1</th>
                    <th scope="col">σ</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr
                      key={r.id}
                      className={r.id === run.id ? "selected" : ""}
                    >
                      <th scope="row">{r.version.toUpperCase()}</th>
                      <td>
                        {count(r.n)} / {count(r.positives)}
                      </td>
                      <td>{num(r.metadata.auc_val)}</td>
                      <td>{num(r.metrics.roc_auc)}</td>
                      <td>{pct(r.metrics.sensitivity)}</td>
                      <td>{pct(r.metrics.specificity)}</td>
                      <td>{num(r.metrics.f1)}</td>
                      <td>{r.metadata.sigma} px</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="model-note">
              <b>Como comparar:</b> v1 → v2 também altera a população após a
              higiene dos dados; esse ganho não isola uma mudança do modelo. v2
              → v3 preserva o tamanho do teste e muda o pré-processamento.
              Sensibilidade, especificidade e F1 são medidas no p95 dos normais
              da validação, recalibrado em cada versão.
            </p>
          </section>

          <div className="model-detail-grid">
            <ThresholdLab key={run.id} run={run} />
            <section
              className="panel model-localization"
              aria-labelledby="localization-title"
            >
              <p className="eyebrow">03 · LOCALIZAÇÃO</p>
              <h2 id="localization-title">O mapa aponta para a opacidade?</h2>
              <p>
                Acerto: o pixel mais quente cai dentro de uma caixa do laudo. O
                acaso aumenta quando o mapa fica restrito aos pulmões.
              </p>
              <div className="localization-values">
                <div>
                  <strong>{pct(m.localization.pointing_game)}</strong>
                  <span>Acerto observado</span>
                </div>
                <span aria-hidden="true">−</span>
                <div>
                  <strong>{pct(chance)}</strong>
                  <span>Acaso da configuração</span>
                </div>
              </div>
              <div className="localization-gain">
                <span>Acima do acaso</span>
                <strong>{pp(localizationGain(run))}</strong>
              </div>
              <p className="model-note">
                Compare o ganho acima do acaso entre versões. Um mapa
                visualmente mais limpo não prova uma localização mais precisa. A
                máscara é média, aprendida nos normais de treino; não segmenta
                cada paciente.
              </p>
              <p className="model-note">
                Score × área da lesão (Spearman):{" "}
                <b>{num(m.localization.extent_spearman, 2)}</b>. Esse número não
                valida o score como medida de gravidade.
              </p>
            </section>
          </div>
        </>
      )}

      {section === "methods" && (
        <section
          className="panel model-methods"
          aria-labelledby="methods-title"
        >
          <div className="model-section-heading">
            <div>
              <p className="eyebrow">
                MESMO PIPELINE · {run.version.toUpperCase()} ·{" "}
                {scope === "pa" ? "PA" : "AP + PA"}
              </p>
              <h2 id="methods-title">
                O VAE precisa superar referências simples
              </h2>
              <p>
                Todos os métodos abaixo vêm da mesma execução experimental. O
                número de treinos está explícito.
              </p>
            </div>
          </div>
          <div
            className="model-table-scroll"
            role="region"
            aria-label="Comparação de métodos"
            tabIndex={0}
          >
            <table className="model-table">
              <caption>
                AUC no teste · média ± desvio-padrão quando há vários treinos
              </caption>
              <thead>
                <tr>
                  <th scope="col">Método</th>
                  <th scope="col">Usa rótulos na tarefa?</th>
                  <th scope="col">Treinos</th>
                  <th scope="col">AUC ROC</th>
                  <th scope="col">± DP</th>
                  <th scope="col">Precisão média</th>
                </tr>
              </thead>
              <tbody>
                {run.methods.map((method) => (
                  <tr
                    key={method.method}
                    className={method.method === "VAE" ? "selected" : ""}
                  >
                    <th scope="row">{method.method.replace(/\*$/, "")}</th>
                    <td>{method.uses_labels ? "Sim" : "Não"}</td>
                    <td>{method.n_runs}</td>
                    <td>
                      <div className="method-auc">
                        <i style={{ width: `${method.auc_test * 100}%` }} />
                        <b>{num(method.auc_test)}</b>
                      </div>
                    </td>
                    <td>
                      {method.n_runs > 1 ? num(method.auc_test_std) : "—"}
                    </td>
                    <td>{num(method.ap_test)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="model-reading">
            <Icon name="layers" size={24} />
            <div>
              <h3>O que o “variacional” acrescenta?</h3>
              <p>
                Na v3, as médias do VAE, do autoencoder comum e do PCA ficam
                próximas. Estes resultados não demonstram superioridade do VAE
                na detecção. A parte variacional permite organizar e explorar o
                espaço latente, gerar imagens e interpolar entre exames.
              </p>
              <p>
                “Não” significa que o treinamento do reconstrutor não usa
                rótulos de opacidade. O σ é escolhido com AUC na validação, que
                usa rótulos. A coluna não torna o pipeline inteiro livre de
                supervisão.
              </p>
            </div>
          </div>
        </section>
      )}

      {section === "ablations" && (
        <section
          className="panel model-ablations"
          aria-labelledby="ablations-title"
        >
          <div className="model-section-heading">
            <div>
              <p className="eyebrow">ESCOLHA DA V3 · SOMENTE PA</p>
              <h2 id="ablations-title">
                Mudar uma coisa. Medir antes de escolher.
              </h2>
              <p>
                Ablações comparadas na validação, com{" "}
                {catalog.ablations[0].n_seeds} sementes por configuração. Esta
                tabela é de PA, independentemente da população selecionada
                acima.
              </p>
            </div>
            <span className="model-badge">
              TESTE RESERVADO À CONFIGURAÇÃO FINAL
            </span>
          </div>
          <div className="ablation-rule">
            <strong>Regra fixada antes dos resultados</strong>
            <p>
              Aceitar com ganho ≥ 0,02 de AUC <b>ou</b> ≥ 8 p.p. de localização
              acima do acaso, desde que a outra métrica não caia mais que um
              desvio-padrão da v2. A combinação B + C1 segue uma regra
              adicional: não perder mais de um DP de AUC da v2 em relação a B.
            </p>
          </div>
          <div
            className="model-table-scroll"
            role="region"
            aria-label="Resultados das ablações"
            tabIndex={0}
          >
            <table className="model-table">
              <caption>Validação · média de sementes ± desvio-padrão</caption>
              <thead>
                <tr>
                  <th scope="col">Configuração</th>
                  <th scope="col">AUC val ± DP</th>
                  <th scope="col">Δ AUC</th>
                  <th scope="col">Acerto / acaso</th>
                  <th scope="col">Acima do acaso</th>
                  <th scope="col">Decisão registrada</th>
                </tr>
              </thead>
              <tbody>
                {catalog.ablations.map((a, index) => (
                  <tr
                    key={a.run}
                    className={a.run === "abl-B+C1" ? "selected" : ""}
                  >
                    <th scope="row">
                      {ablationNames[a.run] ??
                        (index === 0 ? "V2 · Referência" : a.run)}
                    </th>
                    <td>
                      {num(a.val_auc)} ± {num(a.val_auc_std)}
                    </td>
                    <td>
                      {index === 0
                        ? "—"
                        : `${a.delta_auc >= 0 ? "+" : ""}${num(a.delta_auc)}`}
                    </td>
                    <td>
                      {pct(a.val_pointing)} / {pct(a.chance)}
                    </td>
                    <td>{pp(a.pointing_above_chance)}</td>
                    <td>
                      <span
                        className={`ablation-status ${a.accepted === true ? "accepted" : ""}`}
                      >
                        {a.accepted == null
                          ? "Referência"
                          : a.accepted
                            ? "Aceita"
                            : "Rejeitada"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="ablation-explanations">
            <div>
              <span>A</span>
              <h3>Recorte de bordas</h3>
              <p>
                Remove molduras pretas e redimensiona. O ganho médio não atingiu
                a regra; não entrou na v3.
              </p>
            </div>
            <div>
              <span>B</span>
              <h3>Normalização</h3>
              <p>
                Estica a intensidade de cada imagem entre seus percentis 1 e 99.
                Reduz diferenças de brilho e foi aceita pelo ganho de AUC.
              </p>
            </div>
            <div>
              <span>C1</span>
              <h3>Máscara pulmonar média</h3>
              <p>
                Zera o mapa fora de uma região fixa derivada do treino. Foi
                aceita pela AUC; o ganho de localização isolado ficou abaixo dos
                8 p.p. exigidos.
              </p>
            </div>
          </div>
          <p className="model-note">
            <b>Escolha final: B + C1.</b> A combinação preservou a AUC em
            relação a B dentro da tolerância definida. Três sementes revelam
            variação entre treinos, mas não são um intervalo de confiança nem
            uma prova de significância estatística.
          </p>
        </section>
      )}

      <details className="panel model-protocol">
        <summary>Como esses números foram produzidos e onde conferir</summary>
        <div className="protocol-grid">
          <div>
            <h3>Treino → validação → teste</h3>
            <p>
              O VAE aprende a reconstruir usando apenas normais de treino. A
              validação escolhe σ e calibra o limiar; as ablações também são
              decididas nela. O teste mede a configuração final. A fila de
              demonstração não substitui essa avaliação.
            </p>
            <p>
              Modelo selecionado: {run.metadata.image_size} ×{" "}
              {run.metadata.image_size} px, {run.metadata.latent_dim} dimensões
              latentes, {run.metadata.epochs} épocas, semente{" "}
              {run.metadata.seed ?? 0}. Limiar: {run.metadata.threshold_rule}.
              Limiar bruto: {num(run.metadata.threshold, 6)}; a interface mostra
              × 1.000.
            </p>
            <p>
              A fila simulada dos relatórios usa amostragens controladas de
              exames; a prevalência da simulação é diferente da prevalência do
              teste completo. Precision@10 no teste (
              {pct(m.ranking?.precision_at_10)}) e opacidades nas primeiras 10
              posições da simulação ({num(m.queue_simulation.top10_model, 1)})
              são medidas de populações diferentes.
            </p>
          </div>
          <div>
            <h3>Fontes versionadas</h3>
            <p>
              Dados extraídos dos arquivos do repositório ao iniciar ou
              construir a interface. Sem inferência no navegador e sem métricas
              digitadas no componente.
            </p>
            <ul>
              {run.sources.map((source) => (
                <li key={source}>
                  <code>{source}</code>
                </li>
              ))}
              <li>
                <code>{catalog.ablationSource}</code>
              </li>
            </ul>
            <a
              className="btn"
              href="/data/models.json"
              download="vae-modelos-relatorios.json"
            >
              <Icon name="arrow" size={15} />
              Baixar dados da comparação
            </a>
          </div>
        </div>
        <div className="model-definitions">
          <h3>Como ler as métricas</h3>
          <dl>
            <div>
              <dt>AUC ROC</dt>
              <dd>
                Resume a ordenação em todos os limiares. 0,5 corresponde à
                ordenação aleatória; 1,0, à separação perfeita. AUC de 0,804 não
                significa 80,4% de acurácia.
              </dd>
            </div>
            <div>
              <dt>Sensibilidade e especificidade</dt>
              <dd>
                Sensibilidade (recall) é a fração das opacidades com alerta.
                Especificidade é a fração dos normais sem alerta. Ambas dependem
                do limiar escolhido.
              </dd>
            </div>
            <div>
              <dt>Precisão e AP</dt>
              <dd>
                Precisão é a fração de alertas que são opacidades. Average
                precision (AP) resume precisão e recall em vários limiares; sua
                referência aleatória acompanha a prevalência. Aqui AP não
                significa incidência anteroposterior.
              </dd>
            </div>
            <div>
              <dt>F1 e acurácia</dt>
              <dd>
                F1 é a média harmônica entre precisão e recall. Acurácia é a
                fração total de acertos: com poucas opacidades, prever normal
                para todos já pode parecer bom. Campos ausentes aparecem como
                “—”, sem assumir zero.
              </dd>
            </div>
            <div>
              <dt>Sementes e desvio-padrão</dt>
              <dd>
                Sementes mudam a aleatoriedade do treino. Média ± DP descreve a
                dispersão entre esses treinos; não é um intervalo de confiança.
                Uma única execução não permite medir essa dispersão.
              </dd>
            </div>
            <div>
              <dt>Localização e gravidade</dt>
              <dd>
                O pointing game verifica se o ponto mais quente cai numa caixa.
                Subtrair o acaso torna a leitura mais justa entre máscaras. Isso
                não mede a área da lesão nem estabelece urgência clínica.
              </dd>
            </div>
          </dl>
        </div>
      </details>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="model-kpi">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  );
}

function ThresholdLab({ run }: { run: ModelReport }) {
  const rows = run.thresholds.filter((r) => r.split === "test");
  const [rule, setRule] = useState(rows[0].threshold);
  const selected = rows.find((r) => r.threshold === rule)!;
  const tp = Math.round(selected.sensitivity * run.positives);
  const fn = run.positives - tp;
  const tn = Math.round(selected.specificity * (run.n - run.positives));
  const fp = run.n - run.positives - tn;
  return (
    <section className="panel threshold-lab" aria-labelledby="threshold-title">
      <p className="eyebrow">02 · TRADE-OFF DO ALERTA</p>
      <h2 id="threshold-title">Mais sensível, mais falsos alarmes</h2>
      <p>
        Selecione uma regra já calibrada na validação e veja seu resultado no
        teste. Isso não altera a fila da demo nem recalibra o modelo.
      </p>
      <label>
        Regra do limiar
        <select value={rule} onChange={(e) => setRule(e.target.value)}>
          {rows.map((r) => (
            <option key={r.threshold} value={r.threshold}>
              {r.threshold}
            </option>
          ))}
        </select>
      </label>
      <div className="threshold-rates">
        <span>
          Sensibilidade <b>{pct(selected.sensitivity)}</b>
        </span>
        <span>
          Especificidade <b>{pct(selected.specificity)}</b>
        </span>
      </div>
      <div
        className="confusion-grid"
        role="group"
        aria-label="Resultados do alerta no teste"
      >
        <div>
          <span>Opacidades com alerta</span>
          <strong data-testid="true-positives">{count(tp)}</strong>
          <small>Verdadeiros positivos</small>
        </div>
        <div className="missed">
          <span>Opacidades sem alerta</span>
          <strong>{count(fn)}</strong>
          <small>Falsos negativos</small>
        </div>
        <div className="missed">
          <span>Normais com alerta</span>
          <strong data-testid="false-positives">{count(fp)}</strong>
          <small>Falsos positivos</small>
        </div>
        <div>
          <span>Normais sem alerta</span>
          <strong>{count(tn)}</strong>
          <small>Verdadeiros negativos</small>
        </div>
      </div>
      <p className="model-note">
        Precisão dos alertas: <b>{pct(selected.precision)}</b>. Youden usa
        rótulos de opacidade para escolher o limiar. As regras p95, p90 e p80
        usam apenas os scores dos normais de validação. Alterar o limiar muda os
        alertas, não a ordem por score nem a AUC.
      </p>
    </section>
  );
}
