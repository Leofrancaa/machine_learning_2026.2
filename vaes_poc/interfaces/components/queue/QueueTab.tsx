"use client";

import { useEffect, useMemo, useState } from "react";

import Segmented from "@/components/Segmented";
import Icon from "@/components/Icon";
import Viewer from "@/components/queue/Viewer";
import Worklist from "@/components/queue/Worklist";
import QueueComparison from "@/components/queue/QueueComparison";
import ExamDialog from "@/components/queue/ExamDialog";
import type { InspectionMode } from "@/components/queue/ExamDialog";
import { byScore, fmtScore } from "@/lib/format";
import type { Exam, VaeData } from "@/lib/types";

type Order = "arrival" | "vae";
const pt = (x: number) => x.toFixed(2).replace(".", ",");

const isPositive = (e: Exam) => e.label === "opacidade";
const positivesInTop5 = (list: Exam[]) =>
  list.slice(0, 5).filter(isPositive).length;
function meanPositivePosition(list: Exam[]) {
  const pos = list.flatMap((e, i) => (isPositive(e) ? [i + 1] : []));
  return pos.length
    ? (pos.reduce((a, b) => a + b, 0) / pos.length).toFixed(1).replace(".", ",")
    : "–";
}

// Cada modelo ("pa" e, se exportado, "ap_pa") tem a própria fila, sorteada do seu teste.
export default function QueueTab({ data }: { data: VaeData }) {
  const [runKey, setRunKey] = useState(data.main);
  const run = data.runs[runKey];
  const [arrived, setArrived] = useState(() => run.queue.map((e) => e.id)); // ordem de chegada
  const [order, setOrder] = useState<Order>("vae");
  const [reveal, setReveal] = useState(false);
  const [selectedId, setSelectedId] = useState(() => byScore(run.queue)[0].id);
  const [simulating, setSimulating] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "flagged">("all");
  const [inspection, setInspection] = useState<InspectionMode | null>(null);
  const [intervalMs, setIntervalMs] = useState(650);
  const [started, setStarted] = useState(false);
  const [followArrivals, setFollowArrivals] = useState(true);
  const [presentation, setPresentation] = useState(false);

  const byId = useMemo(
    () => Object.fromEntries(run.queue.map((e) => [e.id, e])),
    [run],
  );
  const arrivedExams = arrived.map((id) => byId[id]);
  const vaeOrder = byScore(arrivedExams);
  const ordered = order === "vae" ? vaeOrder : arrivedExams;
  const shown = ordered.filter(
    (exam) =>
      (filter === "all" || exam.flagged) &&
      exam.id.toLowerCase().includes(query.trim().toLowerCase()),
  );
  const selected = shown.find((exam) => exam.id === selectedId) ?? shown[0];
  const selectedIndex = selected ? shown.indexOf(selected) : -1;
  const flagged = arrivedExams.filter((exam) => exam.flagged).length;
  const partial = arrived.length < run.queue.length;
  const multiView = run.model.views.length > 1;
  const test = run.model.test;
  const latest = started ? arrivedExams[arrivedExams.length - 1] : undefined;

  useEffect(() => {
    if (followArrivals && latest) setSelectedId(latest.id);
  }, [followArrivals, latest]);

  useEffect(() => {
    const closeInspection = () => setInspection(null);
    window.addEventListener("hashchange", closeInspection);
    return () => window.removeEventListener("hashchange", closeInspection);
  }, []);

  // "Simular chegada": os exames entram um a um, e a fila se reordena a cada chegada.
  useEffect(() => {
    if (!simulating) return;
    const timer = setInterval(() => {
      setArrived((prev) =>
        prev.length < run.queue.length
          ? [...prev, run.queue[prev.length].id]
          : prev,
      );
    }, intervalMs);
    return () => clearInterval(timer);
  }, [simulating, run, intervalMs]);

  useEffect(() => {
    if (simulating && arrived.length === run.queue.length) setSimulating(false);
  }, [simulating, arrived, run]);

  function switchRun(key: string) {
    const next = data.runs[key];
    setRunKey(key);
    setSimulating(false);
    setArrived(next.queue.map((e) => e.id));
    setSelectedId(byScore(next.queue)[0].id);
    setQuery("");
    setFilter("all");
    setStarted(false);
  }

  function simulate() {
    if (partial) {
      setSimulating((prev) => !prev);
      return;
    }
    setArrived([]);
    setSelectedId("");
    setStarted(true);
    setSimulating(true);
  }

  function selectExam(id: string, resetFilters = false) {
    if (resetFilters) {
      setQuery("");
      setFilter("all");
    }
    setSelectedId(id);
    if (started) setFollowArrivals(false);
    if (window.matchMedia("(max-width: 760px)").matches) setInspection("image");
  }

  function previousExam() {
    if (started) setFollowArrivals(false);
    if (selectedIndex > 0) setSelectedId(shown[selectedIndex - 1].id);
  }
  function nextExam() {
    if (started) setFollowArrivals(false);
    if (selectedIndex < shown.length - 1)
      setSelectedId(shown[selectedIndex + 1].id);
  }

  return (
    <>
      <div className="queue-toolbar">
        <div className="model-control">
          <span className="control-label">MODELO</span>
          <Segmented
            label="Modelo"
            value={runKey}
            onChange={switchRun}
            options={[
              { value: "pa", label: "Somente PA" },
              { value: "ap_pa", label: "AP + PA", disabled: !data.runs.ap_pa },
            ]}
          />
          <span className="run-provenance">
            <strong>{run.model.run}</strong>
            <span>
              Exportação: {data.generated.split("-").reverse().join("/")}
            </span>
          </span>
        </div>
        <div className="controls">
          <button
            className="btn presentation-toggle"
            aria-pressed={presentation}
            onClick={() => {
              if (!presentation) {
                setQuery("");
                setFilter("all");
              }
              setPresentation((value) => !value);
            }}
          >
            <Icon name="expand" size={15} />
            {presentation ? "Voltar à exploração" : "Modo apresentação"}
          </button>
          {partial && (
            <button
              className="btn"
              onClick={() => {
                setSimulating(false);
                setStarted(false);
                setArrived(run.queue.map((exam) => exam.id));
              }}
            >
              <Icon name="reset" size={15} />
              Restaurar fila
            </button>
          )}
          <button className="btn primary" onClick={simulate}>
            <Icon name={simulating ? "pause" : "play"} size={15} />
            {simulating
              ? "Pausar simulação"
              : partial
                ? "Continuar simulação"
                : "Simular chegada"}
          </button>
        </div>
      </div>
      <section
        className="simulation-console"
        aria-label="Controles da simulação"
      >
        <div className="simulation-phase">
          <span className={`status-dot ${simulating ? "is-live" : ""}`} />
          <strong>
            {simulating
              ? "Recebendo exames"
              : partial
                ? "Simulação pausada"
                : started
                  ? "Simulação concluída"
                  : "Fila carregada"}
          </strong>
          <span>
            {arrived.length} / {run.queue.length}
          </span>
        </div>
        <progress
          value={arrived.length}
          max={run.queue.length}
          aria-label="Progresso da chegada dos exames"
        />
        <div className="simulation-settings">
          <label className="switch small reveal-control">
            <input
              type="checkbox"
              checked={reveal}
              onChange={(event) => setReveal(event.target.checked)}
            />
            <span>Laudos</span>
          </label>
          <label>
            Velocidade
            <select
              aria-label="Velocidade da simulação"
              value={intervalMs}
              onChange={(event) => setIntervalMs(Number(event.target.value))}
            >
              <option value={1300}>Lenta · 1,3 s</option>
              <option value={650}>Normal · 0,65 s</option>
              <option value={325}>Rápida · 0,325 s</option>
            </select>
          </label>
          <label className="switch">
            <input
              type="checkbox"
              checked={followArrivals}
              onChange={(event) => setFollowArrivals(event.target.checked)}
            />
            <span>Seguir novas chegadas</span>
          </label>
          <button
            className="btn"
            disabled={simulating || !partial}
            onClick={() =>
              setArrived((previous) =>
                previous.length < run.queue.length
                  ? [...previous, run.queue[previous.length].id]
                  : previous,
              )
            }
          >
            <Icon name="plus" size={15} />
            Adicionar próximo exame
          </button>
        </div>
        <div className="arrival-narration" aria-live="polite">
          {latest ? (
            <>
              <img src={latest.img} alt="" width={34} height={34} />
              <p>
                <span>ÚLTIMA CHEGADA</span>
                <strong>{latest.id}</strong>
                <span>
                  Score × 1.000: {fmtScore(latest.score)} ·{" "}
                  {latest.flagged ? "acima do limiar" : "abaixo do limiar"}
                </span>
              </p>
            </>
          ) : (
            <>
              <Icon name="queue" size={24} />
              <p>
                <strong>Observe a mesma chegada nas duas filas.</strong>
                <span>
                  O ritmo controla a apresentação; os horários da fila são
                  simulados.
                </span>
              </p>
            </>
          )}
        </div>
      </section>
      <div className="kpis">
        <div className="kpi">
          <span className="kpi-label">
            Exames na fila <Icon name="queue" size={16} />
          </span>
          <span className="kpi-value">
            {String(arrived.length).padStart(2, "0")}
            <small>/ {run.queue.length}</small>
          </span>
          <span className="kpi-sub">
            <span className="status-dot" /> {flagged} acima do limiar de alerta
          </span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Opacidades entre os 5 primeiros</span>
          <span className="kpi-value">
            <span>{positivesInTop5(arrivedExams)}</span>
            <small>chegada</small>
            <span className="arrow">→</span>
            <span className="good">{positivesInTop5(vaeOrder)}</span>
            <small>VAE</small>
          </span>
          <span className="kpi-sub">Comparação nesta amostra da fila</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Posição média de uma opacidade</span>
          <span className="kpi-value">
            <span>{meanPositivePosition(arrivedExams)}</span>
            <small>chegada</small>
            <span className="arrow">→</span>
            <span className="good">{meanPositivePosition(vaeOrder)}</span>
            <small>VAE</small>
          </span>
          <span className="kpi-sub">
            Quanto menor a posição, mais cedo na fila
          </span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Limiar de alerta · × 1.000</span>
          <span className="kpi-value">{fmtScore(run.model.threshold)}</span>
          <span className="kpi-sub">
            {run.model.threshold_rule ?? "calibrado na validação"}
          </span>
        </div>
      </div>

      <div
        className={`triage-experience ${presentation ? "presentation" : ""}`}
      >
        <QueueComparison
          key={runKey}
          exams={arrivedExams}
          selectedId={selected?.id}
          reveal={reveal}
          onSelect={(id) => selectExam(id, true)}
          latestId={latest?.id}
        />
        <div className="queue-layout">
          <div className="panel worklist" hidden={presentation}>
            <div className="panel-head">
              <div className="list-title">
                <h2>Lista de trabalho</h2>
                <span className="count-badge">{shown.length}</span>
              </div>
              <div className="controls">
                <Segmented<Order>
                  label="Ordem da fila"
                  value={order}
                  onChange={setOrder}
                  options={[
                    { value: "arrival", label: "Chegada" },
                    { value: "vae", label: "Prioridade VAE" },
                  ]}
                />
              </div>
            </div>
            <div className="list-tools">
              <label className="search-field">
                <Icon name="search" size={16} />
                <input
                  aria-label="Buscar exame pelo código"
                  placeholder="Buscar exame..."
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  type="search"
                />
              </label>
              <Segmented
                label="Filtrar exames"
                value={filter}
                onChange={setFilter}
                options={[
                  { value: "all", label: "Todos" },
                  { value: "flagged", label: `Alertas (${flagged})` },
                ]}
              />
            </div>
            <div className="table-head">
              <span>#</span>
              <span>Exame</span>
              <span>Chegada</span>
              <span title="Score de anomalia multiplicado por 1.000">
                Score × 1.000
              </span>
              <span>Decisão</span>
              <span>Laudo</span>
            </div>
            {/* key: trocar de modelo monta uma lista nova, sem animar a troca */}
            <Worklist
              key={runKey}
              exams={shown}
              maxScore={Math.max(
                run.model.threshold,
                Number.EPSILON,
                ...run.queue.map((e) => e.score),
              )}
              threshold={run.model.threshold}
              multiView={multiView}
              reveal={reveal}
              selectedId={selected?.id ?? ""}
              onSelect={selectExam}
              positions={Object.fromEntries(
                ordered.map((exam, index) => [exam.id, index + 1]),
              )}
            />
            {!shown.length && (
              <div className="empty-state" role="status">
                <Icon name={simulating ? "pulse" : "search"} size={28} />
                <strong>
                  {simulating && !arrived.length
                    ? "Aguardando o primeiro exame"
                    : "Nenhum exame encontrado"}
                </strong>
                <p>
                  {simulating && !arrived.length
                    ? "A fila será atualizada a cada chegada."
                    : "Ajuste a busca ou o filtro para explorar a fila."}
                </p>
              </div>
            )}
            <div className="list-footer">
              <span>
                <span className="legend-line" /> Limiar de alerta:{" "}
                {fmtScore(run.model.threshold)}
              </span>
              <span role="status">
                {simulating
                  ? "Simulação em andamento"
                  : `${shown.length} de ${arrived.length} exames`}
              </span>
            </div>
            {multiView && (
              <p className="note shortcut-note">
                <b>Atalho da incidência.</b> Este modelo foi treinado com AP e
                PA. Na incidência AP o paciente está deitado ou sentado, e a
                maioria das opacidades do RSNA é AP: o score sobe com a
                incidência, não só com a doença.
                {test?.auc_by_view.AP !== undefined &&
                  ` No teste inteiro, a AUC geral é ${pt(test.roc_auc)}, mas dentro de cada incidência cai para ${pt(test.auc_by_view.AP)} (AP) e ${pt(test.auc_by_view.PA)} (PA): parte do acerto vem da incidência.`}
                {test?.auc_ap_vs_pa_normals !== undefined &&
                  ` Só entre exames normais, o score separa AP de PA com AUC ${pt(test.auc_ap_vs_pa_normals)}.`}
              </p>
            )}
          </div>

          {selected ? (
            <Viewer
              exam={selected}
              reveal={reveal}
              position={selectedIndex + 1}
              total={shown.length}
              onPrevious={previousExam}
              onNext={nextExam}
              onInspect={setInspection}
            />
          ) : (
            <div className="panel viewer viewer-empty">
              <Icon name="eye" size={36} />
              <h2>Visualizador de exames</h2>
              <p>
                Selecione um exame da fila para explorar a imagem e a
                reconstrução.
              </p>
            </div>
          )}
        </div>
      </div>
      <div className="explanation-strip">
        <Icon name="info" size={18} />
        <p>
          <b>Como ler esta fila?</b> O score mede a diferença entre o exame e
          sua reconstrução. Acima do limiar, o modelo sugere revisão. As
          prioridades são uma simulação para estudo.
        </p>
      </div>
      {test && (
        <details className="model-performance">
          <summary>Desempenho e limites deste modelo</summary>
          <dl>
            <div>
              <dt>AUC no teste</dt>
              <dd>{pt(test.roc_auc)}</dd>
            </div>
            <div>
              <dt>Sensibilidade no limiar</dt>
              <dd>{(test.sensitivity * 100).toFixed(1).replace(".", ",")}%</dd>
            </div>
            <div>
              <dt>Especificidade no limiar</dt>
              <dd>{(test.specificity * 100).toFixed(1).replace(".", ",")}%</dd>
            </div>
            <div>
              <dt>Exames de teste</dt>
              <dd>{run.model.n_test.toLocaleString("pt-BR")}</dd>
            </div>
          </dl>
          <p>
            O score não é uma probabilidade de pneumonia. “Padrão” indica apenas
            score abaixo do limiar. O mapa de calor não confirma a localização
            de uma lesão. Estes valores são do teste completo; os indicadores
            acima descrevem apenas a amostra da fila.
          </p>
        </details>
      )}
      {selected && inspection && (
        <ExamDialog
          exam={selected}
          mode={inspection}
          onModeChange={setInspection}
          onClose={() => setInspection(null)}
          reveal={reveal}
          sigma={run.model.sigma}
          threshold={run.model.threshold}
          imageSize={run.model.image_size}
          originalSize={run.model.original_image_size}
          normalized={run.model.normalize}
          lungMask={run.model.lung_mask}
          position={selectedIndex + 1}
          total={shown.length}
          arrivalPosition={
            arrivedExams.findIndex((exam) => exam.id === selected.id) + 1
          }
          vaePosition={
            vaeOrder.findIndex((exam) => exam.id === selected.id) + 1
          }
          onPrevious={previousExam}
          onNext={nextExam}
        />
      )}
    </>
  );
}
