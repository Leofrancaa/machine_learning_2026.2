"use client";

import { useState } from "react";

import Boxes from "@/components/Boxes";
import LabelTag from "@/components/LabelTag";
import Segmented from "@/components/Segmented";
import Icon from "@/components/Icon";
import { fmtScore } from "@/lib/format";
import type { Exam } from "@/lib/types";
import DecisionExplanation from "@/components/queue/DecisionExplanation";
import { navigateExam } from "@/components/queue/ExamDialog";
import type { InspectionMode } from "@/components/queue/ExamDialog";

type Mode = "original" | "rec" | "swipe";
const MODE_TAG: Record<Mode, string> = {
  original: "original",
  rec: "reconstrução pelo VAE",
  swipe: "original | reconstrução",
};

/** O exame selecionado na fila: original, reconstrução, mapa de calor e caixas do laudo. */
export default function Viewer({
  exam,
  reveal,
  position,
  total,
  onPrevious,
  onNext,
  onInspect,
}: {
  exam: Exam;
  reveal: boolean;
  position: number;
  total: number;
  onPrevious: () => void;
  onNext: () => void;
  onInspect: (mode: InspectionMode) => void;
}) {
  const [mode, setMode] = useState<Mode>("original");
  const [swipe, setSwipe] = useState(50);
  const [heatOn, setHeatOn] = useState(false);
  const [heatOpacity, setHeatOpacity] = useState(60);
  const [boxesOn, setBoxesOn] = useState(false);

  return (
    <div
      className={`panel viewer ${mode === "swipe" ? "swipe" : ""}`}
      tabIndex={0}
      role="region"
      aria-label="Visualizador do exame selecionado"
      onKeyDown={(event) =>
        navigateExam(event, position, total, onPrevious, onNext)
      }
    >
      <div className="panel-head">
        <div>
          <p className="viewer-eyebrow">VISUALIZADOR</p>
          <h2>
            {exam.id}
            <span className="view-tag">{exam.view}</span>
          </h2>
        </div>
        <span className={`pill ${exam.flagged ? "flag" : "ok"}`}>
          {exam.flagged ? "Revisar com prioridade" : "Dentro do padrão"}
        </span>
      </div>
      <div className="inspect-actions">
        <button className="btn primary" onClick={() => onInspect("analysis")}>
          <Icon name="layers" size={16} />
          Analisar lado a lado
        </button>
        <button className="btn" onClick={() => onInspect("image")}>
          <Icon name="expand" size={16} />
          Ampliar
        </button>
      </div>
      <div className="viewer-mode">
        <Segmented<Mode>
          label="Imagem"
          value={mode}
          onChange={setMode}
          options={[
            { value: "original", label: "Original" },
            { value: "rec", label: "Reconstrução" },
            { value: "swipe", label: "Comparar" },
          ]}
        />
        <div className="viewer-navigation">
          <button
            className="icon-btn previous"
            aria-label="Exame anterior"
            onClick={onPrevious}
            disabled={position <= 1}
          >
            <Icon name="chevron" size={16} />
          </button>
          <span>
            {position}/{total}
          </span>
          <button
            className="icon-btn"
            aria-label="Próximo exame"
            onClick={onNext}
            disabled={position >= total}
          >
            <Icon name="chevron" size={16} />
          </button>
        </div>
      </div>
      <div className="stage">
        <img src={exam.img} alt="Raio-X original" />
        <img
          src={exam.rec}
          className="layer pixel"
          alt=""
          style={{
            opacity: mode === "original" ? 0 : 1,
            clipPath: mode === "swipe" ? `inset(0 0 0 ${swipe}%)` : "none",
          }}
        />
        <img
          src={exam.heat}
          className="layer pixel"
          alt=""
          style={{ opacity: heatOn ? heatOpacity / 100 : 0 }}
        />
        <Boxes boxes={exam.boxes} visible={boxesOn && reveal} />
        <div className="swipe-line" style={{ left: `${swipe}%` }} />
        <span className="stage-tag">{MODE_TAG[mode]}</span>
        <span className="stage-corner">
          {exam.id} · {exam.view}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={swipe}
        onChange={(e) => setSwipe(Number(e.target.value))}
        className="swipe-range"
        aria-label="Comparar original e reconstrução"
      />
      <div className="viewer-controls">
        <label className="switch">
          <input
            type="checkbox"
            checked={heatOn}
            onChange={(e) => setHeatOn(e.target.checked)}
          />
          <span>Mapa de calor</span>
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={heatOpacity}
          disabled={!heatOn}
          onChange={(e) => setHeatOpacity(Number(e.target.value))}
          aria-label="Opacidade do mapa"
        />
        <label
          className="switch"
          title={
            !reveal
              ? "Ative Laudo na fila para revelar as anotações"
              : undefined
          }
        >
          <input
            type="checkbox"
            checked={boxesOn && reveal}
            disabled={!reveal}
            onChange={(e) => setBoxesOn(e.target.checked)}
          />
          <span>Caixas do laudo</span>
        </label>
      </div>
      {heatOn && (
        <div className="heat-legend">
          <span>Menor diferença</span>
          <span className="heat-scale" />
          <span>Maior diferença</span>
        </div>
      )}
      <dl className="facts">
        <div>
          <dt>Score × 1.000</dt>
          <dd>{fmtScore(exam.score)}</dd>
        </div>
        <div>
          <dt>Mais anômalo que</dt>
          <dd>{Math.round(exam.percentile * 100)}% dos normais</dd>
        </div>
        <div>
          <dt>Incidência</dt>
          <dd>{exam.view}</dd>
        </div>
        <div>
          <dt>Laudo</dt>
          <dd>{reveal ? <LabelTag label={exam.label} /> : "oculto"}</dd>
        </div>
      </dl>
      <DecisionExplanation exam={exam} reveal={reveal} />
      <p className="hint">
        A <b>reconstrução</b> vem do padrão aprendido com exames normais. O{" "}
        <b>mapa de calor</b> mostra diferenças, que também podem refletir
        anatomia e aquisição da imagem.
      </p>
      <p className="viewer-keyboard-hint">
        Use ← → com o foco no visualizador para navegar.
      </p>
    </div>
  );
}
