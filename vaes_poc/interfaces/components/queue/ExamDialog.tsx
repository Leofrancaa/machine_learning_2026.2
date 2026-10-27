"use client";

import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import Boxes from "@/components/Boxes";
import Icon from "@/components/Icon";
import Segmented from "@/components/Segmented";
import ExamComparison from "@/components/queue/ExamComparison";
import DecisionExplanation from "@/components/queue/DecisionExplanation";
import type { Exam } from "@/lib/types";

export type InspectionMode = "analysis" | "image";

export function navigateExam(
  event: KeyboardEvent<HTMLElement>,
  position: number,
  total: number,
  previous: () => void,
  next: () => void,
) {
  const target = event.target as HTMLElement;
  if (
    event.altKey ||
    event.ctrlKey ||
    event.metaKey ||
    event.shiftKey ||
    target.closest("input, textarea, select, [contenteditable='true']")
  )
    return;
  if (event.key === "ArrowLeft" && position > 1) {
    event.preventDefault();
    previous();
  }
  if (event.key === "ArrowRight" && position < total) {
    event.preventDefault();
    next();
  }
}

export default function ExamDialog({
  exam,
  mode,
  onModeChange,
  onClose,
  reveal,
  sigma,
  threshold,
  imageSize,
  originalSize,
  normalized,
  lungMask,
  position,
  total,
  arrivalPosition,
  vaePosition,
  onPrevious,
  onNext,
}: {
  exam: Exam;
  mode: InspectionMode;
  onModeChange: (mode: InspectionMode) => void;
  onClose: () => void;
  reveal: boolean;
  sigma: number;
  threshold: number;
  imageSize?: number;
  originalSize?: number;
  normalized?: boolean;
  lungMask?: boolean;
  position: number;
  total: number;
  arrivalPosition: number;
  vaePosition: number;
  onPrevious: () => void;
  onNext: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    const returnFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    if (!dialog.open) dialog.showModal();
    return () => {
      dialog.close();
      document.body.style.overflow = overflow;
      if (returnFocus?.isConnected) returnFocus.focus({ preventScroll: true });
    };
  }, []);

  return (
    <dialog
      ref={ref}
      className="inspection-dialog"
      aria-labelledby="inspection-title"
      onCancel={onClose}
      onKeyDown={(event) =>
        navigateExam(event, position, total, onPrevious, onNext)
      }
      onClick={(event) => {
        if (event.target === ref.current) {
          const bounds = ref.current.getBoundingClientRect();
          if (
            event.clientX < bounds.left ||
            event.clientX > bounds.right ||
            event.clientY < bounds.top ||
            event.clientY > bounds.bottom
          )
            onClose();
        }
      }}
    >
      <header className="inspection-header">
        <div>
          <p className="eyebrow">ANÁLISE DO EXAME · {exam.view}</p>
          <h2 id="inspection-title">
            {exam.id}
            <span className={`pill ${exam.flagged ? "flag" : "ok"}`}>
              {exam.flagged ? "Acima do limiar" : "Abaixo do limiar"}
            </span>
          </h2>
        </div>
        <button
          className="icon-btn close-inspection"
          aria-label="Fechar análise"
          onClick={onClose}
        >
          <Icon name="close" />
        </button>
      </header>
      <div className="inspection-toolbar">
        <Segmented<InspectionMode>
          label="Modo de análise"
          value={mode}
          onChange={onModeChange}
          options={[
            { value: "analysis", label: "Comparação explicada" },
            { value: "image", label: "Imagem ampliada" },
          ]}
        />
        <div className="inspection-navigation">
          <button
            className="btn"
            disabled={position <= 1}
            aria-label="Exame anterior na análise"
            onClick={onPrevious}
          >
            <Icon
              name="chevron"
              size={16}
              style={{ transform: "rotate(180deg)" }}
            />
          </button>
          <span aria-live="polite">
            {position} / {total}
          </span>
          <button
            className="btn"
            disabled={position >= total}
            aria-label="Próximo exame na análise"
            onClick={onNext}
          >
            <Icon name="chevron" size={16} />
          </button>
        </div>
      </div>
      <div className="inspection-body">
        {mode === "analysis" ? (
          <ExamComparison
            exam={exam}
            sigma={sigma}
            threshold={threshold}
            imageSize={imageSize}
            originalSize={originalSize}
            normalized={normalized}
            lungMask={lungMask}
          />
        ) : (
          <ZoomImage key={exam.id} exam={exam} reveal={reveal} />
        )}
        <DecisionExplanation exam={exam} reveal={reveal} />
      </div>
      <footer className="inspection-footer">
        <p>
          <strong>Na fila:</strong> posição {arrivalPosition} na chegada →{" "}
          {vaePosition} pelo VAE.
        </p>
        <span>← → navegam entre exames · Esc fecha</span>
      </footer>
    </dialog>
  );
}

function ZoomImage({ exam, reveal }: { exam: Exam; reveal: boolean }) {
  const [zoom, setZoom] = useState(1);
  const [image, setImage] = useState<"original" | "rec" | "heat">("original");
  const [boxes, setBoxes] = useState(false);
  const viewport = useRef<HTMLDivElement>(null);
  function reset() {
    setZoom(1);
    viewport.current?.scrollTo({ top: 0, left: 0 });
  }
  return (
    <div className="zoom-experience">
      <div className="zoom-controls">
        <Segmented
          label="Imagem ampliada"
          value={image}
          onChange={setImage}
          options={[
            { value: "original", label: "Original" },
            { value: "rec", label: "Reconstrução" },
            { value: "heat", label: "Mapa de calor" },
          ]}
        />
        <div className="zoom-buttons">
          <button
            className="icon-btn"
            aria-label="Diminuir zoom"
            disabled={zoom <= 1}
            onClick={() => setZoom(Math.max(1, zoom - 0.25))}
          >
            <Icon name="minus" size={16} />
          </button>
          <output aria-label="Nível de zoom">{Math.round(zoom * 100)}%</output>
          <button
            className="icon-btn"
            aria-label="Aumentar zoom"
            disabled={zoom >= 3}
            onClick={() => setZoom(Math.min(3, zoom + 0.25))}
          >
            <Icon name="plus" size={16} />
          </button>
          <button className="btn" onClick={reset}>
            Ajustar
          </button>
        </div>
      </div>
      <div
        className="zoom-viewport"
        ref={viewport}
        tabIndex={0}
        role="region"
        aria-label="Imagem com zoom, use as barras de rolagem para explorar"
        onKeyDown={(event) => event.stopPropagation()}
      >
        <div className="zoom-canvas" style={{ width: `${zoom * 100}%` }}>
          <img
            src={image === "rec" ? exam.rec : exam.img}
            alt={`${image === "rec" ? "Reconstrução" : "Radiografia"} ${exam.id} ampliada`}
          />
          {image === "heat" && (
            <img
              className="layer"
              src={exam.heat}
              alt=""
              style={{ opacity: 0.7 }}
            />
          )}
          <Boxes
            boxes={exam.boxes}
            visible={boxes && reveal && image !== "rec"}
          />
        </div>
      </div>
      <div className="zoom-caption">
        <label className="switch">
          <input
            type="checkbox"
            checked={boxes && reveal && image !== "rec"}
            disabled={!reveal || image === "rec"}
            onChange={(event) => setBoxes(event.target.checked)}
          />
          <span>Caixas do laudo</span>
        </label>
        <p>
          Zoom de apresentação: amplia os pixels disponíveis, sem recuperar
          detalhe além da resolução dos arquivos exportados.
        </p>
      </div>
    </div>
  );
}
