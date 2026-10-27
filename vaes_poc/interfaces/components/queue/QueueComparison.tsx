"use client";

import Icon from "@/components/Icon";
import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import LabelTag from "@/components/LabelTag";
import { byScore, fmtScore } from "@/lib/format";
import type { Exam } from "@/lib/types";

export default function QueueComparison({
  exams,
  selectedId,
  reveal,
  onSelect,
  latestId,
}: {
  exams: Exam[];
  selectedId?: string;
  reveal: boolean;
  onSelect?: (id: string) => void;
  latestId?: string;
}) {
  const ranked = byScore(exams);
  const arrivalPosition = exams.findIndex((exam) => exam.id === selectedId) + 1;
  const vaePosition = ranked.findIndex((exam) => exam.id === selectedId) + 1;
  const movement = arrivalPosition - vaePosition;
  const maxScore = Math.max(Number.EPSILON, ...exams.map((exam) => exam.score));
  const normalIndex = ranked.findIndex(
    (exam, index) =>
      exam.label === "normal" &&
      ranked.slice(index + 1).some((other) => other.label === "opacidade"),
  );
  const normalAhead = ranked[normalIndex];
  const opacityBehind =
    normalIndex >= 0
      ? ranked.slice(normalIndex + 1).find((exam) => exam.label === "opacidade")
      : undefined;
  const root = useRef<HTMLElement>(null);
  const markerId = useId();
  const [connection, setConnection] = useState<{
    width: number;
    height: number;
    left: number;
    right: number;
    from: number;
    to: number;
  } | null>(null);
  const previousTops = useRef(new Map<string, number>());
  const fingerprint = `${exams.map((exam) => exam.id).join()}|${ranked.map((exam) => exam.id).join()}`;
  useLayoutEffect(() => {
    const next = new Map<string, number>();
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    root.current
      ?.querySelectorAll<HTMLElement>("li[data-exam]")
      .forEach((row) => {
        const key = `${row.closest("ol")?.dataset.order}:${row.dataset.exam}`;
        const top = row.offsetTop;
        const before = previousTops.current.get(key);
        next.set(key, top);
        if (!reduceMotion && before !== undefined && Math.abs(before - top) > 1)
          row.animate(
            [
              { transform: `translateY(${before - top}px)` },
              { transform: "none" },
            ],
            { duration: 420, easing: "cubic-bezier(.2,.8,.2,1)" },
          );
      });
    previousTops.current = next;
  }, [fingerprint]);
  useEffect(() => {
    const lists = root.current?.querySelectorAll<HTMLOListElement>(
      ".comparison-list ol",
    );
    function centerSelection() {
      lists?.forEach((list) => {
        const item = list
          .querySelector<HTMLElement>("[aria-pressed='true'], .is-selected")
          ?.closest("li");
        if (item && list.clientHeight)
          list.scrollTop =
            item.offsetTop - (list.clientHeight - item.clientHeight) / 2;
      });
    }
    const observer = new ResizeObserver(centerSelection);
    lists?.forEach((list) => observer.observe(list));
    centerSelection();
    return () => observer.disconnect();
  }, [selectedId, exams.length]);
  useEffect(() => {
    const grid = root.current?.querySelector<HTMLElement>(
      ".queue-comparison-grid",
    );
    const lists = grid?.querySelectorAll<HTMLOListElement>("ol");
    if (!grid || !lists || lists.length !== 2) return;
    let frame = 0;
    function measure() {
      if (!grid || !lists) return;
      const bounds = grid.getBoundingClientRect();
      const anchors = Array.from(lists).map((list) => {
        const item = list
          .querySelector<HTMLElement>("[aria-pressed='true'], .is-selected")
          ?.closest("li");
        const box = list.getBoundingClientRect();
        if (!item || !box.height) return null;
        const y =
          box.top + item.offsetTop - list.scrollTop + item.offsetHeight / 2;
        return y >= box.top && y <= box.bottom
          ? { y: y - bounds.top, box }
          : null;
      });
      if (!anchors[0] || !anchors[1]) return setConnection(null);
      setConnection({
        width: bounds.width,
        height: bounds.height,
        left: anchors[0].box.right - bounds.left - 7,
        right: anchors[1].box.left - bounds.left + 7,
        from: anchors[0].y,
        to: anchors[1].y,
      });
    }
    function schedule() {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    }
    const observer = new ResizeObserver(schedule);
    observer.observe(grid);
    lists.forEach((list) => {
      observer.observe(list);
      list.addEventListener("scroll", schedule, { passive: true });
    });
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      lists.forEach((list) => list.removeEventListener("scroll", schedule));
    };
  }, [selectedId, fingerprint, reveal]);
  const lists = [
    {
      key: "arrival",
      title: "Ordem de chegada",
      description: "Quem chegou primeiro",
      items: exams,
    },
    {
      key: "vae",
      title: "Ordem pelo VAE",
      description: "Maior score primeiro",
      items: ranked,
    },
  ];

  return (
    <section
      ref={root}
      className="panel queue-comparison"
      aria-label="Comparação das filas"
    >
      <div className="panel-head">
        <div>
          <h2>O que muda na fila?</h2>
          <p className="sub">
            {onSelect
              ? "Os mesmos exames, em duas ordens. Selecione um código para acompanhar sua posição."
              : "Os mesmos exames, em duas ordens. O destaque acompanha o exemplo desta explicação."}
          </p>
        </div>
        <span className="version-tag">
          {exams.length} EXAMES QUE JÁ CHEGARAM
        </span>
      </div>
      <div className="queue-comparison-grid">
        {lists.map(({ key, title, description, items }) => (
          <div key={key} className={`comparison-list ${key}`}>
            <div className="comparison-list-heading">
              <span>
                <Icon name={key === "vae" ? "spark" : "queue"} size={16} />
                {title}
              </span>
              <small>{description}</small>
            </div>
            <ol aria-label={title} data-order={key}>
              {items.map((exam, index) => (
                <li
                  key={exam.id}
                  data-exam={exam.id}
                  className={
                    exam.id === latestId ? "latest-arrival" : undefined
                  }
                >
                  {onSelect ? (
                    <button
                      aria-label={`${title}: ${exam.id}, posição ${index + 1}`}
                      aria-pressed={selectedId === exam.id}
                      onClick={() => onSelect(exam.id)}
                    >
                      <span className="comparison-rank">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <img
                        src={exam.img}
                        width={26}
                        height={26}
                        alt=""
                        loading="lazy"
                      />
                      <span className="comparison-exam-id">
                        {exam.id}
                        {exam.id === latestId && (
                          <em className="arrival-chip">Novo</em>
                        )}
                      </span>
                      <span className="comparison-exam-detail">
                        {reveal ? (
                          <LabelTag label={exam.label} />
                        ) : key === "vae" ? (
                          `${fmtScore(exam.score)}`
                        ) : (
                          exam.arrival
                        )}
                      </span>
                      {key === "vae" && (
                        <span
                          className="comparison-score-track"
                          aria-hidden="true"
                        >
                          <i
                            className={exam.flagged ? "over" : undefined}
                            style={{
                              width: `${(exam.score / maxScore) * 100}%`,
                            }}
                          />
                        </span>
                      )}
                    </button>
                  ) : (
                    <div
                      className={`comparison-static-row ${selectedId === exam.id ? "is-selected" : ""}`}
                    >
                      <span className="comparison-rank">{index + 1}</span>
                      <span className="comparison-exam-id">{exam.id}</span>
                      <span>
                        {key === "vae" ? fmtScore(exam.score) : exam.arrival}
                      </span>
                    </div>
                  )}
                </li>
              ))}
            </ol>
            {!items.length && (
              <p className="comparison-waiting">
                Aguardando a chegada dos exames.
              </p>
            )}
          </div>
        ))}
        {connection && arrivalPosition > 0 && vaePosition > 0 && (
          <svg
            className="queue-connection"
            aria-hidden="true"
            viewBox={`0 0 ${connection.width} ${connection.height}`}
          >
            <defs>
              <marker
                id={markerId}
                markerWidth="6"
                markerHeight="6"
                refX="5"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L6,3 L0,6" fill="var(--accent)" />
              </marker>
            </defs>
            <path
              d={`M${connection.left},${connection.from} C${(connection.left + connection.right) / 2},${connection.from} ${(connection.left + connection.right) / 2},${connection.to} ${connection.right},${connection.to}`}
              markerEnd={`url(#${markerId})`}
            />
            <circle cx={connection.left} cy={connection.from} r="3" />
          </svg>
        )}
      </div>
      {arrivalPosition > 0 && vaePosition > 0 && (
        <div className="movement-explanation" role="status">
          <Icon name="arrow" size={20} />
          <p>
            <strong>{selectedId}</strong>: posição <b>{arrivalPosition}</b> na
            chegada <span aria-hidden="true">→</span> posição{" "}
            <b>{vaePosition}</b> pelo VAE.{" "}
            <span>
              {movement > 0
                ? `Avançou ${movement} ${movement === 1 ? "posição" : "posições"}.`
                : movement < 0
                  ? `Recuou ${Math.abs(movement)} ${movement === -1 ? "posição" : "posições"}.`
                  : "Manteve a posição."}
            </span>
          </p>
        </div>
      )}
      <div className="queue-visual-legend">
        <span>
          <i className="legend-selected" />
          Exame acompanhado
        </span>
        {latestId && (
          <span>
            <i className="legend-new" />
            Última chegada
          </span>
        )}
        <span>Score × 1.000 · barras relativas à amostra</span>
      </div>
      <p className="comparison-footnote">
        A ordem usa o score, sem consultar os rótulos. Posição na simulação não
        mede urgência clínica nem minutos de espera. A busca e o filtro afetam
        apenas a lista de trabalho abaixo.
      </p>
      {reveal && normalAhead && opacityBehind && (
        <details className="ranking-reading">
          <summary>Por que um normal aparece antes de uma opacidade?</summary>
          <p>
            <b>{normalAhead.id}</b> é normal no rótulo, mas tem score{" "}
            <b>{fmtScore(normalAhead.score)}</b>. <b>{opacityBehind.id}</b> tem
            opacidade, com score <b>{fmtScore(opacityBehind.score)}</b>. A fila
            coloca o maior erro médio de reconstrução primeiro; ela não consulta
            os rótulos para ordenar.
          </p>
          <p>
            Diferenças de anatomia e aquisição podem elevar o score de um
            normal. Uma alteração localizada ou parcialmente reconstruída pode
            ter pouco peso na média do mapa. Essas são possibilidades; o score
            não identifica a causa exata deste par.
          </p>
          <p>
            <b>
              O limiar define os alertas. Alterar somente o limiar não muda a
              ordem.
            </b>{" "}
            Os rótulos servem para medir acertos, falsos alertas e falsos
            negativos.
          </p>
        </details>
      )}
    </section>
  );
}
