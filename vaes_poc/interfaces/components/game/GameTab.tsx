"use client";

import { useState } from "react";

import SiteQr from "@/components/game/SiteQr";
import Icon from "@/components/Icon";
import LabelTag from "@/components/LabelTag";
import { byScore, fmtScore } from "@/lib/format";
import type { Exam } from "@/lib/types";

/** A plateia ordena os exames por prioridade e compara com o VAE e com o laudo. */
export default function GameTab({ exams }: { exams: Exam[] }) {
  const [picks, setPicks] = useState<string[]>([]);
  const [revealed, setRevealed] = useState(false);

  function pick(id: string) {
    if (revealed) return;
    setPicks((prev) =>
      prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id],
    );
  }

  function reset() {
    setPicks([]);
    setRevealed(false);
  }

  return (
    <div className="game-layout">
      <div className="panel">
        <div className="panel-head">
          <div>
            <h2>Quem você laudaria primeiro?</h2>
            <p className="sub">
              Clique nos {exams.length} exames na ordem de prioridade (1 = mais
              urgente). Depois compare com o VAE e com o laudo.
            </p>
          </div>
          <div className="controls">
            <button className="btn" onClick={reset}>
              Limpar
            </button>
            <button
              className="btn primary"
              onClick={() => setRevealed(true)}
              disabled={picks.length !== exams.length}
            >
              Revelar
            </button>
          </div>
        </div>
        <div className="game-grid">
          {exams.map((exam) => {
            const pos = picks.indexOf(exam.id);
            return (
              <button
                key={exam.id}
                className={`game-card ${pos >= 0 ? "picked" : ""} ${revealed ? "revealed" : ""}`}
                onClick={() => pick(exam.id)}
                aria-pressed={pos >= 0}
                aria-label={`${exam.id}${pos >= 0 ? `, prioridade ${pos + 1}` : ", não selecionado"}`}
              >
                <img src={exam.img} alt={`Exame ${exam.id}`} />
                <span className="order">{pos >= 0 ? pos + 1 : ""}</span>
                <span className="reveal">
                  <LabelTag label={exam.label} />
                  <span>
                    VAE: <b>{fmtScore(exam.score)}</b>
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <aside className="panel game-side">
        <span className="challenge-icon">
          <Icon name="spark" size={24} />
        </span>
        <h3>Sua ordem de prioridade</h3>
        <p>
          Selecione todos os exames, começando por aquele que você revisaria
          primeiro. Clique de novo para remover uma escolha.
        </p>
        <div className="challenge-progress">
          <span role="status">
            {picks.length} de {exams.length} selecionados
          </span>
          <progress
            max={exams.length}
            value={picks.length}
            aria-label="Exames selecionados"
          />
        </div>
        <p className="note">
          Ao revelar, compare quantas opacidades aparecem no início de cada
          fila. Este desafio é local, sem votação em tempo real.
        </p>
        <h3 className="qr-title">Jogue pelo celular</h3>
        <SiteQr />
        <p>
          Escaneie o QR code para abrir este desafio no seu celular, monte a sua
          ordem e compare com o VAE.
        </p>
        {revealed && <Results exams={exams} picks={picks} />}
      </aside>
    </div>
  );
}

function Results({ exams, picks }: { exams: Exam[]; picks: string[] }) {
  const byId = Object.fromEntries(exams.map((e) => [e.id, e]));
  const yours = picks.map((id) => byId[id]);
  const vae = byScore(exams);
  const nPos = exams.filter((e) => e.label === "opacidade").length;
  const hits = (list: Exam[]) =>
    list.slice(0, nPos).filter((e) => e.label === "opacidade").length;
  const list = (items: Exam[]) => (
    <ol>
      {items.map((e) => (
        <li key={e.id} className={e.label}>
          {e.id} · {e.label}
        </li>
      ))}
    </ol>
  );

  return (
    <div className="results">
      <h4>Opacidades nas {nPos} primeiras posições</h4>
      <div className="score-line">
        <span>Você</span>
        <b>
          {hits(yours)} de {nPos}
        </b>
      </div>
      <div className="score-line">
        <span>VAE</span>
        <b>
          {hits(vae)} de {nPos}
        </b>
      </div>
      <div className="order-lists">
        <div>Sua ordem{list(yours)}</div>
        <div>Ordem do VAE{list(vae)}</div>
      </div>
    </div>
  );
}
