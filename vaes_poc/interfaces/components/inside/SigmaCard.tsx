"use client";

import { useState } from "react";

import Boxes from "@/components/Boxes";
import type { VaeData } from "@/lib/types";

const W = 300;
const H = 70;

/** O mesmo exame com cada largura σ testada na calibração, e a AUC de validação de cada uma. */
export default function SigmaCard({ sigma }: { sigma: VaeData["sigma"] }) {
  const { example, steps, chosen } = sigma;
  const [i, setI] = useState(0);
  const chosenIdx = steps.findIndex((s) => s.sigma === chosen);
  const lo = Math.min(...steps.map((s) => s.auc_val)) - 0.02;
  const hi = Math.max(...steps.map((s) => s.auc_val));
  const bw = W / steps.length;

  return (
    <>
      <div className="sigma-layout">
        <div className="stage small">
          <img src={example.img} alt="Exame com opacidade" />
          <img src={steps[i].heat} className="layer pixel" alt="" />
          <Boxes boxes={example.boxes} />
        </div>
        <div className="sigma-side">
          <div className="big-number">
            <span>{steps[i].sigma}</span>
            <small>σ em pixels</small>
          </div>
          <div className="big-number">
            <span>{steps[i].auc_val.toFixed(3)}</span>
            <small>AUC na validação</small>
          </div>
          <svg className="bars" viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
            {steps.map((s, j) => {
              const h = 8 + ((s.auc_val - lo) / (hi - lo)) * (H - 22);
              const fill = j === i ? "var(--accent)" : j === chosenIdx ? "var(--muted)" : "var(--line)";
              return (
                <g key={s.sigma}>
                  <rect x={j * bw + 3} y={H - 12 - h} width={bw - 6} height={h} rx={3} fill={fill} />
                  <text x={j * bw + bw / 2} y={H - 1} textAnchor="middle" fontSize={9} fill="var(--faint)">
                    {s.sigma}
                    {j === chosenIdx ? "★" : ""}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <input
        type="range"
        className="full-range"
        min={0}
        max={steps.length - 1}
        value={i}
        onChange={(e) => setI(Number(e.target.value))}
        aria-label="Largura da suavização"
      />
    </>
  );
}
