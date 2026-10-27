"use client";

import { useState } from "react";

const decimal = (value: number) => value.toFixed(2).replace(".", ",");

/** Exemplo matemático de uma dimensão; não representa um exame do dataset. */
export default function LatentLesson() {
  const [mean, setMean] = useState(0);
  const [deviation, setDeviation] = useState(1);
  const [noise, setNoise] = useState(0.7);
  const sample = mean + deviation * noise;
  const sx = (x: number) => 30 + ((x + 6) / 12) * 540;
  const curve = Array.from({ length: 241 }, (_, i) => {
    const x = -6 + i * 0.05;
    const y =
      160 - (50 / deviation) * Math.exp(-0.5 * ((x - mean) / deviation) ** 2);
    return `${i ? "L" : "M"}${sx(x).toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");

  return (
    <div className="latent-lesson">
      <div className="distribution-plot">
        <span className="lesson-label">
          UMA DIMENSÃO LATENTE · EXEMPLO DIDÁTICO
        </span>
        <svg
          viewBox="0 0 600 200"
          role="img"
          aria-label={`Distribuição normal com média ${decimal(mean)}, desvio ${decimal(deviation)} e amostra z igual a ${decimal(sample)}`}
        >
          <line x1="30" x2="570" y1="160" y2="160" stroke="var(--line)" />
          {[-4, -2, 0, 2, 4].map((x) => (
            <g key={x}>
              <line
                x1={sx(x)}
                x2={sx(x)}
                y1="160"
                y2="165"
                stroke="var(--muted)"
              />
              <text
                x={sx(x)}
                y="184"
                textAnchor="middle"
                fill="var(--muted)"
                fontSize="12"
              >
                {x}
              </text>
            </g>
          ))}
          <path
            d={`${curve}L570,160L30,160Z`}
            fill="var(--accent)"
            fillOpacity="0.08"
          />
          <path
            d={curve}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="2.5"
          />
          <line
            x1={sx(mean)}
            x2={sx(mean)}
            y1="18"
            y2="160"
            stroke="var(--accent)"
            strokeDasharray="4 5"
          />
          <line
            x1={sx(sample)}
            x2={sx(sample)}
            y1="30"
            y2="160"
            stroke="var(--flag)"
            strokeWidth="2"
          />
          <circle cx={sx(sample)} cy="160" r="5" fill="var(--flag)" />
          <text
            x={sx(sample)}
            y="21"
            textAnchor="middle"
            fill="var(--flag)"
            fontSize="13"
          >
            z = {decimal(sample)}
          </text>
        </svg>
        <p>
          A curva mostra onde os códigos são mais prováveis. A linha amarela é
          uma amostra de <i>z</i>.
        </p>
      </div>
      <div className="distribution-controls">
        {[
          {
            label: "Média μ",
            id: "lesson-mean",
            min: -2,
            max: 2,
            value: mean,
            change: setMean,
          },
          {
            label: "Desvio padrão σ",
            id: "lesson-deviation",
            min: 0.4,
            max: 1.8,
            value: deviation,
            change: setDeviation,
          },
          {
            label: "Ruído ε",
            id: "lesson-noise",
            min: -2,
            max: 2,
            value: noise,
            change: setNoise,
          },
        ].map((control) => (
          <div className="distribution-control" key={control.id}>
            <label htmlFor={control.id}>
              {control.label}
              <output>{decimal(control.value)}</output>
            </label>
            <input
              id={control.id}
              type="range"
              min={control.min}
              max={control.max}
              step="0.1"
              value={control.value}
              onChange={(event) => control.change(Number(event.target.value))}
            />
          </div>
        ))}
        <div className="sample-equation">
          <span>REPARAMETRIZAÇÃO</span>
          <code>z = μ + σ × ε</code>
          <strong>
            {decimal(mean)} + {decimal(deviation)} × {decimal(noise)} ={" "}
            {decimal(sample)}
          </strong>
        </div>
        <p className="note">
          Aqui, você escolhe ε para entender a conta. No treino, ε é sorteado de
          N(0, 1). Na inferência deste projeto, usamos z = μ.
        </p>
      </div>
    </div>
  );
}
