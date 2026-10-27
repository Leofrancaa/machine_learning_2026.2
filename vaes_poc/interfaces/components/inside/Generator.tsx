"use client";

import { useState } from "react";

import type { LatentData } from "@/lib/types";

/**
 * Sliders do código z. Cada eixo tem um sprite com o decoder aplicado a 9 valores (de −3
 * a +3, com os outros eixos em zero); mexer num eixo zera os demais.
 */
export default function Generator({ latent, latentDim }: { latent: LatentData; latentDim: number }) {
  const n = latent.values.length;
  const middle = (n - 1) / 2;
  const [active, setActive] = useState<{ dim: number; step: number } | null>(null);
  const dim = active?.dim ?? 0;
  const step = active?.step ?? middle;

  return (
    <div className="gen-layout">
      <div
        className="gen-image"
        role="img"
        aria-label="Raio-X gerado pelo decoder"
        style={{
          backgroundImage: `url(${latent.dims[dim]})`,
          backgroundSize: `${n * 100}% 100%`,
          backgroundPosition: `${(step / (n - 1)) * 100}% 0`,
        }}
      />
      <div className="sliders">
        {Array.from({ length: latentDim }, (_, d) => {
          const value = active?.dim === d ? active.step : middle;
          return (
            <div key={d} className={`slider-row ${active?.dim === d ? "active" : ""}`}>
              <label htmlFor={`z${d}`}>z{d + 1}</label>
              <input
                id={`z${d}`}
                type="range"
                min={0}
                max={n - 1}
                value={value}
                onChange={(e) => setActive({ dim: d, step: Number(e.target.value) })}
              />
              <output>{latent.values[value].toFixed(1)}</output>
            </div>
          );
        })}
      </div>
    </div>
  );
}
