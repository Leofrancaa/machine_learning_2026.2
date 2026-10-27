"use client";

import { useState } from "react";

import type { LatentData } from "@/lib/types";

/** Pontos do caminho em linha reta entre os μ de dois exames, passados pelo decoder. */
export default function Interpolation({ interpolation }: { interpolation: LatentData["interpolation"] }) {
  const { from, to, sprite, steps } = interpolation;
  const [step, setStep] = useState(0);

  return (
    <>
      <div className="interp-layout">
        <figure>
          <img src={from.img} alt="" />
          <figcaption>
            {from.id} · {from.label}
          </figcaption>
        </figure>
        <div
          className="interp-frame"
          role="img"
          aria-label="Ponto intermediário"
          style={{
            backgroundImage: `url(${sprite})`,
            backgroundSize: `${steps * 100}% 100%`,
            backgroundPosition: `${(step / (steps - 1)) * 100}% 0`,
          }}
        />
        <figure>
          <img src={to.img} alt="" />
          <figcaption>
            {to.id} · {to.label}
          </figcaption>
        </figure>
      </div>
      <input
        type="range"
        className="full-range"
        min={0}
        max={steps - 1}
        value={step}
        onChange={(e) => setStep(Number(e.target.value))}
        aria-label="Posição na interpolação"
      />
    </>
  );
}
