"use client";

import { memo, useMemo, useRef, useState } from "react";

import type { LatentData, LatentPoint, PointClass } from "@/lib/types";

const CLASSES: { key: PointClass; name: string; color: string }[] = [
  { key: "normal", name: "normal", color: "var(--normal)" },
  { key: "outra", name: "anormal sem opacidade", color: "var(--other)" },
  { key: "opacidade", name: "opacidade", color: "var(--positive)" },
];
const W = 600;
const H = 330;
const PAD = 16;
const THUMB = 96;

// Os ~2 mil pontos não mudam: ficam num componente memorizado, para o tooltip não os redesenhar.
const Points = memo(function Points({ points, hidden }: { points: LatentPoint[]; hidden: Set<PointClass> }) {
  const { sx, sy } = useMemo(() => {
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    return {
      sx: (x: number) => PAD + ((x - x0) / (x1 - x0)) * (W - 2 * PAD),
      sy: (y: number) => H - PAD - ((y - y0) / (y1 - y0)) * (H - 2 * PAD),
    };
  }, [points]);

  return (
    <>
      {CLASSES.map((c) => (
        <g
          key={c.key}
          fill={c.color}
          fillOpacity={c.key === "opacidade" ? 0.9 : 0.55}
          style={{ display: hidden.has(c.key) ? "none" : undefined }}
        >
          {points.map(([x, y, cls], k) =>
            cls === c.key ? (
              <circle key={k} cx={sx(x).toFixed(1)} cy={sy(y).toFixed(1)} r={cls === "opacidade" ? 2.6 : 2.1} data-k={k} />
            ) : null
          )}
        </g>
      ))}
    </>
  );
});

export default function LatentMap({ latent }: { latent: LatentData }) {
  const [hidden, setHidden] = useState<Set<PointClass>>(new Set());
  const [tip, setTip] = useState<{ k: number; left: number; top: number } | null>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const cols = latent.atlas_cols;

  function toggle(key: PointClass) {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function onMove(ev: React.MouseEvent<SVGSVGElement>) {
    const k = (ev.target as SVGElement).dataset?.k;
    if (k === undefined || !wrap.current) return setTip(null);
    const box = wrap.current.getBoundingClientRect();
    setTip({
      k: Number(k),
      left: Math.min(ev.clientX - box.left + 12, box.width - 112),
      top: ev.clientY - box.top - 120,
    });
  }

  return (
    <>
      <div className="legend">
        {CLASSES.map((c) => (
          <button key={c.key} className={`chip ${hidden.has(c.key) ? "off" : ""}`} onClick={() => toggle(c.key)}>
            <i style={{ background: c.color }} />
            {c.name}
          </button>
        ))}
      </div>
      <div className="scatter-wrap" ref={wrap}>
        <svg
          className="scatter"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="Exames no espaço latente"
          onMouseMove={onMove}
          onMouseLeave={() => setTip(null)}
        >
          <Points points={latent.points} hidden={hidden} />
        </svg>
        {tip && (
          <div className="tooltip" style={{ display: "block", left: tip.left, top: tip.top }}>
            <div
              className="thumb"
              style={{
                backgroundImage: `url(${latent.atlas})`,
                backgroundSize: `${cols * THUMB}px ${cols * THUMB}px`,
                backgroundPosition: `${-(tip.k % cols) * THUMB}px ${-Math.floor(tip.k / cols) * THUMB}px`,
              }}
            />
            <span>{CLASSES.find((c) => c.key === latent.points[tip.k][2])?.name}</span>
          </div>
        )}
      </div>
    </>
  );
}
