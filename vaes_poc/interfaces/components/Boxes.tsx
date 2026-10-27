import type { Box } from "@/lib/types";

/** Caixas do laudo sobre o exame (coordenadas em % da imagem). */
export default function Boxes({ boxes, visible = true }: { boxes: Box[]; visible?: boolean }) {
  return (
    <div className="layer" style={{ opacity: visible ? 1 : 0 }}>
      {boxes.map(([x, y, w, h], i) => (
        <div key={i} className="box" style={{ left: `${x}%`, top: `${y}%`, width: `${w}%`, height: `${h}%` }} />
      ))}
    </div>
  );
}
