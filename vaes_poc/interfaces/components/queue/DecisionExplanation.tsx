import type { Exam } from "@/lib/types";

export default function DecisionExplanation({
  exam,
  reveal,
}: {
  exam: Exam;
  reveal: boolean;
}) {
  if (!reveal) return null;
  const opacity = exam.label === "opacidade";
  const correct = opacity === exam.flagged;
  const title = opacity
    ? exam.flagged
      ? "Opacidade sinalizada"
      : "Falso negativo neste limiar"
    : exam.flagged
      ? "Falso alerta neste limiar"
      : "Normal abaixo do limiar";
  const text = opacity
    ? exam.flagged
      ? "O rótulo indica opacidade e o score atingiu o limiar. É um acerto de detecção neste conjunto; a posição não mede gravidade."
      : "O rótulo indica opacidade, mas o score não atingiu o limiar. A alteração não elevou suficientemente o erro médio suavizado para gerar um alerta."
    : exam.flagged
      ? "O rótulo indica normal, mas a diferença de reconstrução atingiu o limiar. O modelo sinalizou anomalia; isso não confirma doença."
      : "O rótulo indica normal e o score ficou abaixo do limiar. É um acerto neste conjunto, sem garantir desempenho em outros exames.";
  return (
    <aside
      className={`decision-reading ${correct ? "agrees" : "disagrees"}`}
      aria-label="Leitura da decisão versus laudo"
    >
      <strong>{title}</strong>
      <p>{text}</p>
    </aside>
  );
}
