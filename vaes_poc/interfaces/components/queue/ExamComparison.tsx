import { fmtScore } from "@/lib/format";
import type { Exam } from "@/lib/types";

export default function ExamComparison({
  exam,
  sigma,
  threshold,
  imageSize,
  originalSize,
  normalized,
  lungMask,
}: {
  exam: Exam;
  sigma: number;
  threshold: number;
  imageSize?: number;
  originalSize?: number;
  normalized?: boolean;
  lungMask?: boolean;
}) {
  const cards = [
    {
      title: "O exame original",
      detail: "1 · ENTRADA",
      image: exam.img,
      alt: `Radiografia original ${exam.id}`,
      caption: imageSize
        ? `${normalized ? "A imagem exibida já tem intensidade normalizada entre os percentis 1 e 99. " : ""}${originalSize ? `Imagem apresentada: ${originalSize} × ${originalSize} px. ` : ""}O modelo recebe uma versão de ${imageSize} × ${imageSize} px.`
        : "A imagem original é reduzida e normalizada antes de entrar no modelo. A resolução não foi incluída nesta exportação.",
    },
    {
      title: "O que o VAE reconstruiu",
      detail: "2 · RECONSTRUÇÃO",
      image: exam.rec,
      alt: `Reconstrução pelo VAE de ${exam.id}`,
      caption:
        "O decoder reconstrói a imagem a partir do código latente. Foi treinado apenas com exames normais.",
    },
    {
      title: "Onde aparece a diferença",
      detail: "3 · MAPA DE ANOMALIA",
      image: exam.img,
      alt: `Mapa de diferenças de ${exam.id}`,
      caption: `O resíduo é suavizado com σ = ${sigma} px e elevado ao quadrado.${lungMask ? " O mapa é zerado fora da máscara pulmonar média." : ""} A cor destaca diferenças maiores.`,
    },
  ];
  return (
    <div className="exam-comparison">
      <div className="exam-triptych">
        {cards.map((card, index) => (
          <figure key={card.detail}>
            <div className="comparison-step-label">{card.detail}</div>
            <div className="comparison-image">
              <img src={card.image} alt={card.alt} loading="lazy" />
              {index === 2 && (
                <img className="comparison-heat" src={exam.heat} alt="" />
              )}
            </div>
            <figcaption>
              <h3>{card.title}</h3>
              <p>{card.caption}</p>
            </figcaption>
          </figure>
        ))}
      </div>
      <div className="score-equation">
        <div>
          <span>RESÍDUO</span>
          <code>x − x̂</code>
        </div>
        <span aria-hidden="true">→</span>
        <div>
          <span>MAPA</span>
          <code>{lungMask ? "M ⊙ [Gσ ∗ (x − x̂)]²" : "[Gσ ∗ (x − x̂)]²"}</code>
        </div>
        <span aria-hidden="true">→</span>
        <div>
          <span>MÉDIA DO MAPA × 1.000</span>
          <strong>{fmtScore(exam.score)}</strong>
        </div>
      </div>
      <p className="score-reading">
        Neste exame, o score é{" "}
        <b>{exam.score >= threshold ? "maior ou igual" : "menor"}</b> que o
        limiar exibido de <b>{fmtScore(threshold)}</b>:{" "}
        {exam.score >= threshold
          ? "o modelo gera um alerta para revisão."
          : "o modelo não gera um alerta neste limiar."}{" "}
        O score não é probabilidade de doença. A reconstrução não é uma versão
        comprovadamente saudável, e o mapa pode refletir anatomia e aquisição.
      </p>
      <p className="comparison-footnote">
        Gσ representa o filtro gaussiano; ∗ é a convolução. A suavização
        acontece antes do quadrado. Imagens e score foram calculados na
        exportação, com escala de cor fixa por modelo.
        {lungMask && " M é a máscara fixa dos pulmões; ⊙ multiplica pixel a pixel. A média é calculada na imagem inteira, incluindo os zeros fora da máscara, conforme o código do score."}
      </p>
    </div>
  );
}
