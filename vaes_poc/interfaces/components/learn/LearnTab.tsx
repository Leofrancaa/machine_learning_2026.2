"use client";

import { useRef, useState } from "react";
import Icon from "@/components/Icon";
import ExamComparison from "@/components/queue/ExamComparison";
import QueueComparison from "@/components/queue/QueueComparison";
import LatentLesson from "@/components/learn/LatentLesson";
import { byScore } from "@/lib/format";
import type { VaeData } from "@/lib/types";

const STEPS = [
  { title: "O problema", subtitle: "Por que aprender o normal?" },
  { title: "A arquitetura", subtitle: "Da imagem ao código e de volta" },
  { title: "O variacional", subtitle: "Por que aprender uma distribuição?" },
  { title: "O score", subtitle: "Da reconstrução à diferença" },
  { title: "A aplicação", subtitle: "Da diferença à posição na fila" },
];

export default function LearnTab({
  data,
  onNavigate,
}: {
  data: VaeData;
  onNavigate: (tab: "queue" | "inside" | "game") => void;
}) {
  const [step, setStep] = useState(0);
  const run = data.runs[data.main];
  const sample = byScore(run.queue)[0];
  const current = STEPS[step];
  const imageSize = run.model.image_size;
  const blocks = imageSize ? Math.log2(imageSize / 4) : undefined;
  const resolutions =
    imageSize && blocks !== undefined && Number.isInteger(blocks)
      ? Array.from({ length: blocks + 1 }, (_, index) => imageSize / 2 ** index)
      : [];
  const heading = useRef<HTMLHeadingElement>(null);
  function goTo(index: number) {
    setStep(index);
    requestAnimationFrame(() => {
      heading.current?.focus({ preventScroll: true });
      heading.current?.scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
      });
    });
  }

  return (
    <div className="learn-experience">
      <div className="learning-intro">
        <div>
          <span className="lesson-label">PERCURSO INTERATIVO · 5 ETAPAS</span>
          <h2>
            Aprender a reconstruir.
            <br />
            <span>Usar a diferença para priorizar.</span>
          </h2>
          <p>
            Entenda a ideia, explore a matemática e acompanhe um exame real até
            sua posição na fila.
          </p>
        </div>
        <div className="learning-summary">
          <Icon name="layers" size={25} />
          <strong>VAE</strong>
          <span>Variational Autoencoder</span>
          <small>Autoencoder variacional</small>
        </div>
      </div>
      <nav className="lesson-steps" aria-label="Etapas da explicação">
        {STEPS.map((item, index) => (
          <button
            key={item.title}
            aria-current={step === index ? "step" : undefined}
            onClick={() => goTo(index)}
          >
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{item.title}</strong>
          </button>
        ))}
      </nav>
      <article
        className="panel lesson-content"
        aria-labelledby="lesson-heading"
      >
        <header className="lesson-heading">
          <div>
            <p className="eyebrow">
              ETAPA {step + 1} DE {STEPS.length}
            </p>
            <h2 ref={heading} tabIndex={-1} id="lesson-heading">
              {current.subtitle}
            </h2>
          </div>
          <span className="lesson-source">
            {step === 2
              ? "ILUSTRAÇÃO MATEMÁTICA"
              : "CONFIGURAÇÃO E DADOS REAIS DA POC"}
          </span>
        </header>
        <div className="lesson-body" key={step}>
          {step === 0 && (
            <>
              <p className="lesson-lead">
                Em uma fila de radiografias, a ordem de chegada não diz quais
                imagens apresentam alterações. A pergunta da PoC é:{" "}
                <strong>
                  conhecendo apenas exames normais no treino, conseguimos
                  colocar imagens diferentes mais cedo na fila?
                </strong>
              </p>
              <div className="purpose-grid">
                <div>
                  <span className="purpose-number">01</span>
                  <h3>Aprender com normais</h3>
                  <p>
                    O VAE aprende a comprimir e reconstruir radiografias
                    normais. As opacidades não entram nesse treino.
                  </p>
                </div>
                <div>
                  <span className="purpose-number">02</span>
                  <h3>Medir a diferença</h3>
                  <p>
                    Comparamos a imagem recebida com o que o modelo reconstruiu.
                    Uma diferença maior aumenta o score.
                  </p>
                </div>
                <div>
                  <span className="purpose-number">03</span>
                  <h3>Testar a priorização</h3>
                  <p>
                    Ordenamos pelo score e usamos os rótulos reservados para
                    avaliar se as opacidades apareceram mais cedo.
                  </p>
                </div>
              </div>
              <div className="lesson-callout">
                <Icon name="info" />
                <p>
                  “Diferente do padrão” não significa “pneumonia”. Equipamento,
                  anatomia e aquisição também podem gerar diferenças. O objetivo
                  é estudar a priorização, sem produzir diagnóstico.
                </p>
              </div>
              <details className="technical-note">
                <summary>Como os dados são separados?</summary>
                <p>
                  O RSNA é dividido por paciente: 70% treino, 15% validação e
                  15% teste. O VAE usa apenas normais do treino. A validação
                  escolhe a suavização com normais e positivos e calibra o
                  limiar com normais. O teste mede o resultado final. O
                  experimento principal mantém apenas PA para reduzir o atalho
                  entre incidência e classe.
                </p>
              </details>
            </>
          )}
          {step === 1 && (
            <>
              <p className="lesson-lead">
                O encoder reduz a imagem a uma representação compacta. O decoder
                tenta reconstruí-la. Entre eles, o VAE aprende uma distribuição
                em <strong>{run.model.latent_dim} dimensões</strong>.
              </p>
              <div
                className="architecture-flow"
                aria-label="Arquitetura do VAE"
              >
                <div className="architecture-image">
                  <img
                    src={sample.img}
                    alt="Radiografia de entrada do exemplo"
                  />
                  <span>Entrada x</span>
                  <small>
                    {imageSize
                      ? `1 × ${imageSize} × ${imageSize}`
                      : "Imagem normalizada"}
                  </small>
                </div>
                <Icon name="arrow" />
                <div className="architecture-block encoder">
                  <Icon name="layers" size={28} />
                  <h3>Encoder</h3>
                  <p>
                    {resolutions.length
                      ? `${resolutions.length - 1} convoluções`
                      : "Convoluções"}
                  </p>
                  <small>
                    {resolutions.length
                      ? resolutions.join(" → ")
                      : "Reduz a resolução"}
                  </small>
                </div>
                <Icon name="arrow" />
                <div className="architecture-block latent">
                  <span className="latent-symbol">μ · log σ²</span>
                  <h3>Distribuição latente</h3>
                  <p>
                    {run.model.latent_dim} médias + {run.model.latent_dim}{" "}
                    log-variâncias
                  </p>
                  <small>Treino: z = μ + σ ⊙ ε</small>
                </div>
                <Icon name="arrow" />
                <div className="architecture-block decoder">
                  <Icon name="layers" size={28} />
                  <h3>Decoder</h3>
                  <p>Convoluções transpostas</p>
                  <small>
                    {resolutions.length
                      ? [...resolutions].reverse().join(" → ")
                      : "Recupera a resolução"}
                  </small>
                </div>
                <Icon name="arrow" />
                <div className="architecture-image">
                  <img
                    src={sample.rec}
                    alt="Reconstrução do exemplo pelo decoder"
                  />
                  <span>Reconstrução x̂</span>
                  <small>
                    {imageSize
                      ? `1 × ${imageSize} × ${imageSize}`
                      : "Imagem reconstruída"}
                  </small>
                </div>
              </div>
              <div className="training-objective">
                <div>
                  <span className="lesson-label">O QUE O TREINO MINIMIZA</span>
                  <code>Loss = BCE(x, x̂) + β · KL(q(z|x) ∥ N(0, I))</code>
                </div>
                <p>
                  <b>BCE:</b> penaliza o erro de reconstrução. <b>KL:</b>{" "}
                  regulariza as distribuições latentes em direção à normal
                  padrão. β controla o peso dessa regularização.
                </p>
              </div>
              <details className="technical-note">
                <summary>Detalhes da implementação em PyTorch</summary>
                <p>
                  No VAE convolucional desta PoC, o tensor do encoder é achatado
                  e alimenta duas camadas lineares, para μ e log σ². O decoder
                  recebe z e usa uma Sigmoid para gerar pixels entre 0 e 1.
                  {run.model.epochs
                    ? ` Esta execução treinou por ${run.model.epochs} épocas.`
                    : " O número de épocas não foi incluído nesta exportação."}
                  {imageSize
                    ? ` A resolução exportada é ${imageSize} px.`
                    : " A resolução não foi incluída nesta exportação."}
                  Os parâmetros de otimização devem ser conferidos nos metadados
                  da execução.
                </p>
              </details>
            </>
          )}
          {step === 2 && (
            <>
              <p className="lesson-lead">
                Um autoencoder comum produz um código por imagem. O VAE aprende{" "}
                <strong>uma média e uma variância para cada dimensão</strong>,
                permitindo amostrar códigos e reconstruí-los. Explore uma
                dimensão abaixo.
              </p>
              <LatentLesson />
              <div className="lesson-callout">
                <Icon name="info" />
                <p>
                  Os valores deste gráfico são didáticos e não são o código do
                  exame selecionado. O σ aqui é o desvio da distribuição
                  latente; o σ da próxima etapa é a largura de um filtro na
                  imagem.
                </p>
              </div>
              <details className="technical-note">
                <summary>Por que usar a reparametrização?</summary>
                <p>
                  O encoder fornece μ e log σ². Calculamos σ = exp(0,5 × log σ²)
                  e amostramos ε de uma normal padrão. Escrever z = μ + σ ⊙ ε
                  permite propagar os gradientes até μ e σ durante o treino. A
                  KL ajuda a organizar o espaço latente; ela não garante
                  separação entre exames normais e alterados. Na avaliação deste
                  projeto, z = μ evita variação aleatória entre reconstruções do
                  mesmo exame.
                </p>
              </details>
            </>
          )}
          {step === 3 && (
            <>
              <p className="lesson-lead">
                Este é o exame <strong>{sample.id}</strong>, do conjunto de
                teste. As três imagens abaixo foram exportadas do modelo
                principal. Acompanhe como sua reconstrução gera o score.
              </p>
              <ExamComparison
                exam={sample}
                sigma={run.model.sigma}
                threshold={run.model.threshold}
                imageSize={run.model.image_size}
                originalSize={run.model.original_image_size}
                normalized={run.model.normalize}
                lungMask={run.model.lung_mask}
              />
              <details className="technical-note">
                <summary>Por que suavizar antes de medir o erro?</summary>
                <p>
                  O VAE reconstrói detalhes finos de forma imprecisa, inclusive
                  em normais. O filtro gaussiano reduz a influência dessas
                  diferenças e preserva estruturas maiores. O σ escolhido nesta
                  execução é {run.model.sigma} px, pela AUC na validação. O
                  limiar desta execução é descrito como:{" "}
                  {run.model.threshold_rule ??
                    "calibrado na validação; a regra não foi incluída na exportação"}
                  . Não é uma porcentagem de doença.
                </p>
              </details>
            </>
          )}
          {step === 4 && (
            <>
              <p className="lesson-lead">
                O score determina a ordem, sem consultar o rótulo do exame. Veja
                o mesmo conjunto em ordem de chegada e por anomalia. As posições
                destacadas pertencem ao exame usado no exemplo.
              </p>
              <QueueComparison
                exams={run.queue}
                selectedId={sample.id}
                reveal={false}
              />
              <div className="learning-results">
                <div>
                  <span>AUC NO TESTE</span>
                  <strong>
                    {run.model.test?.roc_auc.toFixed(3).replace(".", ",") ??
                      "—"}
                  </strong>
                  <p>
                    Mede ordenação entre opacidade e normal; não é acurácia.
                  </p>
                </div>
                <div>
                  <span>SENSIBILIDADE NO LIMIAR</span>
                  <strong>
                    {run.model.test
                      ? `${(run.model.test.sensitivity * 100).toFixed(1).replace(".", ",")}%`
                      : "—"}
                  </strong>
                  <p>Percentual de opacidades sinalizadas neste limiar.</p>
                </div>
                <div>
                  <span>O QUE A DEMO EXECUTA</span>
                  <strong>Exploração</strong>
                  <p>
                    Scores e imagens são pré-calculados. O navegador ordena e
                    apresenta os resultados.
                  </p>
                </div>
              </div>
              <div className="lesson-actions">
                <button
                  className="btn primary"
                  onClick={() => onNavigate("queue")}
                >
                  Explorar a fila <Icon name="arrow" size={16} />
                </button>
                <button className="btn" onClick={() => onNavigate("inside")}>
                  Experimentar o espaço latente
                </button>
              </div>
            </>
          )}
        </div>
        <div className="lesson-navigation">
          <button
            className="btn"
            disabled={step === 0}
            onClick={() => goTo(step - 1)}
          >
            <Icon
              name="chevron"
              size={15}
              style={{ transform: "rotate(180deg)" }}
            />
            Anterior
          </button>
          <span role="status">
            {step + 1} / {STEPS.length} · {current.title}
          </span>
          <button
            className="btn primary"
            onClick={() =>
              step < STEPS.length - 1 ? goTo(step + 1) : onNavigate("queue")
            }
          >
            {step < STEPS.length - 1 ? "Próxima etapa" : "Abrir fila"}
            <Icon name="arrow" size={15} />
          </button>
        </div>
      </article>
      <p className="learning-footer">
        Continue pelo laboratório para explorar geração, PCA, suavização e
        interpolação, ou teste sua ordenação no desafio Você × VAE.
      </p>
    </div>
  );
}
