"use client";

import { useEffect, useState } from "react";

import GameTab from "@/components/game/GameTab";
import InsideTab from "@/components/inside/InsideTab";
import QueueTab from "@/components/queue/QueueTab";
import Icon from "@/components/Icon";
import LearnTab from "@/components/learn/LearnTab";
import ModelsTab from "@/components/models/ModelsTab";
import type { ModelCatalog } from "@/lib/models";
import type { VaeData } from "@/lib/types";

const TABS = [
  {
    key: "learn",
    label: "Entenda o VAE",
    icon: "pulse",
    title: "Entenda o VAE",
    description:
      "Uma experiência para entender o modelo, sua aplicação e os limites do que ele aprendeu.",
  },
  {
    key: "queue",
    label: "Fila de triagem",
    icon: "queue",
    title: "Fila de triagem",
    description:
      "Compare a ordem de chegada com a prioridade por anomalia. Selecione um exame para inspecionar.",
  },
  {
    key: "inside",
    label: "Por dentro do VAE",
    icon: "layers",
    title: "Por dentro do VAE",
    description:
      "Explore a geração de imagens, o espaço latente e o cálculo do score de anomalia.",
  },
  {
    key: "models",
    label: "Modelos e métricas",
    icon: "pulse",
    title: "Modelos e métricas",
    description: "Acompanhe a evolução dos modelos, compare métodos e entenda o que cada métrica permite concluir.",
  },
  {
    key: "game",
    label: "Você × VAE",
    icon: "spark",
    title: "Você × VAE",
    description:
      "Ordene os exames por prioridade e compare sua escolha com o modelo e os rótulos do dataset.",
  },
] as const;
type TabKey = (typeof TABS)[number]["key"];

export default function Demo({ data, catalog }: { data: VaeData; catalog: ModelCatalog }) {
  const [tab, setTab] = useState<TabKey>("learn");

  useEffect(() => {
    function syncHash() {
      const match = TABS.find((t) => t.key === window.location.hash.slice(1));
      if (match) setTab(match.key);
      else if (!window.location.hash) setTab("learn");
    }
    syncHash();
    window.addEventListener("hashchange", syncHash);
    return () => window.removeEventListener("hashchange", syncHash);
  }, []);

  function open(key: TabKey) {
    setTab(key);
    window.location.hash = key;
  }

  const current = TABS.find((t) => t.key === tab)!;

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Pular para o conteúdo
      </a>
      <aside className="sidebar">
        <a className="brand" href="#learn" aria-label="VAE Lab, entenda o VAE">
          <span className="logo">
            <Icon name="pulse" size={23} />
          </span>
          <span className="brand-name">
            VAE<span>LAB</span>
            <small>RADIOLOGIA · PESQUISA</small>
          </span>
        </a>
        <p className="nav-label">EXPLORAR</p>
        <nav
          className="tabs"
          role="tablist"
          aria-label="Áreas da demonstração"
          aria-orientation="vertical"
        >
          {TABS.map((t, index) => (
            <button
              key={t.key}
              id={`tab-${t.key}`}
              role="tab"
              aria-selected={tab === t.key}
              aria-controls={`panel-${t.key}`}
              tabIndex={tab === t.key ? 0 : -1}
              onClick={() => open(t.key)}
              onKeyDown={(event) => {
                const step =
                  event.key === "ArrowDown" || event.key === "ArrowRight"
                    ? 1
                    : event.key === "ArrowUp" || event.key === "ArrowLeft"
                      ? -1
                      : 0;
                if (!step && event.key !== "Home" && event.key !== "End")
                  return;
                event.preventDefault();
                const next =
                  event.key === "Home"
                    ? TABS[0]
                    : event.key === "End"
                      ? TABS[TABS.length - 1]
                      : TABS[(index + step + TABS.length) % TABS.length];
                open(next.key);
                document.getElementById(`tab-${next.key}`)?.focus();
              }}
            >
              <Icon name={t.icon} size={19} />
              {t.label}
              <Icon name="chevron" size={14} />
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="sidebar-note-icon">
            <Icon name="layers" />
          </span>
          <strong>
            Aprender o normal.
            <br />
            Encontrar o diferente.
          </strong>
          <p>Um autoencoder variacional treinado apenas com exames normais.</p>
          <span className="version-tag">PROVA DE CONCEITO</span>
        </div>
        <div className="sidebar-bottom">
          <span className="status-dot" /> Ambiente de pesquisa
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <Icon name="chevron" size={13} />
            <span>{current.label}</span>
          </div>
          <div className="dataset-badge">
            <span className="status-dot" /> RSNA Pneumonia{" "}
            <span className="dataset-version">
              {data.generated.split("-").reverse().join("/")}
            </span>
          </div>
        </header>

        {/* Trocar de aba não perde o estado de cada experiência. */}
        <main id="workspace">
          <div className="workspace-intro">
            <div>
              <p className="eyebrow">
                RADIOGRAFIA DE TÓRAX · INTELIGÊNCIA ARTIFICIAL
              </p>
              <h1>{current.title}</h1>
              <p className="intro-description">{current.description}</p>
            </div>
            <span className="research-badge">
              <Icon name="pulse" size={16} /> {tab === "models" ? "Relatórios reais · validação e teste" : "Exames reais · conjunto de teste"}
            </span>
          </div>
          <section
            id="panel-learn"
            aria-labelledby="tab-learn"
            hidden={tab !== "learn"}
            className={`tab ${tab === "learn" ? "active" : ""}`}
            role="tabpanel"
            tabIndex={0}
          >
            <LearnTab data={data} onNavigate={open} />
          </section>
          <section
            id="panel-queue"
            aria-labelledby="tab-queue"
            hidden={tab !== "queue"}
            className={`tab ${tab === "queue" ? "active" : ""}`}
            role="tabpanel"
            tabIndex={0}
          >
            <QueueTab data={data} />
          </section>
          <section
            id="panel-inside"
            aria-labelledby="tab-inside"
            hidden={tab !== "inside"}
            className={`tab ${tab === "inside" ? "active" : ""}`}
            role="tabpanel"
            tabIndex={0}
          >
            <InsideTab data={data} />
          </section>
          <section
            id="panel-models"
            aria-labelledby="tab-models"
            hidden={tab !== "models"}
            className={`tab ${tab === "models" ? "active" : ""}`}
            role="tabpanel"
            tabIndex={0}
          >
            <ModelsTab catalog={catalog} activeRun={data.runs[data.main].model.run} />
          </section>
          <section
            id="panel-game"
            aria-labelledby="tab-game"
            hidden={tab !== "game"}
            className={`tab ${tab === "game" ? "active" : ""}`}
            role="tabpanel"
            tabIndex={0}
          >
            <GameTab exams={data.game} />
          </section>
        </main>

        <footer>
          <p>Prova de conceito acadêmica · não é um dispositivo médico</p>
          <p>
            Imagens:{" "}
            <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC">
              NIH Chest X-ray Dataset
            </a>
            , fornecido pelo NIH Clinical Center (Wang et al.,{" "}
            <i>ChestX-ray8</i>, IEEE CVPR 2017, pp. 3462–3471). Rótulos e
            caixas:{" "}
            <a href="https://www.rsna.org/rsnai/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018">
              RSNA Pneumonia Detection Challenge
            </a>{" "}
            (Shih et al.,{" "}
            <i>
              Augmenting the National Institutes of Health Chest Radiograph
              Dataset with Expert Annotations of Possible Pneumonia
            </i>
            , Radiology: AI, 2019).
          </p>
        </footer>
      </div>
    </div>
  );
}
