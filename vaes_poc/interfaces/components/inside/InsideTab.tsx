import Generator from "@/components/inside/Generator";
import Interpolation from "@/components/inside/Interpolation";
import LatentMap from "@/components/inside/LatentMap";
import SigmaCard from "@/components/inside/SigmaCard";
import type { VaeData } from "@/lib/types";

export default function InsideTab({ data }: { data: VaeData }) {
  const model = data.runs[data.main].model;
  return (
    <div className="inside-grid">
      <article className="panel card">
        <header>
          <h2>Gerador</h2>
          <p>
            O decoder transforma {model.latent_dim} números (o código <i>z</i>)
            num raio-X que nunca existiu.
          </p>
        </header>
        <Generator latent={data.latent} latentDim={model.latent_dim} />
        <p className="note">
          Um eixo por vez: cada posição foi pré-calculada pelo decoder. Combinar
          os eixos ao mesmo tempo exige rodar o decoder no navegador (ONNX), um
          próximo passo.
        </p>
      </article>

      <article className="panel card">
        <header>
          <h2>Mapa do espaço latente</h2>
          <p>
            Cada ponto é um exame de teste, posicionado pelo seu código <i>μ</i>{" "}
            (PCA em 2D). O VAE nunca viu os rótulos.
          </p>
        </header>
        <LatentMap latent={data.latent} />
      </article>

      <article className="panel card">
        <header>
          <h2>Suavização do score (σ)</h2>
          <p>
            Comparar o exame com a reconstrução <b>em baixa frequência</b> foi o
            achado principal da PoC.
          </p>
        </header>
        <SigmaCard sigma={data.sigma} />
        <p className="note">
          σ = 0 é o erro pixel a pixel ingênuo: acende no detalhe fino do exame
          inteiro. Com σ maior, sobram as regiões. O exemplo é um caso
          favorável: na média do teste, o ponto mais quente cai dentro da caixa
          em poucos exames.
        </p>
      </article>

      <article className="panel card">
        <header>
          <h2>Interpolação</h2>
          <p>
            Caminhando em linha reta no espaço latente, de um exame normal até
            uma opacidade.
          </p>
        </header>
        <Interpolation interpolation={data.latent.interpolation} />
        <p className="note">
          Neste exemplo, a reconstrução não preserva a opacidade. O decoder foi
          treinado apenas com exames normais; sua saída não é uma versão
          comprovadamente saudável do paciente.
        </p>
      </article>
    </div>
  );
}
