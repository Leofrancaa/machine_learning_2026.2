# Decisões tomadas e por quê

## Por que o RSNA

A PoC precisava de um dataset com adultos, com as lesões marcadas (para validar o mapa de calor), com alterações além da pneumonia (para testar "anomalias nunca vistas") e com resolução maior que 64 px. Candidatos avaliados no Kaggle:

| Dataset | O que acrescentaria | Por que não (ou sim) |
|---|---|---|
| **RSNA Pneumonia Detection Challenge** | adultos, caixas nas lesões, classe "anormal sem opacidade" | ✅ escolhido: atende os três requisitos; é o raio-X de tórax do benchmark BMAD de anomalias médicas |
| NIH ChestX-ray14 / ChestMNIST | 112 mil exames, 14 achados | rótulos extraídos por NLP (ruidosos); bom para testar "doenças nunca vistas" |
| VinBigData (VinDr-CXR) | caixas de radiologistas para vários achados | mais esforço; rótulos só no treino da competição |
| COVID-QU-Ex | máscaras dos pulmões (restringir o score) | junta várias fontes: risco de aprender "qual hospital" |
| Chest X-Ray Images (Kermany) | pneumonia × normal em resolução cheia | só crianças, de um hospital; sem caixas nas lesões |
| CheXpert-small | 224 mil exames | cópia não oficial; grande demais |

## Por que a "opção A" (triagem por anomalia, validada)

Com rótulos sobrando, a pergunta "por que VAE e não CNN?" fica mais forte. As opções discutidas:

- **A — triagem por anomalia, agora validada:** mesma pergunta da PoC, com avaliação mais forte (outras anomalias, localização com caixas, fila). ✅ escolhida: aproveita tudo o que já existia e cobre o barema.
- **B — VAE como explicação de uma CNN:** a CNN decide, o VAE mostra a "versão saudável". Mais trabalho e muda a narrativa.
- **C — poucos rótulos (semi-supervisionado):** VAE treinado sem rótulos + classificador com 50/100/500 rótulos. Boa extensão futura.

**Só PA:** por causa do atalho AP/PA (78% das opacidades são AP × 18% dos normais). O experimento com AP + PA roda à parte, para medir o atalho.

## Por que não Docker (por enquanto)

- No Mac, o Docker roda numa máquina virtual Linux que **não usa a GPU** (MPS): o treino ficaria várias vezes mais lento.
- O ambiente já é reproduzível com `uv.lock` + Python 3.12 + `make requirements`.
- Faz sentido **na demo**: uma interface de inferência roda bem na CPU e sobe em qualquer máquina com um comando.

## Plano da demo (pasta `interfaces/`, hospedagem na Vercel)

Site em Next.js (React), pré-renderizado no build, com três abas:

1. **Fila de triagem:** a lista de trabalho se reordena pelo score; visualizador com original, reconstrução, mapa de calor e caixas do laudo; botão AP + PA mostra o atalho ao vivo.
2. **Por dentro do VAE:** gerador de raios-X pelos 8 eixos de *z*, mapa do espaço latente, efeito do σ, interpolação.
3. **Você × VAE:** a turma ordena 8 exames pelo celular e compara com o VAE e com o laudo.

Pré-calculado + decoder em ONNX no navegador: funciona offline, sem GPU e sem Wi-Fi. Ensaiar e gravar um vídeo de reserva.

**Imagens no site:** os termos de uso do RSNA Pneumonia Detection Challenge permitem redistribuir as imagens e as anotações com atribuição ao NIH e ao RSNA, sem tentar identificar os pacientes. A regra geral do Kaggle contra redistribuir (seção 7.B) não foi substituída pelas regras específicas da competição. Mesmo assim, por ser uma competição encerrada e um trabalho acadêmico, decidimos publicar, com a atribuição completa no rodapé do site e em `interfaces/README.md`.
