# A natureza dos VAEs, comparada aos outros projetos

Comparação com os projetos da pasta `ml-exercises`, cada um respondendo a um tipo diferente de pergunta.

| Projeto | Família | Pergunta que o modelo responde |
|---|---|---|
| **Olist** (LightGBM/XGBoost) | supervisionado, tabular | "Este pedido vai atrasar?" |
| **deep_learning** (rede neural no TensorFlow) | supervisionado, rede neural | a mesma do Olist, com uma rede no lugar das árvores |
| **classificador_imagens** (CNN no CIFAR-10) | supervisionado, imagem | "Qual destas 10 classes é esta imagem?" |
| **reconhecimento_facial** (Facenet512) | embeddings + distância | "Este rosto se parece com algum que eu conheço?" |
| **VAE** | **gerativo, não supervisionado** | "**Como é um exame normal?**" e, a partir disso, "o quanto este exame foge disso?" |

## 1. O que cada modelo aprende

- **Classificação e regressão** (Olist, rede neural, CNN) aprendem a **relação entre a entrada e um rótulo**. Na prática, aprendem **a fronteira** entre as classes, e precisam de exemplos rotulados de **todas** elas.
- **O VAE** aprende **como são os próprios dados**, sem rótulo. Não aprende uma fronteira: aprende o **formato de uma única classe**, a dos normais.

**Analogia:** o classificador estudou fotos de gatos e de cachorros e aprendeu a separar os dois. O VAE é um fiscal de alfândega que viu milhares de malas comuns: não sabe como é o contrabando, mas percebe quando uma mala "não parece normal".

**Autoencoder como regressão:** é uma regressão em que o alvo é a própria entrada. A rede prevê a imagem a partir da própria imagem, passando por um funil de 8 números. Como não cabe tudo no funil, ela aprende só o que é típico. O atípico (uma opacidade) ela não consegue reproduzir, e essa falha vira o score.

## 2. Etapa por etapa

| | Olist / rede neural | CNN (CIFAR-10) | Reconhecimento facial | VAE (PoC) |
|---|---|---|---|---|
| **Dados de treino** | pedidos + rótulo `is_late` | imagens + classe | nenhum treino: 8 fotos de 2 pessoas viram um "banco" | **só exames normais, sem rótulo** |
| **Loss** | log-loss / entropia cruzada: compara a previsão com o **rótulo** | entropia cruzada com 10 classes | — (a Facenet já veio treinada) | BCE(x, x̂) + β·KL: compara a saída com **a própria entrada** |
| **Saída** | probabilidade de atraso | 10 probabilidades (softmax) | vetor de 512 números | reconstrução x̂, código μ e mapa de calor |
| **Decisão** | limiar 0,14, escolhido pelo F1 **com rótulos** | a classe mais provável | distância cosseno ≤ 0,3 | score ≥ percentil 95 **dos normais** |
| **A loss mede a tarefa?** | sim | sim: acurácia de validação 0,87 com loss 0,39 | — | **não**: mede a reconstrução de normais; a detecção se mede à parte (AUC 0,76) |
| **Gera dados novos?** | não | não | não | **sim**: sorteia *z* e o decoder cria um raio-X |

## 3. O parente mais próximo do VAE é o reconhecimento facial

- **Reconhecimento facial:** encoder → 512 números → distância até os rostos conhecidos → limiar.
- **VAE:** encoder e decoder → versão "normal" do exame → distância entre o exame e ela (resíduo suavizado) → limiar.

Os dois respondem "**isto se parece com o que eu conheço?**" e lidam bem com o **desconhecido**: um rosto nunca cadastrado é negado; uma doença nunca vista é sinalizada.

A CNN do CIFAR-10 é o oposto: mostre um raio-X a ela e ela responde "avião" ou "navio", porque o softmax **sempre escolhe uma das 10 classes**. Ela não tem como dizer "isso não é nada que eu conheço".

Em **arquitetura**, o VAE é parente da CNN: o encoder dele é uma CNN, e a CNN de comparação da PoC é literalmente o encoder do VAE com uma camada de classificação no fim. A diferença está no **objetivo de treino**: um classifica, o outro reconstrói.

## 4. O preço e o ganho de não usar rótulos

**Preço:**
- Quando há rótulos, o supervisionado ganha: CNN 0,83 × VAE 0,76 no RSNA.

  > ⚠️ 18/10/2026: na v3 (normalização + máscara dos pulmões), CNN 0,86 × VAE 0,80 no RSNA só PA; o VAE segue empatado com o autoencoder comum e o PCA. Ver [09-resultados-v3.md](09-resultados-v3.md).

- O VAE sinaliza **qualquer** coisa incomum, inclusive o que não importa (enquadramento atípico, incidência AP). Não distingue "estranho" de "doente".
- O limiar não pode ser escolhido pelo F1, como no Olist, porque isso exige exemplos da doença.

**Ganho:**
- Funciona quando rótulos são raros ou caros. É parente do experimento de active learning do Olist ("e se rotular fosse caro?").
- Sinaliza alterações que nunca viu (AUC 0,69 nos "anormais sem opacidade" do RSNA).
- O "V" dá um **espaço latente organizado**: gerar exames, interpolar e inspecionar o que o modelo aprendeu. O autoencoder comum não faz isso, embora tenha empatado na detecção.

## 5. Onde VAEs são usados (slide "casos de uso")

| Uso | Com raio-X de tórax | Encaixe |
|---|---|---|
| **Detecção de anomalias** ("detecta doenças raras pelo desvio do padrão normal") | treinar só com normais e sinalizar exames alterados | ✅ **é a PoC** |
| **Redução de dimensionalidade / representação** | usar o código latente com poucos rótulos | ✅ bom como extensão |
| **Geração / aumento de dados** | gerar raios-X sintéticos para treinar uma CNN | ⚠️ fraco: amostras borradas, ganho pequeno |
| **Remoção de ruído / preenchimento** | apagar regiões e reconstruir | ❌ o VAE perde o detalhe fino (que é diagnóstico) e pode "inventar" anatomia |
| **Descoberta científica** (moléculas, exames 3D) | não se aplica a raio-X 2D | ❌ |

Sobre alternativas: modelos de difusão hoje produzem reconstruções "saudáveis" melhores, mas são bem mais pesados; GANs são instáveis de treinar. Para uma PoC, o VAE é a escolha certa; difusão entra como próximo passo.

## Em uma frase

Classificadores e regressores aprendem a **responder perguntas sobre os dados**. O VAE aprende **como os dados são**, e a resposta à pergunta ("este exame é anormal?") sai como subproduto, medindo o quanto ele falha em explicar cada exame.
