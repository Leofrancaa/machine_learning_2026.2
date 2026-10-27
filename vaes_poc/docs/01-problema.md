# O problema

## Frase de abertura

> Usamos um VAE treinado só com raios-X normais para reordenar a fila de laudos, colocando primeiro os exames que mais fogem do padrão — sem precisar de exemplos rotulados da doença.

## O problema real

Em serviços de radiologia com muita demanda, os exames de tórax esperam na fila até serem laudados, normalmente **por ordem de chegada**. Um exame com pneumonia pode ficar horas atrás de dezenas de exames normais.

**Objetivo:** reordenar a fila para que os exames com **maior suspeita de alteração** sejam vistos primeiro. O radiologista continua laudando todos; o modelo só muda a ordem. Não é diagnóstico.

## Por que um VAE, e não um classificador

- Um classificador precisa de muitos exemplos rotulados de **cada** doença que deve reconhecer.
- Exames normais são abundantes e baratos de obter; doenças raras, não.
- A ideia: aprender **só como é um pulmão normal** e sinalizar o que foge disso, inclusive alterações nunca vistas.
- O VAE ainda entrega uma reconstrução "normalizada" do exame. A diferença entre o exame e ela mostra **onde** está a alteração (mapa de calor).

## A tarefa de ML

| | |
|---|---|
| Tipo | detecção de anomalias sem rótulos (o modelo só conhece a classe "normal") |
| Entrada | raio-X de tórax frontal (128×128 px) |
| Saída | score de anomalia, mapa de calor, decisão "revisar com prioridade" ou "dentro do padrão", posição na fila |
| Métrica técnica | AUC-ROC: o score separa os exames alterados dos normais? |
| Métrica de produto | posição média de um exame alterado numa fila de 100 exames |

**Pergunta central da PoC:** dá para priorizar a fila sem rótulos da doença? Quanto se perde em relação a um modelo supervisionado?

## O que a PoC **não** resolve

- **Diagnóstico:** o score indica suspeita, não doença.
- **Gravidade:** o score mede o quanto o exame foge do normal, não o quão grave é o caso. No RSNA, a correlação entre o score e a área da lesão é **−0,14**: o score não acompanha a extensão. Além disso, exames com enquadramento atípico recebem score alto sem ter nada.
- **Outros hospitais e equipamentos:** o limiar teria que ser recalibrado, porque qualquer mudança no "normal" vira alarme.
- **"Qualquer doença":** só foi testado com pneumonia/opacidade e com os "anormais sem opacidade" do RSNA (AUC 0,69).

> ⚠️ 18/10/2026: na v3, "anormal sem opacidade" × normal subiu para AUC 0,74, e a correlação score × área da lesão ficou em −0,05 (continua sem medir gravidade). Ver [09-resultados-v3.md](09-resultados-v3.md).

## Como falar

- Correto: "O VAE ordena a fila pelo grau de desvio em relação ao padrão normal. Os exames com maior suspeita de alteração são vistos primeiro."
- Evitar: "o modelo prevê a gravidade" ou "os casos mais graves vão primeiro".
