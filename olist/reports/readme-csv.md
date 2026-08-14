# Guia dos arquivos CSV — Olist

Este projeto contém o **Brazilian E-Commerce Public Dataset by Olist**. Os dados descrevem pedidos feitos em um marketplace brasileiro entre **setembro de 2016 e outubro de 2018**. O modelo é relacional: o pedido é o centro e as demais tabelas acrescentam cliente, itens, pagamento, avaliação, produto, vendedor e localização.

Os CSVs estão em `olist/data/raw/`.

## Visão do relacionamento

```text
customers (customer_id) ──< orders (order_id) ──< order_items >── products
                                  │                    │
                                  │                    └────────> sellers
                                  ├──< order_payments
                                  └──< order_reviews

customers/sellers (zip prefix) ──> geolocation
products (category) ─────────────> category_translation
```

`<` indica uma relação de um-para-muitos. Ao juntar itens, pagamentos ou avaliações aos pedidos, um mesmo pedido pode aparecer em várias linhas.

## Tabelas

### `olist_orders_dataset.csv` — pedidos

Tabela central, com **99.441 pedidos** (um por `order_id`). Registra o cliente, status e marcos do processo de compra e entrega.

- Chaves: `order_id` (pedido) e `customer_id` (ligação com clientes).
- Datas: compra, aprovação, postagem/entrega à transportadora, entrega ao cliente e prazo estimado.
- Status: predominam pedidos `delivered` (96.478); também há `shipped`, `canceled`, `unavailable`, `invoiced`, `processing`, `created` e `approved`.
- As datas de aprovação e entrega podem estar vazias, principalmente em pedidos não concluídos.

### `olist_customers_dataset.csv` — clientes e destino

Possui **99.441 registros**, um para cada `customer_id` usado nos pedidos. Traz CEP resumido, cidade e UF do destino.

- `customer_id` identifica a ocorrência do cliente em um pedido.
- `customer_unique_id` identifica a pessoa de forma consolidada e deve ser usado para recorrência/frequência de compra: há **96.096 clientes únicos**.
- Liga com `orders` por `customer_id` e com `geolocation` pelo prefixo de CEP.

### `olist_order_items_dataset.csv` — itens vendidos

Contém **112.650 itens**, relativos a **98.666 pedidos**. É a tabela para receita de produtos, frete, mix e vendedores.

- Chave lógica: `order_id` + `order_item_id`.
- `product_id` liga a produtos; `seller_id` liga a vendedores.
- `price` é o valor do produto e `freight_value` o frete do item; `shipping_limit_date` é o limite de postagem do vendedor.
- Um pedido pode ter vários itens (até 21 neste conjunto); portanto, somar valores após joins exige cuidado para não duplicar pagamentos ou avaliações.

### `olist_order_payments_dataset.csv` — pagamentos

Há **103.886 registros** para **99.440 pedidos**. Registra a forma, parcelas e valor de cada pagamento.

- Chave lógica: `order_id` + `payment_sequential`.
- Tipos: principalmente cartão de crédito, boleto, voucher e cartão de débito.
- Um pedido pode ter mais de uma linha de pagamento (por exemplo, voucher + cartão). Para o total pago, agregue `payment_value` por `order_id` antes de uni-lo a itens.

### `olist_order_reviews_dataset.csv` — avaliações

Reúne **99.224 avaliações**, cobrindo **98.673 pedidos**.

- `review_score` vai de 1 a 5; notas 5 são as mais frequentes.
- Inclui título e mensagem opcionais, data de criação e data da resposta. A maioria dos títulos e muitas mensagens estão ausentes — ausência normalmente significa que o cliente apenas atribuiu a nota.
- A ligação é por `order_id`. Embora tipicamente haja uma avaliação por pedido, existem pedidos com mais de uma linha de review.

### `olist_products_dataset.csv` — catálogo de produtos

Catálogo de **32.951 produtos**, identificado por `product_id`.

- Categoria em português, tamanho do nome e descrição, quantidade de fotos, peso e dimensões.
- Há **73 categorias** cadastradas; 610 produtos não têm categoria nem métricas textuais/fotos preenchidas.
- Liga com itens por `product_id` e à tradução por `product_category_name`.

### `olist_sellers_dataset.csv` — vendedores

Traz **3.095 vendedores**, identificados por `seller_id`.

- Contém prefixo de CEP, cidade e UF de cada vendedor.
- Liga com itens por `seller_id` e pode ser associado à geolocalização pelo prefixo de CEP.

### `olist_geolocation_dataset.csv` — referência geográfica

Possui **1.000.163 linhas** com latitude, longitude, cidade e UF para prefixos de CEP; há **19.015 prefixos** distintos.

- A tabela tem várias coordenadas por prefixo de CEP. Antes de juntá-la a clientes ou vendedores, agregue por CEP (por exemplo, média/mediana de latitude e longitude) para evitar multiplicar linhas.
- Liga por `geolocation_zip_code_prefix` = CEP prefixo de clientes ou vendedores.

### `product_category_name_translation.csv` — tradução de categorias

Mapa com **71 traduções** entre `product_category_name` e `product_category_name_english`.

- Use para apresentar as categorias em inglês ou padronizar relatórios.
- Nem todas as 73 categorias observadas em produtos possuem tradução nesta tabela.

## Cuidados nas análises

- Use `orders` como base quando a unidade de análise for **pedido**; agregue tabelas de detalhe por `order_id` antes de juntar.
- Para faturamento de itens, use `sum(price + freight_value)` por pedido; para valor pago, use `sum(payment_value)` por pedido. Eles podem divergir por descontos, vouchers e arredondamentos.
- `customer_id` não é a melhor chave para analisar recompra; prefira `customer_unique_id`.
- Datas de entrega nulas não significam necessariamente erro: são esperadas em pedidos cancelados ou ainda não entregues.
- Mantenha CEP como identificador, não como número para cálculos; o zero à esquerda pode ser relevante.
