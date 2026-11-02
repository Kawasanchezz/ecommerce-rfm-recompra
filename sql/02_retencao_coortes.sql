-- Retenção por coorte: % de clientes de cada mês de primeira compra que voltam
-- a comprar nos 12 meses seguintes (mes_idx = meses desde a primeira compra).
WITH primeira AS (
    SELECT customer_id, MIN(order_date) AS primeira_data
    FROM orders
    WHERE status = 'entregue'
    GROUP BY customer_id
),
atividade AS (
    SELECT DISTINCT
        o.customer_id,
        strftime('%Y-%m', p.primeira_data) AS coorte,
        (CAST(strftime('%Y', o.order_date) AS INTEGER)
            - CAST(strftime('%Y', p.primeira_data) AS INTEGER)) * 12
        + CAST(strftime('%m', o.order_date) AS INTEGER)
            - CAST(strftime('%m', p.primeira_data) AS INTEGER) AS mes_idx
    FROM orders o
    JOIN primeira p ON p.customer_id = o.customer_id
    WHERE o.status = 'entregue'
),
tamanho_coorte AS (
    SELECT coorte, COUNT(*) AS clientes
    FROM atividade
    WHERE mes_idx = 0
    GROUP BY coorte
)
SELECT
    a.coorte,
    a.mes_idx,
    COUNT(DISTINCT a.customer_id)                                    AS clientes_ativos,
    ROUND(100.0 * COUNT(DISTINCT a.customer_id) / t.clientes, 1)    AS retencao_pct
FROM atividade a
JOIN tamanho_coorte t ON t.coorte = a.coorte
WHERE a.mes_idx BETWEEN 0 AND 12
GROUP BY a.coorte, a.mes_idx
ORDER BY a.coorte, a.mes_idx;
