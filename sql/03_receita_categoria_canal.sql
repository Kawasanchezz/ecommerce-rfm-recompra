-- Receita bruta por canal de aquisição e categoria, com ranking dentro de cada canal
-- (RANK) e participação da categoria na receita do canal.
WITH receita AS (
    SELECT
        c.channel                  AS canal,
        i.category                 AS categoria,
        COUNT(DISTINCT o.order_id) AS pedidos,
        ROUND(SUM(i.price), 2)     AS receita_bruta
    FROM order_items i
    JOIN orders o    ON o.order_id    = i.order_id
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'entregue'
    GROUP BY c.channel, i.category
)
SELECT
    canal,
    categoria,
    pedidos,
    receita_bruta,
    RANK() OVER (PARTITION BY canal ORDER BY receita_bruta DESC) AS ranking_no_canal,
    ROUND(100.0 * receita_bruta / SUM(receita_bruta) OVER (PARTITION BY canal), 1) AS participacao_pct
FROM receita
ORDER BY canal, ranking_no_canal;
