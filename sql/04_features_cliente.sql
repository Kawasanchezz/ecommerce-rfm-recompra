-- Tabela analítica por cliente, usada na segmentação e nos modelos.
--
-- Parâmetro nomeado: :ref_date (data de corte, formato YYYY-MM-DD).
--   * Todas as variáveis explicativas usam SOMENTE pedidos até :ref_date.
--   * Os alvos (recomprou_90d, gasto_90d) olham os 90 dias seguintes.
-- Essa separação temporal evita vazamento de dados (data leakage).
WITH historico AS (
    SELECT
        customer_id,
        COUNT(*)                   AS frequencia,
        SUM(order_total)           AS valor_total,
        AVG(order_total)           AS ticket_medio,
        MIN(order_date)            AS primeira_compra,
        MAX(order_date)            AS ultima_compra,
        AVG(coupon_used)           AS pct_cupom,
        SUM(CASE WHEN order_date > date(:ref_date, '-90 days') THEN 1 ELSE 0 END)
                                   AS compras_ult_90d
    FROM orders
    WHERE status = 'entregue' AND order_date <= :ref_date
    GROUP BY customer_id
),
categorias AS (
    SELECT o.customer_id, COUNT(DISTINCT i.category) AS n_categorias
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    WHERE o.status = 'entregue' AND o.order_date <= :ref_date
    GROUP BY o.customer_id
),
cancelamentos AS (
    SELECT
        customer_id,
        1.0 * SUM(CASE WHEN status = 'cancelado' THEN 1 ELSE 0 END) / COUNT(*)
            AS taxa_cancelamento
    FROM orders
    WHERE order_date <= :ref_date
    GROUP BY customer_id
),
futuro AS (
    SELECT customer_id, COUNT(*) AS pedidos_futuros, SUM(order_total) AS gasto_futuro
    FROM orders
    WHERE status = 'entregue'
      AND order_date >  :ref_date
      AND order_date <= date(:ref_date, '+90 days')
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.channel                                                         AS canal,
    c.state                                                           AS estado,
    c.age                                                             AS idade,
    CAST(julianday(:ref_date) - julianday(h.ultima_compra) AS INTEGER) AS recencia_dias,
    h.frequencia,
    ROUND(h.valor_total, 2)                                           AS valor_total,
    ROUND(h.ticket_medio, 2)                                          AS ticket_medio,
    CAST(julianday(:ref_date) - julianday(c.signup_date) AS INTEGER)  AS idade_conta_dias,
    ROUND(
        (julianday(h.ultima_compra) - julianday(h.primeira_compra))
        / NULLIF(h.frequencia - 1, 0),
        1
    )                                                                 AS dias_entre_compras,
    ROUND(h.pct_cupom, 3)                                             AS pct_cupom,
    h.compras_ult_90d,
    COALESCE(cat.n_categorias, 0)                                     AS n_categorias,
    ROUND(COALESCE(canc.taxa_cancelamento, 0), 3)                     AS taxa_cancelamento,
    CASE WHEN f.pedidos_futuros > 0 THEN 1 ELSE 0 END                 AS recomprou_90d,
    ROUND(COALESCE(f.gasto_futuro, 0), 2)                             AS gasto_90d
FROM customers c
JOIN historico h             ON h.customer_id    = c.customer_id
LEFT JOIN categorias cat     ON cat.customer_id  = c.customer_id
LEFT JOIN cancelamentos canc ON canc.customer_id = c.customer_id
LEFT JOIN futuro f           ON f.customer_id    = c.customer_id;
