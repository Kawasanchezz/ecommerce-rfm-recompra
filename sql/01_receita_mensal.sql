WITH mensal AS (
    SELECT
        strftime('%Y-%m', order_date)  AS mes,
        COUNT(*)                       AS pedidos,
        COUNT(DISTINCT customer_id)    AS clientes_ativos,
        ROUND(SUM(order_total), 2)     AS receita
    FROM orders
    WHERE status = 'entregue'
    GROUP BY 1
)
SELECT
    mes,
    pedidos,
    clientes_ativos,
    receita,
    ROUND(receita / pedidos, 2) AS ticket_medio,
    ROUND(
        100.0 * (receita - LAG(receita) OVER (ORDER BY mes))
        / LAG(receita) OVER (ORDER BY mes),
        1
    ) AS variacao_mom_pct,
    ROUND(
        AVG(receita) OVER (ORDER BY mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW),
        2
    ) AS media_movel_3m
FROM mensal
ORDER BY mes;
