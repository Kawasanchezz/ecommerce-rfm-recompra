"""Gera uma base SINTÉTICA de e-commerce em SQLite (data/loja.db).

Os dados são fictícios e reprodutíveis (seed fixa). Eles simulam clientes com
perfis de compra diferentes (alto valor, regular, ocasional e baixo engajamento),
sazonalidade de fim de ano, cupons e cancelamentos, para que as análises do
projeto tenham padrões reais para descobrir.

Uso:
    python src/generate_data.py
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 6000
DATA_START = date(2023, 1, 1)
DATA_END = date(2025, 12, 31)
LAST_SIGNUP = date(2025, 9, 15)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "loja.db"

CHANNELS = ["organico", "busca_paga", "social", "email", "indicacao"]
CHANNEL_PROBS = [0.30, 0.25, 0.20, 0.15, 0.10]

# Probabilidade de cada perfil (alto valor, regular, ocasional, baixo engajamento) por canal.
SEGMENT_PROBS_BY_CHANNEL = {
    "organico": [0.10, 0.33, 0.42, 0.15],
    "busca_paga": [0.08, 0.30, 0.42, 0.20],
    "social": [0.07, 0.28, 0.43, 0.22],
    "email": [0.18, 0.38, 0.34, 0.10],
    "indicacao": [0.20, 0.38, 0.30, 0.12],
}

# Parâmetros por perfil: pedidos/mês, vida média (meses), ticket médio (R$), cupom, cancelamento.
SEGMENT_PARAMS = [
    {"rate": 1.30, "life_months": 30, "ticket": 320, "coupon": 0.15, "cancel": 0.03},
    {"rate": 0.55, "life_months": 14, "ticket": 180, "coupon": 0.25, "cancel": 0.04},
    {"rate": 0.20, "life_months": 8, "ticket": 110, "coupon": 0.35, "cancel": 0.05},
    {"rate": 0.07, "life_months": 4, "ticket": 80, "coupon": 0.55, "cancel": 0.08},
]

STATES = ["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "PE", "GO", "DF"]
STATE_PROBS = [0.35, 0.12, 0.11, 0.07, 0.07, 0.06, 0.05, 0.05, 0.06, 0.06]

CATEGORIES = ["Eletronicos", "Moda", "Casa", "Beleza", "Esporte", "Livros", "Pet"]
CATEGORY_PROBS = [0.18, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05]

# Sazonalidade: novembro (Black Friday) e dezembro vendem mais.
SEASONALITY = {11: 1.8, 12: 1.4}
SEASONALITY_MAX = max(SEASONALITY.values())

SCHEMA = """
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    signup_date TEXT    NOT NULL,
    channel     TEXT    NOT NULL,
    state       TEXT    NOT NULL,
    age         INTEGER NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
    order_date  TEXT    NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('entregue', 'cancelado')),
    coupon_used INTEGER NOT NULL CHECK (coupon_used IN (0, 1)),
    order_total REAL    NOT NULL
);

CREATE TABLE order_items (
    item_id  INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders (order_id),
    category TEXT    NOT NULL,
    price    REAL    NOT NULL
);
"""

INDEXES = """
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date     ON orders (order_date);
CREATE INDEX idx_items_order     ON order_items (order_id);
"""


def build_customers(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Cria os clientes e devolve também o perfil latente de cada um (não vai para o banco)."""
    total_days = (LAST_SIGNUP - DATA_START).days
    # Beta(1.3, 1) concentra cadastros mais recentes: a base cresce ao longo do tempo.
    signup_offsets = (rng.beta(1.3, 1.0, N_CUSTOMERS) * total_days).astype(int)
    signup_dates = [DATA_START + timedelta(days=int(d)) for d in signup_offsets]

    channels = rng.choice(CHANNELS, size=N_CUSTOMERS, p=CHANNEL_PROBS)
    segments = np.array(
        [rng.choice(4, p=SEGMENT_PROBS_BY_CHANNEL[ch]) for ch in channels]
    )
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(1, N_CUSTOMERS + 1),
            "signup_date": [d.isoformat() for d in signup_dates],
            "channel": channels,
            "state": rng.choice(STATES, size=N_CUSTOMERS, p=STATE_PROBS),
            "age": np.clip(rng.normal(34, 10, N_CUSTOMERS), 18, 70).astype(int),
        }
    )
    return customers, segments


def build_orders(
    rng: np.random.Generator, customers: pd.DataFrame, segments: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders: list[dict] = []
    items: list[dict] = []
    order_id = 0
    item_id = 0

    for row, seg in zip(customers.itertuples(index=False), segments):
        params = SEGMENT_PARAMS[seg]
        signup = date.fromisoformat(row.signup_date)
        first_order = signup + timedelta(days=int(rng.integers(0, 7)))
        if first_order > DATA_END:
            continue

        life_days = rng.exponential(params["life_months"] * 30.4)
        active_end = min(first_order + timedelta(days=int(life_days)), DATA_END)
        active_days = max((active_end - first_order).days, 0)

        order_dates = [first_order]
        n_extra = rng.poisson(params["rate"] * (active_days / 30.4) * 1.5)
        for offset in rng.integers(0, active_days + 1, size=n_extra):
            candidate = first_order + timedelta(days=int(offset))
            weight = SEASONALITY.get(candidate.month, 1.0)
            if rng.random() <= weight / SEASONALITY_MAX:
                order_dates.append(candidate)

        for order_date in sorted(order_dates):
            order_id += 1
            n_items = min(1 + int(rng.poisson(0.7)), 4)
            basket_value = rng.lognormal(np.log(params["ticket"]), 0.4)
            shares = rng.dirichlet(np.ones(n_items))
            categories = rng.choice(CATEGORIES, size=n_items, p=CATEGORY_PROBS)
            prices = np.round(basket_value * shares, 2)

            coupon_used = int(rng.random() < params["coupon"])
            status = "cancelado" if rng.random() < params["cancel"] else "entregue"
            total = round(float(prices.sum()) * (0.9 if coupon_used else 1.0), 2)

            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": int(row.customer_id),
                    "order_date": order_date.isoformat(),
                    "status": status,
                    "coupon_used": coupon_used,
                    "order_total": total,
                }
            )
            for category, price in zip(categories, prices):
                item_id += 1
                items.append(
                    {
                        "item_id": item_id,
                        "order_id": order_id,
                        "category": str(category),
                        "price": float(price),
                    }
                )

    return pd.DataFrame(orders), pd.DataFrame(items)


def write_database(
    customers: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame
) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)
        customers.to_sql("customers", conn, if_exists="append", index=False)
        orders.to_sql("orders", conn, if_exists="append", index=False)
        items.to_sql("order_items", conn, if_exists="append", index=False)
        conn.executescript(INDEXES)


def main() -> None:
    rng = np.random.default_rng(SEED)
    customers, segments = build_customers(rng)
    orders, items = build_orders(rng, customers, segments)
    write_database(customers, orders, items)
    print(
        f"Base criada em {DB_PATH.relative_to(ROOT)}: "
        f"{len(customers):,} clientes, {len(orders):,} pedidos, {len(items):,} itens."
    )


if __name__ == "__main__":
    main()
