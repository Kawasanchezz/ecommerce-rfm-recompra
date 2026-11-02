"""Pipeline analítico: SQL -> EDA -> segmentação -> classificação -> regressão.

Etapas
1. Executa as consultas SQL (receita mensal, coortes, categorias, features por cliente).
2. Segmenta clientes por RFM com K-Means (k escolhido por silhouette).
3. Classificação: prevê quem recompra nos próximos 90 dias.
4. Regressão: prevê quanto quem recompra vai gastar nesses 90 dias.
5. Salva tabelas (CSV), gráficos (PNG) e métricas (JSON) em reports/.

Uso:
    python src/generate_data.py
    python src/analysis.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
REF_DATE = "2025-09-30"

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "loja.db"
SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"

NUMERIC_FEATURES = [
    "recencia_dias",
    "frequencia",
    "valor_total",
    "ticket_medio",
    "idade_conta_dias",
    "dias_entre_compras",
    "pct_cupom",
    "compras_ult_90d",
    "n_categorias",
    "taxa_cancelamento",
    "idade",
]
CATEGORICAL_FEATURES = ["canal", "estado"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def run_sql(conn: sqlite3.Connection, filename: str, params: dict | None = None) -> pd.DataFrame:
    query = (SQL_DIR / filename).read_text(encoding="utf-8")
    return pd.read_sql_query(query, conn, params=params or {})


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORTS / name, bbox_inches="tight")
    plt.close(fig)


def exploratory_analysis(conn: sqlite3.Connection) -> dict:
    monthly = run_sql(conn, "01_receita_mensal.sql")
    cohorts = run_sql(conn, "02_retencao_coortes.sql")
    categories = run_sql(conn, "03_receita_categoria_canal.sql")
    monthly.to_csv(REPORTS / "receita_mensal.csv", index=False)
    cohorts.to_csv(REPORTS / "retencao_coortes.csv", index=False)
    categories.to_csv(REPORTS / "receita_categoria_canal.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(monthly["mes"], monthly["receita"] / 1e3, color="#9db4d6", label="Receita mensal")
    ax.plot(monthly["mes"], monthly["media_movel_3m"] / 1e3, color="#1f3b73", lw=2, label="Média móvel 3m")
    ax.set_ylabel("Receita (R$ mil)")
    ax.set_title("Receita mensal de pedidos entregues")
    ax.set_xticks(range(0, len(monthly), 3))
    ax.set_xticklabels(monthly["mes"].iloc[::3], rotation=45, ha="right")
    ax.legend(frameon=False)
    save_fig(fig, "receita_mensal.png")

    heat = cohorts[cohorts["coorte"] <= "2024-12"].pivot(index="coorte", columns="mes_idx", values="retencao_pct")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(heat.values, aspect="auto", cmap="Blues", vmin=0, vmax=60)
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(0, heat.shape[0], 3))
    ax.set_yticklabels(heat.index[::3])
    ax.set_xlabel("Meses desde a primeira compra")
    ax.set_title("Retenção por coorte (% de clientes que compram no mês)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="%")
    save_fig(fig, "retencao_coortes.png")

    nov = monthly.loc[monthly["mes"] == "2024-11", "receita"].iloc[0]
    oct_ = monthly.loc[monthly["mes"] == "2024-10", "receita"].iloc[0]
    avg_m1 = cohorts.loc[(cohorts["mes_idx"] == 1) & (cohorts["coorte"] <= "2024-12"), "retencao_pct"].mean()
    return {
        "receita_total_entregue": round(float(monthly["receita"].sum()), 2),
        "pedidos_entregues": int(monthly["pedidos"].sum()),
        "ticket_medio": round(float(monthly["receita"].sum() / monthly["pedidos"].sum()), 2),
        "alta_nov_2024_vs_out_2024_pct": round(100 * (nov / oct_ - 1), 1),
        "retencao_media_mes1_pct": round(float(avg_m1), 1),
    }


def segment_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rfm = df[["recencia_dias", "frequencia", "valor_total"]]
    scaled = StandardScaler().fit_transform(np.log1p(rfm))

    scores = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(scaled)
        scores[k] = silhouette_score(scaled, labels, sample_size=4000, random_state=SEED)

    best_k = max(range(3, 7), key=lambda k: scores[k])
    model = KMeans(n_clusters=best_k, n_init=10, random_state=SEED)
    out = df.copy()
    out["cluster"] = model.fit_predict(scaled)

    profile = (
        out.groupby("cluster")
        .agg(
            clientes=("customer_id", "count"),
            recencia_media=("recencia_dias", "mean"),
            frequencia_media=("frequencia", "mean"),
            valor_medio=("valor_total", "mean"),
            ticket_medio=("ticket_medio", "mean"),
            taxa_recompra_90d=("recomprou_90d", "mean"),
        )
        .reset_index()
        .sort_values("valor_medio", ascending=False)
        .reset_index(drop=True)
    )
    median_recency = profile["recencia_media"].median()
    median_value = profile["valor_medio"].median()
    names = []
    for rank, row in profile.iterrows():
        value_label = "alto valor" if row["valor_medio"] >= median_value else "baixo valor"
        recency_label = "recente" if row["recencia_media"] <= median_recency else "inativo"
        names.append(f"{rank + 1}. {value_label}, {recency_label}")
    profile["segmento"] = names
    profile["pct_clientes"] = 100 * profile["clientes"] / profile["clientes"].sum()
    profile["pct_receita"] = [
        100 * out.loc[out["cluster"] == c, "valor_total"].sum() / out["valor_total"].sum()
        for c in profile["cluster"]
    ]
    out = out.merge(profile[["cluster", "segmento"]], on="cluster")
    profile = profile.round(2)
    profile.to_csv(REPORTS / "segmentos_rfm.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    axes[0].plot(list(scores), list(scores.values()), marker="o", color="#1f3b73")
    axes[0].axvline(best_k, ls="--", color="#c0392b")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title(f"Escolha de k (selecionado: {best_k})")
    for name, group in out.groupby("segmento"):
        axes[1].scatter(
            np.log1p(group["frequencia"]), np.log1p(group["valor_total"]), s=6, alpha=0.5, label=name
        )
    axes[1].set_xlabel("log(1 + frequência)")
    axes[1].set_ylabel("log(1 + valor total)")
    axes[1].set_title("Segmentos de clientes (RFM)")
    axes[1].legend(frameon=False, markerscale=3, fontsize=8)
    save_fig(fig, "segmentacao_rfm.png")

    metrics = {
        "k_escolhido": int(best_k),
        "silhouette_por_k": {str(k): round(float(v), 3) for k, v in scores.items()},
        "silhouette_escolhido": round(float(scores[best_k]), 3),
    }
    return out, metrics


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def classification(df: pd.DataFrame) -> dict:
    X, y = df[FEATURES], df["recomprou_90d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    models = {
        "Regressão Logística": Pipeline(
            [("prep", make_preprocessor(scale=True)), ("model", LogisticRegression(max_iter=1000))]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", make_preprocessor(scale=False)),
                ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=SEED)),
            ]
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, pipeline in models.items():
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc_cv_media": round(float(cv_auc.mean()), 3),
            "roc_auc_cv_desvio": round(float(cv_auc.std()), 3),
            "roc_auc_teste": round(float(roc_auc_score(y_test, proba)), 3),
            "pr_auc_teste": round(float(average_precision_score(y_test, proba)), 3),
            "precisao_teste": round(float(precision_score(y_test, pred)), 3),
            "recall_teste": round(float(recall_score(y_test, pred)), 3),
            "f1_teste": round(float(f1_score(y_test, pred)), 3),
        }
        fitted[name] = (pipeline, proba)

    best_name = max(results, key=lambda n: results[n]["roc_auc_teste"])
    best_pipeline, best_proba = fitted[best_name]

    ranking = pd.DataFrame({"y": y_test.values, "proba": best_proba}).sort_values("proba", ascending=False)
    ranking["decil"] = pd.qcut(np.arange(len(ranking)), 10, labels=range(1, 11))
    base_rate = ranking["y"].mean()
    lift = ranking.groupby("decil", observed=True).agg(clientes=("y", "size"), taxa_recompra=("y", "mean")).reset_index()
    lift["lift"] = lift["taxa_recompra"] / base_rate
    lift["ganho_acumulado_pct"] = 100 * (lift["taxa_recompra"] * lift["clientes"]).cumsum() / ranking["y"].sum()
    lift.round(3).to_csv(REPORTS / "lift_por_decil.csv", index=False)

    importance = permutation_importance(
        best_pipeline, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED
    )
    imp = (
        pd.DataFrame({"variavel": FEATURES, "queda_auc": importance.importances_mean})
        .sort_values("queda_auc", ascending=False)
        .reset_index(drop=True)
    )
    imp.round(4).to_csv(REPORTS / "importancia_variaveis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, (_, proba) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc_teste']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("Falso positivo")
    axes[0].set_ylabel("Verdadeiro positivo")
    axes[0].set_title("Curva ROC (teste)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(lift["decil"].astype(int), lift["lift"], color="#1f3b73")
    axes[1].axhline(1, ls="--", color="gray")
    axes[1].set_xlabel("Decil de propensão (1 = maior)")
    axes[1].set_ylabel("Lift")
    axes[1].set_title("Lift por decil")
    top = imp.head(8).iloc[::-1]
    axes[2].barh(top["variavel"], top["queda_auc"], color="#4c78a8")
    axes[2].set_xlabel("Queda no AUC ao embaralhar")
    axes[2].set_title("Importância das variáveis")
    save_fig(fig, "modelo_recompra.png")

    return {
        "taxa_base_recompra_90d": round(float(y.mean()), 3),
        "modelos": results,
        "melhor_modelo": best_name,
        "lift_decil_1": round(float(lift.loc[0, "lift"]), 2),
        "ganho_top_20pct_clientes": round(float(lift.loc[1, "ganho_acumulado_pct"]), 1),
        "ganho_top_30pct_clientes": round(float(lift.loc[2, "ganho_acumulado_pct"]), 1),
        "variaveis_mais_importantes": imp["variavel"].head(5).tolist(),
    }


def regression(df: pd.DataFrame) -> dict:
    buyers = df[df["recomprou_90d"] == 1]
    X, y = buyers[FEATURES], buyers["gasto_90d"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
    models = {
        "Baseline (média)": Pipeline([("prep", make_preprocessor(scale=False)), ("model", DummyRegressor())]),
        "Ridge": Pipeline([("prep", make_preprocessor(scale=True)), ("model", Ridge(alpha=1.0))]),
        "Gradient Boosting": Pipeline(
            [
                ("prep", make_preprocessor(scale=False)),
                ("model", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)),
            ]
        ),
    }
    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results[name] = {
            "mae": round(float(mean_absolute_error(y_test, pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "r2": round(float(r2_score(y_test, pred)), 3),
        }
    return {"clientes_que_recompraram": int(len(buyers)), "gasto_medio_90d": round(float(y.mean()), 2), "modelos": results}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        eda = exploratory_analysis(conn)
        features = run_sql(conn, "04_features_cliente.sql", {"ref_date": REF_DATE})

    segmented, seg_metrics = segment_customers(features)
    segmented.to_csv(REPORTS / "clientes_segmentados.csv", index=False)

    metrics = {
        "data_de_corte": REF_DATE,
        "clientes_analisados": int(len(features)),
        "eda": eda,
        "segmentacao": seg_metrics,
        "classificacao": classification(features),
        "regressao": regression(features),
    }
    (REPORTS / "metricas.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
