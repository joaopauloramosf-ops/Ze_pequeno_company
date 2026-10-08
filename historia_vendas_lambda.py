import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tabelas_vendas_ZePequeno_LIMPA.csv", encoding="utf-8-sig")


df["Data"] = pd.to_datetime(df["Data"])

#verificando o total de vendas de cada produto ao longo dos anos

total_pacocas = (
    df[df["Produto"].apply(lambda p: p == "Paçoca")]["Valor_Venda"].sum()
)
total_rolhas = (
    df[df["Produto"].apply(lambda p: p == "Rolhas")]["Valor_Venda"].sum()
)
faturamento_total = total_pacocas + total_rolhas

print("=" * 60)
print("  HISTÓRIA DE VENDAS ZÉ PEQUENO: ECONOMIA CIRCULAR EM AÇÃO")
print("=" * 60)
print(f"💰 Faturamento Total Acumulado: R$ {faturamento_total:,.2f}")
print(
    f"🥜 Paçoca  (Produto Principal): R$ {total_pacocas:,.2f} ({total_pacocas/faturamento_total:.1%})"
)
print(
    f"🍾 Rolhas  (Subproduto Casca): R$ {total_rolhas:,.2f} ({total_rolhas/faturamento_total:.1%})"
)
print("=" * 60)


vendas_rolhas_ano = (
    df[df["Produto"] == "Rolhas"]
    .groupby("Ano")["Valor_Venda"]
    .sum()
    .reset_index()
)


vendas_pacocas_ano = (
    df[df["Produto"] == "Paçoca"]
    .groupby("Ano")["Valor_Venda"]
    .sum()
    .reset_index()
)


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True
)  


bars1 = ax1.bar(
    vendas_rolhas_ano["Ano"].astype(str),
    vendas_rolhas_ano["Valor_Venda"],
    color="#8B4513",  # Tom Marrom / Cortiça
    edgecolor="black",
)
ax1.set_title(
    "Evolução Anual: Vendas de Rolhas (Aproveitamento de Casca)",
    fontsize=12,
    fontweight="bold",
)
ax1.set_ylabel("Faturamento (R$)")
ax1.grid(axis="y", linestyle="--", alpha=0.7)


for bar in bars1:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"R$ {yval:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

bars2 = ax2.bar(
    vendas_pacocas_ano["Ano"].astype(str),
    vendas_pacocas_ano["Valor_Venda"],
    color="#DAA520",  
    edgecolor="black",
)
ax2.set_title(
    "Evolução Anual: Vendas de Paçoca (Produto Base)",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel("Ano")
ax2.set_ylabel("Faturamento (R$)")
ax2.grid(axis="y", linestyle="--", alpha=0.7)


for bar in bars2:
    yval = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"R$ {yval:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

#Fazendo analise de estagnação do modelo de negocio por cada região

import sys
import warnings
import numpy as np
from scipy.optimize import curve_fit


warnings.filterwarnings("ignore")


def von_bertalanffy(t, L_inf, k, t0):
    return L_inf * (1 - np.exp(-k * (t - t0)))


top_4_regioes = (
    df.groupby("Região")["Valor_Venda"].sum().nlargest(4).index.tolist()
)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
axes = axes.flatten()


resultados_prints = []

for i, regiao in enumerate(top_4_regioes):
    ax = axes[i]
    df_reg = df[df["Região"] == regiao].copy()

    
    df_mensal = (
        df_reg.set_index("Data")
        .resample("MS")["Valor_Venda"]
        .sum()
        .reset_index()
    )
    df_mensal["Acumulado"] = df_mensal["Valor_Venda"].cumsum()

    
    t_min = df_mensal["Data"].min()
    df_mensal["Dias"] = (df_mensal["Data"] - t_min).dt.days

    X = df_mensal["Dias"].values
    Y = df_mensal["Acumulado"].values

    max_y = Y.max()
    chute_inicial = [max_y * 1.1, 0.003, 0]

    try:
        popt, _ = curve_fit(
            von_bertalanffy,
            X,
            Y,
            p0=chute_inicial,
            bounds=(0, [max_y * 3, 0.1, 365]),
            maxfev=20000,
        )
        L_inf, k, t0 = popt

        
        X_linha = np.linspace(X.min(), X.max(), 100)
        Y_linha = von_bertalanffy(X_linha, *popt)
        datas_linha = t_min + pd.to_timedelta(X_linha, unit="D")

        
        ax.scatter(
            df_mensal["Data"],
            Y,
            color="#2b5c8f",
            alpha=0.7,
            label="Real Acumulado",
            s=30,
        )
        ax.plot(
            datas_linha,
            Y_linha,
            color="#d95f02",
            linewidth=2,
            linestyle="--",
            label="von Bertalanffy",
        )

        equacao_str = (
            f"y(t) = {L_inf:,.0f} · [1 - e^(-{k:.4f} · (t - {t0:.1f}))]"
        )
        ax.set_title(
            f"Região: {regiao}\n{equacao_str}", fontsize=9, fontweight="bold"
        )

        
        resultados_prints.append(
            f"📍 Região: {regiao}\n"
            f"   • Teto Estimado (L∞): R$ {L_inf:,.2f}\n"
            f"   • Taxa (k): {k:.4f}\n"
            f"   • Offset (t0): {t0:.1f} dias\n"
            f"   • Equação: {equacao_str}\n"
        )

    except Exception as e:
        ax.set_title(f"Região: {regiao}\n(Erro de ajuste)", fontsize=9)
        resultados_prints.append(f"❌ Região: {regiao} -> Erro no ajuste: {e}\n")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="x", rotation=30)

fig.supxlabel("Data da Venda", fontsize=12)
fig.supylabel("Vendas Acumuladas (R$)", fontsize=12)
plt.tight_layout()
plt.show()


print("\n" + "=" * 75, flush=True)
print(
    "  PARÂMETROS DO MODELO DE VON BERTALANFFY (TOP 4 REGIONAL)", flush=True
)
print("=" * 75 + "\n", flush=True)

for texto in resultados_prints:
    print(texto, flush=True)


import matplotlib.ticker as mtick
df["Ano_Mes"] = df["Data"].dt.to_period("M")


top_4_regioes = (
    df.groupby("Região")["Valor_Venda"].sum().nlargest(4).index.tolist()
)


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

print("=" * 80)
print(
    "  ANÁLISE 3: TENDÊNCIA E-COMMERCE VS. LOJA FÍSICA AO LONGO DO TEMPO (TOP 4)"
)
print("=" * 80)

for i, regiao in enumerate(top_4_regioes):
    ax = axes[i]
    df_reg = df[df["Região"] == regiao].copy()

    
    vendas_canal = (
        df_reg.groupby(["Ano_Mes", "Canal"])["Valor_Venda"]
        .sum()
        .unstack(fill_value=0)
    )



   
    vendas_canal["Total"] = vendas_canal.sum(axis=1)
    vendas_canal["Pct_ECommerce"] = (
        vendas_canal["E-commerce"] / vendas_canal["Total"]
    ) * 100
    vendas_canal["Pct_Fisica"] = (
        vendas_canal["Loja Física"] / vendas_canal["Total"]
    ) * 100

    datas = vendas_canal.index.to_timestamp()

    
    ax.plot(
        datas,
        vendas_canal["Pct_ECommerce"],
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="E-Commerce",
    )
    ax.plot(
        datas,
        vendas_canal["Pct_Fisica"],
        marker="s",
        linewidth=2.5,
        color="#2ca02c",
        linestyle="--",
        label="Loja Física",
    )

    
    if len(datas) > 1:
        x_numeric = np.arange(len(datas))
        z = np.polyfit(x_numeric, vendas_canal["Pct_ECommerce"], 1)
        p = np.poly1d(z)
        ax.plot(
            datas,
            p(x_numeric),
            color="#d62728",
            linestyle=":",
            linewidth=1.8,
            label="Tendência E-Commerce",
        )

        inclinacao = z[0]
        tendencia_str = (
            "🚀 Em alta"
            if inclinacao > 0.5
            else ("📉 Em queda" if inclinacao < -0.5 else "➡️ Estável")
        )
        print(f"📍 {regiao}: Tendência do E-Commerce é {tendencia_str}")

    ax.set_title(f"Região: {regiao}", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=8)

fig.supxlabel("Período (Mês/Ano)", fontsize=12)
fig.supylabel("Participação nas Vendas (%)", fontsize=12)
plt.tight_layout()
plt.show()
