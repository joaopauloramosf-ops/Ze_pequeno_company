import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("tabelas_vendas_ZePequeno_LIMPA.csv", encoding="utf-8-sig")


df["Data"] = pd.to_datetime(df["Data"])


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

# --- Gráfico de Baixo: PAÇOCAS (Amendoim) ---
bars2 = ax2.bar(
    vendas_pacocas_ano["Ano"].astype(str),
    vendas_pacocas_ano["Valor_Venda"],
    color="#DAA520",  # Tom Amarelo / Amendoim
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


import numpy as np
from scipy.optimize import curve_fit

df = df.sort_values("Data")
def von_bertalanffy(t, L_inf, k, t0):
    return L_inf * (1 - np.exp(-k * (t - t0)))


# 3. Seleciona AUTOMATICAMENTE as 4 Regiões com Maior Volume de Vendas (R$)
top_4_regioes = (
    df.groupby("Região")["Valor_Venda"]
    .sum()
    .nlargest(4)  # Pega as 4 maiores
    .index.tolist()
)

# Criando a grade de subplots (2x2 Facet Wrap)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

print("=" * 75)
print("  MODELO DE VON BERTALANFFY POR REGIÃO (TOP 4 MAIORES VENDAS)")
print("=" * 75)

for i, regiao in enumerate(top_4_regioes):
    ax = axes[i]
    df_reg = df[df["Região"] == regiao].copy()

    # Agrupa por data e gera a série temporal acumulada
    df_agrup = df_reg.groupby("Data")["Valor_Venda"].sum().reset_index()
    df_agrup["Acumulado"] = df_agrup["Valor_Venda"].cumsum()

    # Eixo X contínuo (dias a partir da primeira venda)
    t_min = df_agrup["Data"].min()
    df_agrup["Dias"] = (df_agrup["Data"] - t_min).dt.days

    X = df_agrup["Dias"].values
    Y = df_agrup["Acumulado"].values

    # Chutes iniciais para o ajuste [L_inf, k, t0]
    chute_inicial = [Y.max() * 1.2, 0.005, 0]

    try:
        # Ajuste do Modelo de von Bertalanffy
        popt, _ = curve_fit(
            von_bertalanffy,
            X,
            Y,
            p0=chute_inicial,
            bounds=(0, [np.inf, 1, 365]),
            maxfev=10000,
        )
        L_inf, k, t0 = popt

        # Linha do Modelo
        X_linha = np.linspace(X.min(), X.max(), 100)
        Y_linha = von_bertalanffy(X_linha, *popt)
        datas_linha = t_min + pd.to_timedelta(X_linha, unit="D")

        # Plot dos Dados Reais
        ax.scatter(
            df_agrup["Data"],
            Y,
            color="#2b5c8f",
            alpha=0.6,
            s=25,
            label="Real Acumulado",
        )

        # Plot da Curva de von Bertalanffy
        ax.plot(
            datas_linha,
            Y_linha,
            color="#d95f02",
            linewidth=2,
            linestyle="--",
            label="von Bertalanffy",
        )

        # String da Fórmula Estimada
        equacao_str = (
            f"y(t) = {L_inf:,.0f} · [1 - e^({-k:.4f} · (t - {t0:.1f}))]"
        )

        # Título do Painel com Nome da Região e Equação
        ax.set_title(
            f"Região: {regiao}\n{equacao_str}", fontsize=10, fontweight="bold"
        )

        # Print no terminal para documentação
        print(f"\n📍 Região: {regiao}")
        print(f"   • Teto Estimado (L∞): R$ {L_inf:,.2f}")
        print(f"   • Taxa (k): {k:.4f}")
        print(f"   • Offset (t0): {t0:.2f} dias")
        print(f"   • Equação: {equacao_str}")

    except Exception as e:
        ax.set_title(
            f"Região: {regiao}\n(Erro no ajuste do modelo)", fontsize=10
        )
        print(f"Erro ao ajustar {regiao}: {e}")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="x", rotation=30)

# Ajustes do Layout
fig.supxlabel("Data da Venda", fontsize=12)
fig.supylabel("Vendas Acumuladas (R$)", fontsize=12)
plt.tight_layout()
plt.show()


import matplotlib.ticker as mtick
df["Ano_Mes"] = df["Data"].dt.to_period("M")

# 2. Identifica as TOP 4 Regiões em volume total de vendas
top_4_regioes = (
    df.groupby("Região")["Valor_Venda"].sum().nlargest(4).index.tolist()
)

# 3. Preparação da figura com Subplots (2x2 Facet Wrap)
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

    # Agrupa por Ano_Mes e Canal
    vendas_canal = (
        df_reg.groupby(["Ano_Mes", "Canal"])["Valor_Venda"]
        .sum()
        .unstack(fill_value=0)
    )

    # Garante que ambos os canais existam na tabela
    for canal in ["E-commerce", "Loja Física"]:
        if canal not in vendas_canal.columns:
            vendas_canal[canal] = 0

    # Calcula a participação % de cada canal no mês
    vendas_canal["Total"] = vendas_canal.sum(axis=1)
    vendas_canal["Pct_ECommerce"] = (
        vendas_canal["E-commerce"] / vendas_canal["Total"]
    ) * 100
    vendas_canal["Pct_Fisica"] = (
        vendas_canal["Loja Física"] / vendas_canal["Total"]
    ) * 100

    datas = vendas_canal.index.to_timestamp()

    # Plot das linhas de tendência de cada canal
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

    # Linha de tendência (regressão linear simples) do E-Commerce
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