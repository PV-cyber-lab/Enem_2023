import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ENEM 2023 · Ceará",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
NOTAS = [
    "Nota Ciências da Natureza", "Nota Ciências Humanas",
    "Nota Linguagens e Códigos", "Nota Matemática", "Nota Redação",
]
NOTAS_SHORT = ["C. Natureza", "C. Humanas", "Linguagens", "Matemática", "Redação"]

PRESENCA = [
    "Presença em Ciências da Natureza", "Presença em Ciências Humanas",
    "Presença em Linguagens e Códigos", "Presença em Matemática",
]

COLS_LOAD = [
    "Sexo", "Faixa Etária", "Cor/Raça", "Treineiro",
    "Tipo de Escola do Ensino Médio", "Sigla da UF da Escola",
    "Situação de Conclusão do Ensino Médio",
    "Renda Mensal Familiar",
    "Escolaridade do Pai/Responsável Homem",
    "Escolaridade da Mãe/Responsável Mulher",
    "Acesso à Internet na Residência",
    "Computador na Residência",
    "Quantidade de Pessoas na Residência",
] + PRESENCA + NOTAS

RENDA_ORDER = [
    "Nenhuma renda", "Até R$ 1.320,00",
    "De R$ 1.320,01 até R$ 1.980,00", "De R$ 1.980,01 até R$ 2.640,00",
    "De R$ 2.640,01 até R$ 3.300,00", "De R$ 3.300,01 até R$ 3.960,00",
    "De R$ 3.960,01 até R$ 5.280,00", "De R$ 5.280,01 até R$ 6.600,00",
    "De R$ 6.600,01 até R$ 7.920,00", "De R$ 7.920,01 até R$ 10.560,00",
    "Acima de R$ 10.560,00",
]

FAIXA_ORDER = (
    ["Menor de 17"] +
    [f"{i} anos" for i in range(17, 26)] +
    ["26-30", "31-35", "36-40", "41-45", "46-50",
     "51-55", "56-60", "61-65", "66-70", "Acima de 70"]
)

CE_COLOR  = "#4C9BE8"
COLORWAY  = [CE_COLOR, "#E8754C", "#4CE8A0", "#E8C44C", "#A04CE8", "#E84CA0"]


# ══════════════════════════════════════════════════════════════════
# CARREGAMENTO OTIMIZADO
# ══════════════════════════════════════════════════════════════════
def _faixa(x):
    if not isinstance(x, str):
        return "Não inf."
    nums = re.findall(r"\d+", x)
    if not nums:
        return x
    n = int(nums[0])
    if n < 17:  return "Menor de 17"
    if n <= 25: return f"{n} anos"
    if n <= 30: return "26-30"
    if n <= 35: return "31-35"
    if n <= 40: return "36-40"
    if n <= 45: return "41-45"
    if n <= 50: return "46-50"
    if n <= 55: return "51-55"
    if n <= 60: return "56-60"
    if n <= 65: return "61-65"
    if n <= 70: return "66-70"
    return "Acima de 70"


@st.cache_data(show_spinner="Carregando dados…")
def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLS_LOAD)
    df = df[(df[PRESENCA] == "Presente").all(axis=1)].drop(columns=PRESENCA).copy()
    df.dropna(subset=NOTAS, inplace=True)
    df["Média Geral"] = df[NOTAS].mean(axis=1).round(2)
    df["Faixa Etária"] = df["Faixa Etária"].apply(_faixa)
    df["Treineiro"] = df["Treineiro"].astype(str).str.strip().str.upper().isin(["1","SIM","S","TRUE"])
    cat_cols = [
        "Sexo", "Cor/Raça", "Tipo de Escola do Ensino Médio", "Sigla da UF da Escola",
        "Situação de Conclusão do Ensino Médio", "Renda Mensal Familiar",
        "Escolaridade do Pai/Responsável Homem", "Escolaridade da Mãe/Responsável Mulher",
        "Acesso à Internet na Residência", "Computador na Residência",
        "Quantidade de Pessoas na Residência",
    ]
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def get_data():
    df_ce = load("https://drive.google.com/uc?export=download&id=1vSH8dq5_Hm0hB-8r9fs9mtdL9sbqWrEF")
    return df_ce


df_ce_full = get_data()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def desc(s: pd.Series) -> dict:
    s = s.dropna()
    return dict(N=len(s), Média=s.mean(), DP=s.std(),
                Min=s.min(), Q1=s.quantile(.25), Med=s.median(),
                Q3=s.quantile(.75), Max=s.max(),
                IQR=s.quantile(.75) - s.quantile(.25))


def group_stats(df: pd.DataFrame, col: str, nota="Média Geral") -> pd.DataFrame:
    g = (df.groupby(col)[nota]
         .agg(N="count", Média="mean", Mediana="median", DP="std",
              Q1=lambda x: x.quantile(.25),
              Q3=lambda x: x.quantile(.75))
         .round(2).reset_index())
    g["IQR"] = (g["Q3"] - g["Q1"]).round(2)
    return g.sort_values("Média", ascending=False)


def reindex_col(df_g, col, order):
    ordem = [r for r in order if r in df_g[col].values]
    return df_g.set_index(col).reindex(ordem).dropna(how="all").reset_index()


def legend_top(fig):
    """Legenda horizontal acima do gráfico (para barras/radar)."""
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ))
    return fig


def legend_right(fig):
    """Legenda vertical à direita (para gráficos de linha com título longo)."""
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="middle",
        y=0.5,
        xanchor="left",
        x=1.01,
    ))
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Filtros")

sexo_opts   = sorted(df_ce_full["Sexo"].dropna().unique())
escola_opts = sorted(df_ce_full["Tipo de Escola do Ensino Médio"].dropna().unique())

sexo_sel   = st.sidebar.multiselect("Sexo",   sexo_opts,   default=sexo_opts)
escola_sel = st.sidebar.multiselect("Escola", escola_opts, default=escola_opts)
trein_sel  = st.sidebar.selectbox("Treineiro", ["Todos", "Não", "Sim"], index=1)

st.sidebar.divider()

all_faixas = df_ce_full["Faixa Etária"].dropna().unique()
faixa_opts = [f for f in FAIXA_ORDER if f in all_faixas]
faixa_sel  = st.sidebar.multiselect("Faixa Etária", faixa_opts, default=faixa_opts)

st.sidebar.divider()

all_renda = df_ce_full["Renda Mensal Familiar"].dropna().unique()
renda_opts = [r for r in RENDA_ORDER if r in all_renda]
renda_sel  = st.sidebar.multiselect("Renda Familiar", renda_opts, default=renda_opts)


def filtrar(df):
    m = (
        df["Sexo"].isin(sexo_sel)
        & df["Tipo de Escola do Ensino Médio"].isin(escola_sel)
        & df["Faixa Etária"].isin(faixa_sel)
        & df["Renda Mensal Familiar"].isin(renda_sel)
    )
    if trein_sel == "Sim":   m &= df["Treineiro"]
    elif trein_sel == "Não": m &= ~df["Treineiro"]
    return df[m]


fce = filtrar(df_ce_full)

st.sidebar.divider()
st.sidebar.caption(f"Ceará: {len(fce):,} registros")

if len(fce) == 0:
    st.warning("⚠️ Nenhum dado disponível para Ceará com os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.title("📊 ENEM 2023 · Ceará (Teste)")
st.divider()


# ══════════════════════════════════════════════════════════════════
# CONTEÚDO PRINCIPAL
# ══════════════════════════════════════════════════════════════════

st.subheader("Distribuição da Média Geral")
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=fce["Média Geral"], nbinsx=80, name="Ceará",
    opacity=0.65, histnorm="probability density", marker_color=CE_COLOR,
))
fig.add_vline(x=fce["Média Geral"].median(), line_dash="dash", line_color=CE_COLOR,
              annotation_text=f"Mediana: {fce['Média Geral'].median():.1f}")
fig.update_layout(barmode="overlay", xaxis_title="Média Geral", yaxis_title="Densidade")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Estatísticas Descritivas")
qce = desc(fce["Média Geral"])
rows = []
for k, lbl in [("N","N"),("Média","Média"),("DP","DP"),("Min","Mín"),
                ("Q1","Q1"),("Med","Mediana"),("Q3","Q3"),("Max","Máx"),("IQR","IQR")]:
    rows.append({"Métrica": lbl,
                 "Ceará": f"{int(qce[k]):,}" if k=="N" else f"{qce[k]:.2f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("Desempenho por Disciplina")
def disc_stats(df):
    rows = []
    for col, short in zip(NOTAS, NOTAS_SHORT):
        d = desc(df[col])
        rows.append({"Disciplina": short, "Média": round(d["Média"],2),
                     "Mediana": round(d["Med"],2), "DP": round(d["DP"],2),
                     "Q1": round(d["Q1"],2), "Q3": round(d["Q3"],2),
                     "IQR": round(d["IQR"],2)})
    return pd.DataFrame(rows)

st.dataframe(disc_stats(fce), hide_index=True, use_container_width=True)

mce = [fce[c].mean() for c in NOTAS]
cats = NOTAS_SHORT + [NOTAS_SHORT[0]]
fig_rad = go.Figure()
fig_rad.add_trace(go.Scatterpolar(r=mce+[mce[0]], theta=cats, fill="toself",
                  name="Ceará", line_color=CE_COLOR))
fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True, range=[400, 630])))
legend_top(fig_rad)
st.plotly_chart(fig_rad, use_container_width=True)

st.subheader("Renda Familiar vs Nota")
g = group_stats(fce, "Renda Mensal Familiar")
g = reindex_col(g, "Renda Mensal Familiar", RENDA_ORDER)
fig = px.bar(g, x="Média", y="Renda Mensal Familiar", orientation="h",
             color="Média", color_continuous_scale=["lightgray", CE_COLOR],
             text="Média", title="Ceará")
fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig.update_coloraxes(showscale=False)
fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Acesso Digital vs Nota")
c1, c2 = st.columns(2)
with c1:
    g = group_stats(fce, "Acesso à Internet na Residência")
    fig = px.bar(g, x="Acesso à Internet na Residência", y="Média",
                 color="Média", color_continuous_scale=["lightgray", CE_COLOR],
                 text="Média", title="Internet na Residência")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    g = group_stats(fce, "Computador na Residência")
    fig = px.bar(g, x="Computador na Residência", y="Média",
                 color="Média", color_continuous_scale=["lightgray", CE_COLOR],
                 text="Média", title="Computador na Residência")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Cor/Raça vs Nota")
g = group_stats(fce, "Cor/Raça")
fig = px.bar(g, x="Cor/Raça", y="Média",
             color="Média", color_continuous_scale=["lightgray", CE_COLOR],
             text="Média")
fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig.update_coloraxes(showscale=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sexo vs Nota")
g = group_stats(fce, "Sexo")
fig = px.bar(g, x="Sexo", y="Média",
             color="Média", color_continuous_scale=["lightgray", CE_COLOR],
             text="Média")
fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig.update_coloraxes(showscale=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Tipo de Escola vs Nota")
g = group_stats(fce, "Tipo de Escola do Ensino Médio")
fig = px.bar(g, x="Tipo de Escola do Ensino Médio", y="Média",
             color="Média", color_continuous_scale=["lightgray", CE_COLOR],
             text="Média")
fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig.update_coloraxes(showscale=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("ENEM 2023 · Ceará (Versão Teste)")
