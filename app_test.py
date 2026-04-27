import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ENEM 2023 · CE vs BR",
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

COLS_LOAD = ["Sigla da UF da Escola"] + PRESENCA + NOTAS

CE_COLOR  = "#4C9BE8"
BR_COLOR  = "#E8754C"
COLORWAY  = [CE_COLOR, BR_COLOR, "#4CE8A0", "#E8C44C", "#A04CE8", "#E84CA0"]


# ══════════════════════════════════════════════════════════════════
# CARREGAMENTO OTIMIZADO
# ══════════════════════════════════════════════════════════════════@st.cache_data(show_spinner="Carregando dados…")
def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLS_LOAD)
    
    # Filtrar por presença antes de converter tipos
    df = df[(df[PRESENCA] == "Presente").all(axis=1)].drop(columns=PRESENCA).copy()
    df.dropna(subset=NOTAS, inplace=True)
    
    # Agora sim, converter notas para int16 (sem NaN)
    for col in NOTAS:
        df[col] = df[col].astype("int16")
    
    df["Média Geral"] = df[NOTAS].astype("float32").mean(axis=1).round(2)
    
    return df


@st.cache_data(show_spinner=False)
def get_data():
    return (
        load("https://drive.google.com/uc?export=download&id=1vSH8dq5_Hm0hB-8r9fs9mtdL9sbqWrEF"),
        load("https://drive.google.com/uc?export=download&id=1uBLgOZHo9AjZxWW8dWnH4bVTGjtafNdk"),
    )


df_ce_full, df_br_full = get_data()


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


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Filtros")

# Apenas filtro básico por região
st.sidebar.info("ℹ️ Dados carregados: apenas notas dos datasets")

fce = df_ce_full.copy()
fbr = df_br_full.copy()

st.sidebar.divider()
st.sidebar.caption(f"Ceará: {len(fce):,} registros")
st.sidebar.caption(f"Resto do Brasil: {len(fbr):,} registros")


# ══════════════════════════════════════════════════════════════════
# HEADER + MÉTRICAS
# ══════════════════════════════════════════════════════════════════
st.title("📊 ENEM 2023 · Ceará vs Resto do Brasil")
st.divider()

with st.expander("📌 Premissas e metodologia", expanded=False):
    st.markdown("""
**Fonte e período**
Microdados oficiais do ENEM 2023 (INEP).

**Critérios de inclusão**
- Presença obrigatória em todas as quatro provas objetivas (Ciências da Natureza, Ciências Humanas, Linguagens e Códigos, Matemática).
- Disponibilidade de todas as cinco notas, incluindo Redação.
- Exclusão de participantes classificados como treineiros (por padrão; alterável via filtro).
- Exclusão de registros com valores faltantes nas variáveis de análise.

**Tratamento de dados**
- Verificação de duplicatas: nenhuma instância detectada.
- Análise de outliers (método IQR): nenhuma exclusão realizada; todos os valores dentro do intervalo plausível (0–1000).

**Cálculo de métricas**
Média Geral: média aritmética simples das cinco notas do participante.

**Segmentação geográfica**
- **Ceará**: escola de registro em UF = CE.
- **Resto do Brasil**: demais estados (excluindo Ceará).

**Variáveis socioeconômicas**
Autodeclaradas no questionário do ENEM. Ordenação de faixas segue sequência oficial, não alfabética.

**Filtros**
Aplicados simultaneamente e de forma independente aos conjuntos de Ceará e Resto do Brasil.
""")

st.divider()


# ══════════════════════════════════════════════════════════════════
# ABAS
# ══════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Distribuição Geral",
    "📚 Por Disciplina",
    "� Correlações",
])


# ─────────────────────────────────────
# TAB 0 — DISTRIBUIÇÃO GERAL
# ─────────────────────────────────────
with tabs[0]:
    st.subheader("Distribuição da Média Geral")

    fig = go.Figure()
    for df_, lbl, cor in [(fce, "Ceará", CE_COLOR), (fbr, "Resto do Brasil", BR_COLOR)]:
        fig.add_trace(go.Histogram(
            x=df_["Média Geral"], nbinsx=80, name=lbl,
            opacity=0.65, histnorm="probability density", marker_color=cor,
        ))
        fig.add_vline(x=df_["Média Geral"].median(), line_dash="dash", line_color=cor,
                      annotation_text=f"Med {lbl}: {df_['Média Geral'].median():.1f}")
    fig.update_layout(barmode="overlay", xaxis_title="Média Geral", yaxis_title="Densidade")
    legend_top(fig)
    st.plotly_chart(fig, use_container_width=True)

    df_box = pd.concat([fce[["Média Geral"]].assign(Região="Ceará"),
                        fbr[["Média Geral"]].assign(Região="Resto do Brasil")])
    fig2 = px.box(df_box, x="Região", y="Média Geral", color="Região", points=False,
                  color_discrete_map={"Ceará": CE_COLOR, "Resto do Brasil": BR_COLOR})
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Estatísticas Descritivas")
    qce = desc(fce["Média Geral"])
    qbr = desc(fbr["Média Geral"])
    rows = []
    for k, lbl in [("N","N"),("Média","Média"),("DP","DP"),("Min","Mín"),
                    ("Q1","Q1"),("Med","Mediana"),("Q3","Q3"),("Max","Máx"),("IQR","IQR")]:
        rows.append({"Métrica": lbl,
                     "Ceará":          f"{int(qce[k]):,}" if k=="N" else f"{qce[k]:.2f}",
                     "Resto do Brasil": f"{int(qbr[k]):,}" if k=="N" else f"{qbr[k]:.2f}",
                     "Δ (BR−CE)":      "" if k=="N" else f"{qbr[k]-qce[k]:+.2f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ─────────────────────────────────────
# TAB 1 — POR DISCIPLINA
# ─────────────────────────────────────
with tabs[1]:
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

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Ceará")
        st.dataframe(disc_stats(fce), hide_index=True, use_container_width=True)
    with c2:
        st.caption("Resto do Brasil")
        st.dataframe(disc_stats(fbr), hide_index=True, use_container_width=True)

    mce = [fce[c].mean() for c in NOTAS]
    mbr = [fbr[c].mean() for c in NOTAS]
    cats = NOTAS_SHORT + [NOTAS_SHORT[0]]
    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatterpolar(r=mce+[mce[0]], theta=cats, fill="toself",
                      name="Ceará", line_color=CE_COLOR))
    fig_rad.add_trace(go.Scatterpolar(r=mbr+[mbr[0]], theta=cats, fill="toself",
                      name="Resto do Brasil", line_color=BR_COLOR))
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[400, 630])),
    )
    legend_top(fig_rad)
    st.plotly_chart(fig_rad, use_container_width=True)

    df_bar = pd.DataFrame({
        "Disciplina": NOTAS_SHORT * 2,
        "Média": mce + mbr,
        "Região": ["Ceará"] * 5 + ["Resto do Brasil"] * 5,
    })
    fig_bar = px.bar(df_bar, x="Disciplina", y="Média", color="Região", barmode="group",
                     color_discrete_map={"Ceará": CE_COLOR, "Resto do Brasil": BR_COLOR})
    legend_top(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────
# TAB 2 — CORRELAÇÕES
# ─────────────────────────────────────
with tabs[2]:
    st.subheader("Matriz de Correlação — Notas")

    def corr_fig(df, lbl, cor):
        corr = df[NOTAS + ["Média Geral"]].corr().round(3)
        labels = NOTAS_SHORT + ["Média Geral"]
        return px.imshow(corr, x=labels, y=labels, text_auto=True,
                         zmin=0, zmax=1,
                         color_continuous_scale=["white", cor],
                         title=lbl)

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(corr_fig(fce, "Ceará", CE_COLOR), use_container_width=True)
    with c2: st.plotly_chart(corr_fig(fbr, "Resto do Brasil", BR_COLOR), use_container_width=True)


st.divider()
st.caption("ENEM 2023 · Ceará × Resto do Brasil · apenas notas carregadas")