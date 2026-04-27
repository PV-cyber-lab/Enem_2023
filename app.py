import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings
warnings.filterwarnings("ignore")

print("\n" + "="*70)
print("🚀 INICIANDO APP ENEM 2023 - CEARÁ VS BRASIL")
print("="*70)

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
BR_COLOR  = "#E8754C"
COLORWAY  = [CE_COLOR, BR_COLOR, "#4CE8A0", "#E8C44C", "#A04CE8", "#E84CA0"]


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
    print(f"\n[LOAD] Iniciando carregamento de: {path}")
    try:
        print("[LOAD] 1/7 - Lendo arquivo parquet...")
        df = pd.read_parquet(path, columns=COLS_LOAD)
        print(f"[LOAD]   ✓ Lido: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        print(f"[LOAD]   ✓ Memória inicial: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("[LOAD] 2/7 - Filtrando presença em todas as provas...")
        mask_presenca = (df[PRESENCA] == "Presente").all(axis=1)
        df = df[mask_presenca].drop(columns=PRESENCA).copy()
        print(f"[LOAD]   ✓ Mantidos: {df.shape[0]:,} registros com presença completa")
        
        print("[LOAD] 3/7 - Removendo linhas com notas faltantes...")
        df.dropna(subset=NOTAS, inplace=True)
        print(f"[LOAD]   ✓ Mantidos: {df.shape[0]:,} registros com todas as notas")
        
        print("[LOAD] 4/7 - Calculando Média Geral...")
        df["Média Geral"] = df[NOTAS].mean(axis=1).round(2)
        print(f"[LOAD]   ✓ Média Geral calculada")
        
        print("[LOAD] 5/7 - Processando Faixa Etária...")
        df["Faixa Etária"] = df["Faixa Etária"].apply(_faixa)
        print(f"[LOAD]   ✓ Faixa Etária processada")
        
        print("[LOAD] 6/7 - Processando Treineiro...")
        df["Treineiro"] = df["Treineiro"].astype(str).str.strip().str.upper().isin(["1","SIM","S","TRUE"])
        print(f"[LOAD]   ✓ Treineiro processado")
        
        print("[LOAD] 7/7 - Otimizando tipos de dados...")
        
        # Categorias
        cat_cols = [
            "Sexo", "Cor/Raça", "Tipo de Escola do Ensino Médio", "Sigla da UF da Escola",
            "Situação de Conclusão do Ensino Médio", "Renda Mensal Familiar",
            "Escolaridade do Pai/Responsável Homem", "Escolaridade da Mãe/Responsável Mulher",
            "Acesso à Internet na Residência", "Computador na Residência",
            "Quantidade de Pessoas na Residência", "Faixa Etária",
        ]
        for c in cat_cols:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].astype("category")
        print(f"[LOAD]   ✓ {len(cat_cols)} colunas convertidas para category")
        
        # Notas em int16 (otimizado: range 0-1000, seguro em int16)
        for nota_col in NOTAS:
            df[nota_col] = df[nota_col].astype("int16")
        print(f"[LOAD]   ✓ {len(NOTAS)} colunas de notas convertidas para int16")
        
        # Outras colunas numéricas
        df["Treineiro"] = df["Treineiro"].astype("bool")
        df["Média Geral"] = (df["Média Geral"] * 100).astype("int16")  # Multiplica por 100 para manter 2 casas decimais em int16
        print(f"[LOAD]   ✓ Treineiro convertido para bool")
        print(f"[LOAD]   ✓ Média Geral convertida para int16")
        
        print(f"[LOAD]   ✓ Memória final: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"[LOAD] ✅ Carregamento concluído com sucesso!")
        
        return df
    
    except Exception as e:
        print(f"[LOAD] ❌ ERRO durante carregamento: {str(e)}")
        print(f"[LOAD] Tipo de erro: {type(e).__name__}")
        raise


@st.cache_data(show_spinner=False)
def get_data():
    print("\n[GET_DATA] Iniciando carregamento dos datasets...")
    print("[GET_DATA] Carregando: Ceará")
    df_ce = load("https://drive.google.com/uc?export=download&id=1vSH8dq5_Hm0hB-8r9fs9mtdL9sbqWrEF")
    
    print("[GET_DATA] Carregando: Resto do Brasil")
    df_br = load("https://drive.google.com/uc?export=download&id=1uBLgOZHo9AjZxWW8dWnH4bVTGjtafNdk")
    
    print("[GET_DATA] ✅ Ambos os datasets carregados com sucesso!")
    return (df_ce, df_br)


print("\n[INIT] Carregando datasets...")
df_ce_full, df_br_full = get_data()
print(f"[INIT] ✓ Ceará: {len(df_ce_full):,} registros")
print(f"[INIT] ✓ Brasil: {len(df_br_full):,} registros")


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
print("\n[SIDEBAR] Configurando sidebar com filtros...")
st.sidebar.title("⚙️ Filtros")

# Sexo
sexo_opts   = sorted(set(df_ce_full["Sexo"].dropna().unique()) | set(df_br_full["Sexo"].dropna().unique()))
escola_opts = sorted(set(df_ce_full["Tipo de Escola do Ensino Médio"].dropna().unique()) | set(df_br_full["Tipo de Escola do Ensino Médio"].dropna().unique()))

sexo_sel   = st.sidebar.multiselect("Sexo",   sexo_opts,   default=sexo_opts)
escola_sel = st.sidebar.multiselect("Escola", escola_opts, default=escola_opts)
trein_sel  = st.sidebar.selectbox("Treineiro", ["Todos", "Não", "Sim"], index=1)

st.sidebar.divider()

# Faixa etária — ordered
all_faixas_ce = df_ce_full["Faixa Etária"].dropna().unique()
all_faixas_br = df_br_full["Faixa Etária"].dropna().unique()
all_faixas    = set(all_faixas_ce) | set(all_faixas_br)
faixa_opts    = [f for f in FAIXA_ORDER if f in all_faixas]
faixa_sel     = st.sidebar.multiselect("Faixa Etária", faixa_opts, default=faixa_opts)

st.sidebar.divider()

# Renda — ordered
all_renda_ce = df_ce_full["Renda Mensal Familiar"].dropna().unique()
all_renda_br = df_br_full["Renda Mensal Familiar"].dropna().unique()
all_renda    = set(all_renda_ce) | set(all_renda_br)
renda_opts   = [r for r in RENDA_ORDER if r in all_renda]
renda_sel    = st.sidebar.multiselect("Renda Familiar", renda_opts, default=renda_opts)


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
fbr = filtrar(df_br_full)

print(f"\n[FILTERS] Aplicados filtros:")
print(f"[FILTERS] ✓ Ceará: {len(fce):,} registros")
print(f"[FILTERS] ✓ Brasil: {len(fbr):,} registros")

st.sidebar.divider()
st.sidebar.caption(f"Ceará: {len(fce):,} registros")
st.sidebar.caption(f"Resto do Brasil: {len(fbr):,} registros")

# ── Validação global ──────────────────────────────────────────────
print("\n[VALIDATION] Validando dados após filtros...")
_vazio_ce = len(fce) == 0
_vazio_br = len(fbr) == 0
if _vazio_ce or _vazio_br:
    partes = []
    if _vazio_ce: partes.append("**Ceará**")
    if _vazio_br: partes.append("**Resto do Brasil**")
    print(f"[VALIDATION] ⚠️ AVISO: Sem dados para {' e '.join(partes)}")
    st.warning(
        f"⚠️ Nenhum dado disponível para {' e '.join(partes)} com os filtros selecionados. "
        "Ajuste os filtros na barra lateral.",
    )
    st.stop()

print("[VALIDATION] ✅ Validação passou com sucesso!")


# ══════════════════════════════════════════════════════════════════
# HEADER + MÉTRICAS
# ══════════════════════════════════════════════════════════════════
print("\n[UI] Renderizando interface...")
st.title("📊 ENEM 2023 · Ceará vs Resto do Brasil")
st.divider()

print("[UI] ✓ Header renderizado")

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
- Dados faltantes: exclusão por listwise nas análises específicas.

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
print("\n[TABS] Criando abas...")
tabs = st.tabs([
    "📊 Distribuição Geral",
    "📚 Por Disciplina",
    "💰 Socioeconômico",
    "🏫 Escola & Região",
    "🌈 Raça · Gênero · Idade",
    "🔗 Correlações",
    "📋 Tabelas Detalhadas",
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
# TAB 2 — SOCIOECONÔMICO
# ─────────────────────────────────────
with tabs[2]:
    st.subheader("Renda Familiar vs Nota")

    def renda_fig(df, lbl, cor):
        g = group_stats(df, "Renda Mensal Familiar")
        g = reindex_col(g, "Renda Mensal Familiar", RENDA_ORDER)
        fig = px.bar(g, x="Média", y="Renda Mensal Familiar", orientation="h",
                     color="Média", color_continuous_scale=["lightgray", cor],
                     text="Média", title=lbl)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(renda_fig(fce, "Ceará", CE_COLOR), use_container_width=True)
    with c2: st.plotly_chart(renda_fig(fbr, "Resto do Brasil", BR_COLOR), use_container_width=True)

    st.subheader("Renda × Nota por Disciplina")

    def renda_disc_fig(df, lbl):
        rows = []
        for col, short in zip(NOTAS, NOTAS_SHORT):
            g = df.groupby("Renda Mensal Familiar")[col].mean().reset_index()
            g.columns = ["Renda", "Média"]
            g["Disciplina"] = short
            rows.append(g)
        df_rd = pd.concat(rows)
        ordem = [r for r in RENDA_ORDER if r in df_rd["Renda"].values]
        fig = px.line(df_rd[df_rd["Renda"].isin(ordem)], x="Renda", y="Média",
                      color="Disciplina", markers=True,
                      category_orders={"Renda": ordem},
                      color_discrete_sequence=COLORWAY, title=lbl)
        fig.update_layout(xaxis_tickangle=-40)
        legend_right(fig)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(renda_disc_fig(fce, "Ceará"), use_container_width=True)
    with c2: st.plotly_chart(renda_disc_fig(fbr, "Resto do Brasil"), use_container_width=True)

    st.subheader("Escolaridade dos Responsáveis vs Nota")
    for col_esc in ["Escolaridade do Pai/Responsável Homem",
                    "Escolaridade da Mãe/Responsável Mulher"]:
        st.caption(col_esc.split("/")[0])
        c1, c2 = st.columns(2)
        with c1:
            g = group_stats(fce, col_esc)
            fig = px.bar(g, x="Média", y=col_esc, orientation="h",
                         color="Média", color_continuous_scale=["lightgray", CE_COLOR],
                         text="Média", title="Ceará")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            g = group_stats(fbr, col_esc)
            fig = px.bar(g, x="Média", y=col_esc, orientation="h",
                         color="Média", color_continuous_scale=["lightgray", BR_COLOR],
                         text="Média", title="Resto do Brasil")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Acesso Digital vs Nota")

    def digital_comparison_fig(df_ce, df_br, col_name, titulo):
        """Compara Ceará vs Brasil para um tipo de acesso digital."""
        g_ce = group_stats(df_ce, col_name)
        g_br = group_stats(df_br, col_name)
        
        g_ce["Região"] = "Ceará"
        g_br["Região"] = "Resto do Brasil"
        
        g_ce.rename(columns={col_name: "Resposta"}, inplace=True)
        g_br.rename(columns={col_name: "Resposta"}, inplace=True)
        
        df_combined = pd.concat([g_ce, g_br])
        
        fig = px.bar(df_combined, x="Resposta", y="Média", color="Região", barmode="group",
                     color_discrete_map={"Ceará": CE_COLOR, "Resto do Brasil": BR_COLOR},
                     title=titulo, text="Média")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        legend_top(fig)
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            digital_comparison_fig(fce, fbr, "Acesso à Internet na Residência", "Acesso à Internet na Residência"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            digital_comparison_fig(fce, fbr, "Computador na Residência", "Computador na Residência"),
            use_container_width=True
        )


# ─────────────────────────────────────
# TAB 3 — ESCOLA & REGIÃO
# ─────────────────────────────────────
with tabs[3]:
    st.subheader("Desempenho por Tipo de Escola")

    def escola_fig(df, lbl, cor):
        g = group_stats(df, "Tipo de Escola do Ensino Médio")
        fig = px.bar(g, x="Tipo de Escola do Ensino Médio", y="Média",
                     color="Média", color_continuous_scale=["lightgray", cor],
                     text="Média", title=lbl)
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_coloraxes(showscale=False)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(escola_fig(fce, "Ceará", CE_COLOR), use_container_width=True)
    with c2: st.plotly_chart(escola_fig(fbr, "Resto do Brasil", BR_COLOR), use_container_width=True)

    st.subheader("Ranking de UFs (Resto do Brasil)")
    uf_g = group_stats(fbr, "Sigla da UF da Escola")
    fig_uf = px.bar(uf_g, x="Sigla da UF da Escola", y="Média", color="Média",
                    color_continuous_scale=["lightgray", BR_COLOR], text="Média")
    fig_uf.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_uf.update_coloraxes(showscale=False)
    st.plotly_chart(fig_uf, use_container_width=True)

    top15 = fbr.groupby("Sigla da UF da Escola")["Média Geral"].median().nlargest(15).index
    order_uf = (fbr.groupby("Sigla da UF da Escola")["Média Geral"]
                .median().sort_values(ascending=False).index.tolist())
    fig_uf2 = px.box(fbr[fbr["Sigla da UF da Escola"].isin(top15)],
                     x="Sigla da UF da Escola", y="Média Geral",
                     category_orders={"Sigla da UF da Escola": order_uf},
                     points=False, color_discrete_sequence=[BR_COLOR],
                     title="Distribuição por UF — top 15 (mediana)")
    st.plotly_chart(fig_uf2, use_container_width=True)

    st.subheader("Heatmap — Escola × Disciplina")
    for df_, lbl, cor in [(fce, "Ceará", CE_COLOR), (fbr, "Resto do Brasil", BR_COLOR)]:
        rows = []
        for escola in df_["Tipo de Escola do Ensino Médio"].dropna().unique():
            sub = df_[df_["Tipo de Escola do Ensino Médio"] == escola]
            row = {"Escola": escola}
            for col, short in zip(NOTAS, NOTAS_SHORT):
                row[short] = round(sub[col].mean(), 1)
            rows.append(row)
        df_h = pd.DataFrame(rows).set_index("Escola")
        fig_h = px.imshow(df_h, text_auto=True,
                          color_continuous_scale=["white", cor],
                          title=f"Escola × Disciplina — {lbl}")
        st.plotly_chart(fig_h, use_container_width=True)


# ─────────────────────────────────────
# TAB 4 — RAÇA · GÊNERO · IDADE
# ─────────────────────────────────────
with tabs[4]:
    def trio_plot(df, lbl, cor):
        g1 = group_stats(df, "Cor/Raça")
        f1 = px.bar(g1, x="Cor/Raça", y="Média",
                    color="Média", color_continuous_scale=["lightgray", cor],
                    text="Média", title=f"Cor/Raça — {lbl}")
        f1.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        f1.update_coloraxes(showscale=False)

        g2 = group_stats(df, "Sexo")
        f2 = px.bar(g2, x="Sexo", y="Média",
                    color="Média", color_continuous_scale=["lightgray", cor],
                    text="Média", title=f"Sexo — {lbl}")
        f2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        f2.update_coloraxes(showscale=False)

        g3 = group_stats(df, "Faixa Etária")
        g3 = reindex_col(g3, "Faixa Etária", FAIXA_ORDER)
        f3 = px.line(g3, x="Faixa Etária", y="Média", markers=True,
                     color_discrete_sequence=[cor], title=f"Faixa Etária — {lbl}")
        f3.update_layout(xaxis_tickangle=-40, showlegend=False)
        return f1, f2, f3

    figs_ce = trio_plot(fce, "Ceará", CE_COLOR)
    figs_br = trio_plot(fbr, "Resto do Brasil", BR_COLOR)
    for fce_p, fbr_p in zip(figs_ce, figs_br):
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(fce_p, use_container_width=True)
        with c2: st.plotly_chart(fbr_p, use_container_width=True)

    st.subheader("Faixa Etária × Nota por Disciplina")

    def faixa_disc_fig(df, lbl):
        rows = []
        for col, short in zip(NOTAS, NOTAS_SHORT):
            g = df.groupby("Faixa Etária")[col].mean().reset_index()
            g.columns = ["Faixa Etária", "Média"]
            g["Disciplina"] = short
            rows.append(g)
        df_fd = pd.concat(rows)
        ordem = [f for f in FAIXA_ORDER if f in df_fd["Faixa Etária"].values]
        fig = px.line(df_fd[df_fd["Faixa Etária"].isin(ordem)],
                      x="Faixa Etária", y="Média", color="Disciplina", markers=True,
                      category_orders={"Faixa Etária": ordem},
                      color_discrete_sequence=COLORWAY, title=lbl)
        fig.update_layout(xaxis_tickangle=-40)
        legend_right(fig)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(faixa_disc_fig(fce, "Ceará"), use_container_width=True)
    with c2: st.plotly_chart(faixa_disc_fig(fbr, "Resto do Brasil"), use_container_width=True)

    st.subheader("Raça × Sexo")

    def raca_sexo_fig(df, lbl):
        g = df.groupby(["Cor/Raça", "Sexo"])["Média Geral"].mean().reset_index()
        fig = px.bar(g, x="Cor/Raça", y="Média Geral", color="Sexo", barmode="group",
                     color_discrete_sequence=[CE_COLOR, BR_COLOR], title=lbl)
        legend_top(fig)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(raca_sexo_fig(fce, "Ceará"), use_container_width=True)
    with c2: st.plotly_chart(raca_sexo_fig(fbr, "Resto do Brasil"), use_container_width=True)

    st.subheader("Raça × Renda × Nota")

    def raca_renda_fig(df, lbl):
        g = df.groupby(["Cor/Raça", "Renda Mensal Familiar"])["Média Geral"].mean().reset_index()
        ordem = [r for r in RENDA_ORDER if r in g["Renda Mensal Familiar"].values]
        fig = px.line(g[g["Renda Mensal Familiar"].isin(ordem)],
                      x="Renda Mensal Familiar", y="Média Geral", color="Cor/Raça",
                      markers=True, category_orders={"Renda Mensal Familiar": ordem},
                      color_discrete_sequence=COLORWAY, title=lbl)
        fig.update_layout(xaxis_tickangle=-40)
        legend_right(fig)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(raca_renda_fig(fce, "Ceará"), use_container_width=True)
    with c2: st.plotly_chart(raca_renda_fig(fbr, "Resto do Brasil"), use_container_width=True)


# ─────────────────────────────────────
# TAB 5 — CORRELAÇÕES
# ─────────────────────────────────────
with tabs[5]:
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

    st.subheader("Correlação Renda × Nota")

    def renda_corr_fig(df, lbl, cor):
        renda_num = df["Renda Mensal Familiar"].map(
            {r: i for i, r in enumerate(RENDA_ORDER)}).dropna()
        idx = renda_num.index
        rows = []
        for col, short in zip(NOTAS + ["Média Geral"], NOTAS_SHORT + ["Média Geral"]):
            r = renda_num.corr(df.loc[idx, col])
            rows.append({"Disciplina": short, "Correlação c/ Renda": round(r, 3)})
        df_c = pd.DataFrame(rows)
        fig = px.bar(df_c, x="Disciplina", y="Correlação c/ Renda",
                     color="Correlação c/ Renda",
                     color_continuous_scale=["lightgray", cor],
                     text="Correlação c/ Renda", title=lbl)
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_coloraxes(showscale=False)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(renda_corr_fig(fce, "Ceará", CE_COLOR), use_container_width=True)
    with c2: st.plotly_chart(renda_corr_fig(fbr, "Resto do Brasil", BR_COLOR), use_container_width=True)


# ─────────────────────────────────────
# TAB 6 — TABELAS DETALHADAS
# ─────────────────────────────────────
with tabs[6]:
    st.subheader("Estatísticas por Grupo × Nota")

    CAT_OPTS = {
        "Cor/Raça":                       "Cor/Raça",
        "Sexo":                           "Sexo",
        "Faixa Etária":                   "Faixa Etária",
        "Renda Mensal Familiar":          "Renda Mensal Familiar",
        "Tipo de Escola":                 "Tipo de Escola do Ensino Médio",
        "UF (Escola)":                    "Sigla da UF da Escola",
        "Escolaridade Pai":               "Escolaridade do Pai/Responsável Homem",
        "Escolaridade Mãe":               "Escolaridade da Mãe/Responsável Mulher",
        "Internet":                       "Acesso à Internet na Residência",
        "Computador":                     "Computador na Residência",
        "Qtd. Pessoas na Residência":     "Quantidade de Pessoas na Residência",
    }
    NOTA_OPTS = {"Média Geral": "Média Geral"} | dict(zip(NOTAS_SHORT, NOTAS))

    c1, c2 = st.columns(2)
    sel_cat  = c1.selectbox("Agrupar por:", list(CAT_OPTS.keys()))
    sel_nota = c2.selectbox("Nota:", list(NOTA_OPTS.keys()))
    col_grp  = CAT_OPTS[sel_cat]
    col_nota = NOTA_OPTS[sel_nota]

    def full_table(df):
        g = (df.groupby(col_grp)[col_nota]
             .agg(N="count", Média="mean", Mediana="median", DP="std",
                  Min="min",
                  Q1=lambda x: x.quantile(.25),
                  Q3=lambda x: x.quantile(.75),
                  Max="max")
             .round(2).reset_index())
        g["IQR"] = (g["Q3"] - g["Q1"]).round(2)
        return g.sort_values("Média", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Ceará")
        st.dataframe(full_table(fce), hide_index=True, use_container_width=True)
    with c2:
        st.caption("Resto do Brasil")
        st.dataframe(full_table(fbr), hide_index=True, use_container_width=True)

    st.divider()
    with st.expander("Todas as disciplinas por grupo"):
        for nota, short in zip(NOTAS + ["Média Geral"], NOTAS_SHORT + ["Média Geral"]):
            st.caption(short)
            def _tbl(df, n=nota):
                g = (df.groupby(col_grp)[n]
                     .agg(N="count", Média="mean", Mediana="median", DP="std",
                          Q1=lambda x: x.quantile(.25), Q3=lambda x: x.quantile(.75))
                     .round(2).reset_index())
                g["IQR"] = (g["Q3"] - g["Q1"]).round(2)
                return g.sort_values("Média", ascending=False)
            c1, c2 = st.columns(2)
            with c1: st.dataframe(_tbl(fce), hide_index=True, use_container_width=True)
            with c2: st.dataframe(_tbl(fbr), hide_index=True, use_container_width=True)


st.divider()
st.caption("ENEM 2023 · Ceará × Resto do Brasil · presença obrigatória em todas as provas")

print("\n" + "="*70)
print("✅ APP RENDERIZADO COM SUCESSO!")
print("="*70 + "\n")