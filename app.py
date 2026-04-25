import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DA PÁGINA
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="📊 Quantis - ENEM 2023 Ceará",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
        font-size: 2.5em;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CARREGAMENTO E PREPARAÇÃO DE DADOS
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dados_ceara.csv')
    
    # Criar média geral
    colunas_notas = [
        'Nota Ciências da Natureza', 
        'Nota Ciências Humanas',
        'Nota Linguagens e Códigos', 
        'Nota Matemática',
        'Nota Redação'
    ]
    df['Media_Geral'] = df[colunas_notas].mean(axis=1)
    
    # Extrair idade
    import re
    def extrair_idade(x):
        if not isinstance(x, str):
            return None
        numeros = re.findall(r'\d+', x)
        return int(numeros[0]) if numeros else None
    
    df['Idade'] = df['Faixa Etária'].apply(extrair_idade)
    
    # Consolidar faixa etária
    def consolidar_faixa(x):
        if not isinstance(x, str):
            return x
        numeros = [int(s) for s in x.split() if s.isdigit()]
        if not numeros:
            return x
        idade = numeros[0]
        if idade < 17:
            return 'Menor de 17 anos'
        elif 26 <= idade <= 30:
            return '26+'
        elif 31 <= idade <= 35:
            return '31+'
        elif idade >= 36:
            return '40+'
        return x
    
    df['Faixa Etária Consolidada'] = df['Faixa Etária'].apply(consolidar_faixa)
    
    return df

df = carregar_dados()

# ═══════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
st.title("📊 Análise de Quantis - Média Geral ENEM Ceara 2023")
st.markdown("**Ceará | Brasil**")
st.markdown("---")

# Métricas principais
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("👥 Participantes", f"{len(df):,}")
with col2:
    st.metric("📊 Mínimo", f"{df['Media_Geral'].min():.2f}")
with col3:
    st.metric("📈 Média", f"{df['Media_Geral'].mean():.2f}")
with col4:
    st.metric("🎯 Mediana", f"{df['Media_Geral'].median():.2f}")
with col5:
    st.metric("📈 Máximo", f"{df['Media_Geral'].max():.2f}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - FILTROS
# ═══════════════════════════════════════════════════════════════════
st.sidebar.title("🎛️ Filtros")

faixa_selecionada = st.sidebar.multiselect(
    "Idade:",
    options=df['Faixa Etária Consolidada'].unique(),
    default=df['Faixa Etária Consolidada'].unique()
)

cor_selecionada = st.sidebar.multiselect(
    "Cor/Raça:",
    options=df['Cor/Raça'].unique(),
    default=df['Cor/Raça'].unique()
)

df_filtrado = df[
    (df['Faixa Etária Consolidada'].isin(faixa_selecionada)) &
    (df['Cor/Raça'].isin(cor_selecionada))
]

st.sidebar.info(f"📊 Registros: {len(df_filtrado)} / {len(df)}")

# ═══════════════════════════════════════════════════════════════════
# CALCULAR QUANTIS
# ═══════════════════════════════════════════════════════════════════
def calcular_quantis(dados):
    return {
        'Q0': dados.min(),
        'Q1': dados.quantile(0.25),
        'Q2': dados.quantile(0.50),
        'Q3': dados.quantile(0.75),
        'Q4': dados.max(),
        'IQR': dados.quantile(0.75) - dados.quantile(0.25),
        'Media': dados.mean(),
        'DP': dados.std()
    }

quantis = calcular_quantis(df_filtrado['Media_Geral'])

# ═══════════════════════════════════════════════════════════════════
# ABAS PRINCIPAIS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Quantis Básicos",
    "📈 Visualização",
    "📋 Análise Detalhada",
    "ℹ️ O que são Quantis?"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: QUANTIS BÁSICOS
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("Quantis Principais")
    
    # Tabela de quantis
    st.subheader("Tabela de Quantis")
    
    quantis_df = pd.DataFrame({
        'Quantil': ['Mínimo (Q0)', 'Q1 (25%)', 'Q2 (50% - Mediana)', 'Q3 (75%)', 'Máximo (Q4)'],
        'Valor': [
            f"{quantis['Q0']:.2f}",
            f"{quantis['Q1']:.2f}",
            f"{quantis['Q2']:.2f}",
            f"{quantis['Q3']:.2f}",
            f"{quantis['Q4']:.2f}"
        ],
        'Descrição': [
            'Menor nota registrada',
            '25% dos alunos têm notas abaixo deste valor',
            '50% têm acima, 50% têm abaixo (mediana)',
            '75% dos alunos têm notas abaixo deste valor',
            'Maior nota registrada'
        ]
    })
    
    st.dataframe(quantis_df, use_container_width=True, hide_index=True)
    
    # Cards com informações
    st.subheader("KPIs Importantes")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Q1 (1º Quartil)",
            f"{quantis['Q1']:.2f}",
            delta="25º percentil"
        )
    
    with col2:
        st.metric(
            "Q2 (Mediana)",
            f"{quantis['Q2']:.2f}",
            delta="50º percentil"
        )
    
    with col3:
        st.metric(
            "Q3 (3º Quartil)",
            f"{quantis['Q3']:.2f}",
            delta="75º percentil"
        )
    
    with col4:
        st.metric(
            "IQR (Intervalo Interquartil)",
            f"{quantis['IQR']:.2f}",
            delta="Q3 - Q1"
        )
    
    st.markdown("---")
    
    # Interpretação dos quantis
    st.subheader("📌 Interpretação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Distribuição dos Dados
        
        **0% a 25%** (Q0 → Q1)
        - Notas de {quantis['Q0']:.2f} a {quantis['Q1']:.2f}
        - {(25)}% dos alunos estão nesta faixa
        
        **25% a 50%** (Q1 → Q2)
        - Notas de {quantis['Q1']:.2f} a {quantis['Q2']:.2f}
        - {(25)}% dos alunos estão nesta faixa
        """)
    
    with col2:
        st.markdown(f"""
        ### Distribuição dos Dados (cont.)
        
        **50% a 75%** (Q2 → Q3)
        - Notas de {quantis['Q2']:.2f} a {quantis['Q3']:.2f}
        - {(25)}% dos alunos estão nesta faixa
        
        **75% a 100%** (Q3 → Q4)
        - Notas de {quantis['Q3']:.2f} a {quantis['Q4']:.2f}
        - {(25)}% dos alunos estão nesta faixa
        """)

# ═══════════════════════════════════════════════════════════════════
# TAB 2: VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("Visualização dos Quantis")
    
    col1, col2 = st.columns(2)
    
    # BOXPLOT PLOTLY
    with col1:
        st.subheader("📦 Boxplot Interativo")
        
        fig_box = px.box(
            y=df_filtrado['Media_Geral'],
            title='Distribuição de Quantis',
            labels={'y': 'Média Geral'},
            color_discrete_sequence=['#1f77b4']
        )
        
        fig_box.add_hline(
            y=quantis['Q1'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Q1: {quantis['Q1']:.2f}",
            annotation_position="right"
        )
        
        fig_box.add_hline(
            y=quantis['Q2'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Q2 (Mediana): {quantis['Q2']:.2f}",
            annotation_position="right"
        )
        
        fig_box.add_hline(
            y=quantis['Q3'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Q3: {quantis['Q3']:.2f}",
            annotation_position="right"
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    # HISTOGRAMA COM QUANTIS
    with col2:
        st.subheader("📊 Histograma com Quantis")
        
        fig_hist = plt.figure(figsize=(8, 6))
        plt.hist(df_filtrado['Media_Geral'].dropna(), bins=50, density=True, alpha=0.7, 
                 color='steelblue', edgecolor='black', label='Distribuição')
        
        # Adicionar linhas dos quantis
        plt.axvline(quantis['Q1'], color='red', linestyle='--', linewidth=2, label=f"Q1: {quantis['Q1']:.2f}")
        plt.axvline(quantis['Q2'], color='green', linestyle='--', linewidth=2, label=f"Q2: {quantis['Q2']:.2f}")
        plt.axvline(quantis['Q3'], color='orange', linestyle='--', linewidth=2, label=f"Q3: {quantis['Q3']:.2f}")
        plt.axvline(quantis['Media'], color='purple', linestyle='-', linewidth=2, label=f"Média: {quantis['Media']:.2f}")
        
        plt.title('Histograma com Quantis', fontsize=12, fontweight='bold')
        plt.xlabel('Média Geral', fontsize=11)
        plt.ylabel('Densidade', fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # GRÁFICO Q-Q PLOT
    st.subheader("📊 Q-Q Plot - Teste de Normalidade")
    
    fig_qq, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Q-Q Plot
    stats.probplot(df_filtrado['Media_Geral'].dropna(), dist="norm", plot=axes[0])
    axes[0].set_title('Q-Q Plot - Média Geral ENEM', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histograma com curva normal
    axes[1].hist(df_filtrado['Media_Geral'].dropna(), bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Distribuição Real')
    
    # Sobrepor curva normal teórica
    mu = df_filtrado['Media_Geral'].mean()
    sigma = df_filtrado['Media_Geral'].std()
    x = np.linspace(df_filtrado['Media_Geral'].min(), df_filtrado['Media_Geral'].max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Curva Normal Teórica')
    
    axes[1].set_title('Histograma vs Distribuição Normal Teórica', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Média Geral', fontsize=11)
    axes[1].set_ylabel('Densidade', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_qq, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3: ANÁLISE DETALHADA
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.header("Análise Detalhada dos Quantis")
    
    # Estatísticas descritivas completas
    st.subheader("📊 Estatísticas Descritivas Completas")
    
    stats_df = pd.DataFrame({
        'Métrica': [
            'Contagem',
            'Média',
            'Desvio Padrão',
            'Mínimo',
            'Q1 (25%)',
            'Q2 (50% - Mediana)',
            'Q3 (75%)',
            'Máximo',
            'IQR (Q3 - Q1)',
            'Amplitude',
            'Coeficiente de Variação'
        ],
        'Valor': [
            f"{len(df_filtrado)}",
            f"{quantis['Media']:.2f}",
            f"{quantis['DP']:.2f}",
            f"{quantis['Q0']:.2f}",
            f"{quantis['Q1']:.2f}",
            f"{quantis['Q2']:.2f}",
            f"{quantis['Q3']:.2f}",
            f"{quantis['Q4']:.2f}",
            f"{quantis['IQR']:.2f}",
            f"{quantis['Q4'] - quantis['Q0']:.2f}",
            f"{(quantis['DP'] / quantis['Media'] * 100):.2f}%"
        ]
    })
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Análise por grupos
    st.subheader("🔍 Quantis por Faixa Etária")
    
    # Criar tabela de quantis por faixa etária
    faixas_unicas = df_filtrado['Faixa Etária Consolidada'].unique()
    
    dados_faixa = []
    for faixa in sorted(faixas_unicas):
        dados_faixa_temp = df_filtrado[df_filtrado['Faixa Etária Consolidada'] == faixa]['Media_Geral']
        
        dados_faixa.append({
            'Faixa Etária': faixa,
            'N': len(dados_faixa_temp),
            'Média': f"{dados_faixa_temp.mean():.2f}",
            'Q1': f"{dados_faixa_temp.quantile(0.25):.2f}",
            'Mediana': f"{dados_faixa_temp.quantile(0.50):.2f}",
            'Q3': f"{dados_faixa_temp.quantile(0.75):.2f}",
            'DP': f"{dados_faixa_temp.std():.2f}"
        })
    
    df_quantis_faixa = pd.DataFrame(dados_faixa)
    st.dataframe(df_quantis_faixa, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Análise por cor/raça
    st.subheader("🔍 Quantis por Cor/Raça")
    
    cores_unicas = df_filtrado['Cor/Raça'].unique()
    
    dados_cor = []
    for cor in sorted(cores_unicas):
        dados_cor_temp = df_filtrado[df_filtrado['Cor/Raça'] == cor]['Media_Geral']
        
        dados_cor.append({
            'Cor/Raça': cor,
            'N': len(dados_cor_temp),
            'Média': f"{dados_cor_temp.mean():.2f}",
            'Q1': f"{dados_cor_temp.quantile(0.25):.2f}",
            'Mediana': f"{dados_cor_temp.quantile(0.50):.2f}",
            'Q3': f"{dados_cor_temp.quantile(0.75):.2f}",
            'DP': f"{dados_cor_temp.std():.2f}"
        })
    
    df_quantis_cor = pd.DataFrame(dados_cor)
    st.dataframe(df_quantis_cor, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4: O QUE SÃO QUANTIS?
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.header("O que são Quantis?")
    
    st.markdown("""
    ### 📚 Definição
    
    **Quantis** são valores que dividem um conjunto de dados ordenados em partes iguais. 
    Eles ajudam a entender como os dados estão distribuídos.
    
    ---
    
    ### 📊 Tipos de Quantis com 4 divisões (Quartis)
    
    **Q0 (Mínimo)**: O menor valor do conjunto
    - Neste caso: **{:.2f}** (pior nota)
    
    **Q1 (1º Quartil)**: Deixa 25% dos dados abaixo dele
    - Neste caso: **{:.2f}**
    - Interpretação: 25% dos alunos têm notas **abaixo** deste valor
    
    **Q2 (Mediana)**: Deixa 50% dos dados abaixo dele
    - Neste caso: **{:.2f}**
    - Interpretação: 50% dos alunos têm notas acima e 50% abaixo
    
    **Q3 (3º Quartil)**: Deixa 75% dos dados abaixo dele
    - Neste caso: **{:.2f}**
    - Interpretação: 75% dos alunos têm notas **abaixo** deste valor
    
    **Q4 (Máximo)**: O maior valor do conjunto
    - Neste caso: **{:.2f}** (melhor nota)
    
    ---
    
    ### 📈 IQR (Intervalo Interquartil)
    
    **IQR = Q3 - Q1**
    
    - É a amplitude da "caixa" do boxplot
    - Contém 50% dos dados
    - Neste caso: **{:.2f}**
    - Quanto menor, mais concentrados são os dados no meio
    - Quanto maior, mais dispersos
    
    ---
    
    ### 🎯 Para que servem?
    
    ✅ Entender a distribuição dos dados  
    ✅ Identificar valores atípicos (outliers)  
    ✅ Comparar grupos diferentes  
    ✅ Fazer análises estatísticas  
    ✅ Explicar dados de forma simples  
    
    ---
    
    ### 📊 Exemplo Visual (Boxplot)
    
    ```
    Mínimo     Q1        Mediana      Q3       Máximo
       |        |          |          |          |
       •--------[========|========]--------•
                    50% dos dados
                    (IQR)
    ```
    
    """.format(
        quantis['Q0'],
        quantis['Q1'],
        quantis['Q2'],
        quantis['Q3'],
        quantis['Q4'],
        quantis['IQR']
    ))
    
    st.markdown("---")
    
    # Comparação visual
    st.subheader("🔄 Comparação: Percentis vs Quartis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Percentis (100 divisões)
        - P0 = Mínimo
        - P25 = Q1 (1º Quartil)
        - P50 = Q2/Mediana
        - P75 = Q3 (3º Quartil)
        - P100 = Máximo
        
        Identifica em que percentual estão os dados
        """)
    
    with col2:
        st.markdown("""
        ### Quartis (4 divisões)
        - Q1 = Deixa 25% abaixo
        - Q2 = Deixa 50% abaixo
        - Q3 = Deixa 75% abaixo
        
        Divide em 4 partes iguais
        """)

# FOOTER
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>📊 Dashboard de Quantis - ENEM 2023 | Ceará</p>", unsafe_allow_html=True)
