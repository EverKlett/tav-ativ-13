## pacotes de tratamento de dados, interface, gráfico e mapas
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.express as px
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(layout="wide")
st.title('App - Tópicos Avançados')

## Leitura dos banco de dados em cache
@st.cache_data
def load_database():
    return pd.read_feather('../dados/ss.feather'), \
        pd.read_feather('../dados/knn_pais.feather'), \
        pd.read_feather('../dados/probabilidade_pais.feather'), \
        pd.read_feather('../dados/classificacao_estado.feather'), \
        pd.read_feather('../dados/clusterizacao_pais.feather')


ss, knn_pais, prb_pai, cla_con, clu_pai = load_database()

## Criação das opções com base em tabs
taberp, tabbi, tabstore = st.tabs(['Sistema Interno', 'Gestão', 'E-Commerce'])

with taberp:
    st.header('Dados do Sistema Interno')
    consumidor = st.selectbox(
        'Selecione o consumidor',
        ss['Customer ID'].unique()
    )
    ss_con = ss[ss['Customer ID'] == consumidor]
    cla_con_con = cla_con[cla_con['Customer ID'] == consumidor].reset_index()
    st.dataframe(ss_con[['Customer Name', 'Segment']].drop_duplicates())
    cl1, cl2, cl3, cl4 = st.columns(4)
    cl1.metric('Score', round(cla_con_con['score'][0],4))
    #cl2.metric('Classe', round(cla_con_con['classe'][0],4))
    cl2.metric('Rank', round(cla_con_con['rank'][0],4))
    #cl4.metric('Lucro', round(cla_con_con['lucro'][0],4))
    cl3.metric('Valor Total Comprado', round(ss_con['Sales'].sum(),2))
    cl4.metric('Valor Lucro', round(ss_con['Profit'].sum(),2))
    #cl3.metric('Valor Médio Comprado', round(gs_con['Sales'].mean(),2))
    #cl4.metric('Quantidade Comprada', round(gs_con['Quantity'].sum(),2))
    st.write(ss_con['Country'].values[0])
    st.dataframe(
        prb_pai[prb_pai['Country'] == ss_con['Country'].values[0]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "prob_lucro": st.column_config.ProgressColumn("Prob Lucro", format="%.2f", min_value=0, max_value=1),
            "prob_prejuizo": st.column_config.ProgressColumn("Prob Prejuízo", format="%.2f", min_value=0, max_value=1),
        }
    )
    prob = st.toggle('Similares')
    if prob:
        st.dataframe(
            knn_pais[knn_pais['referencia'] == ss_con['Country'].values[0]].merge(
                prb_pai, left_on='vizinho', right_on='Country', how='left')[
                ['Country','Sales','Quantity','Profit','prob_prejuizo','prob_lucro']
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "prob_lucro": st.column_config.ProgressColumn("Prob Lucro", format="%.2f", min_value=0, max_value=1),
                "prob_prejuizo": st.column_config.ProgressColumn("Prob Prejuízo", format="%.2f", min_value=0, max_value=1),
            }
        )
    clus = st.toggle('Clusters')
    if clus:
        clu_pai_cli = clu_pai[clu_pai['referencia'] == ss_con['Country'].values[0]]
        st.write('Dados do Cluster do País')
        st.dataframe(clu_pai_cli[
            ['cluster', 'clm_entrega', 'clm_lucro', 'clm_vendas', 'clm_qtde', \
                'clf_vendas', 'cls_lucro', 'clr_dias']],
            hide_index=True,
            use_container_width=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Montante de Entrega',
                  clu_pai_cli['m_entrega'].values[0],
                  delta=clu_pai_cli['m_entrega'].values[0] - clu_pai_cli['clm_entrega'].values[0])
        c2.metric('Montante de Lucro',
                  clu_pai_cli['m_lucro'].values[0],
                  delta=clu_pai_cli['m_lucro'].values[0] - clu_pai_cli['clm_lucro'].values[0])
        c3.metric('Montante de Vendas',
                  clu_pai_cli['m_vendas'].values[0],
                  delta=clu_pai_cli['m_vendas'].values[0] - clu_pai_cli['clm_vendas'].values[0])
        c4.metric('Montante de Quantidade',
                  clu_pai_cli['m_qtde'].values[0],
                  delta=clu_pai_cli['m_qtde'].values[0] - clu_pai_cli['clm_qtde'].values[0])
        c1.metric('Periodicidade em Dias', clu_pai_cli['r_dias'].values[0],
                  delta=clu_pai_cli['r_dias'].values[0] - clu_pai_cli['clr_dias'].values[0],
                  delta_color='inverse')
        c2.metric('Frequencia de Vendas', clu_pai_cli['f_vendas'].values[0],
                  delta=clu_pai_cli['f_vendas'].values[0] - clu_pai_cli['clf_vendas'].values[0])
        c3.metric('Frequencia de Lucro', clu_pai_cli['f_lucro'].values[0],
                  delta=clu_pai_cli['f_lucro'].values[0] - clu_pai_cli['cls_lucro'].values[0])