# Importando Bibliotecas
import streamlit as st
import pandas as pd

# Importando funções
from functions.basic_functions import *
from functions.streamlit_functions import *

# Importando páginas
from pages.page_inicio import *
from pages.vaso_pulmao import *
from pages.circuito_rc import *
from pages.circuito_rlc import *
from pages.sistema_massa_mola_amortecedor import *
from pages.pendulo_simples_amortecido import *
from pages.sistema_eletromecanico import *
from pages.tanque_com_aquecimento import *

# Configurações da página
st.set_page_config(page_title="Simulador PPC", initial_sidebar_state="collapsed", layout="wide", page_icon="💡")

# Cor de Background
st.markdown("""
    <style>
        .stApp {
            background-color: #e6e6e6;
        }
        header {
            visibility: hidden;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Inicializando session state
ss = st.session_state

# Definindo session states
if "page" not in ss:
    if "title" not in ss:
        ss.title = "Simulador PPC"

    if "page_set" not in ss:
        ss.page_set = ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola Amortecedor", "Pêndulo Simples Amortecido", "Sistema Eletromecanico", "Tanque com Aquecimento"]

    ss.page = "inicio"
    st.rerun()

# Definindo páginas
if ss.page == "inicio":
    page_inicio()

elif ss.page == "vaso_pulmao":
    vaso_pulmao()

elif ss.page == "circuito_rc":
    circuito_rc()

elif ss.page == "circuito_rlc":
    circuito_rlc()

elif ss.page == "sistema_massa_mola_amortecedor":
    sistema_massa_mola_amortecedor()

elif ss.page == "pendulo_simples_amortecido":
    pendulo_simples_amortecido()

elif ss.page == "sistema_eletromecanico":
    sistema_eletromecanico()

elif ss.page == "tanque_com_aquecimento":
    tanque_com_aquecimento()