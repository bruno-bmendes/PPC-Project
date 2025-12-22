# Importando Bibliotecas
import streamlit as st

# Importando Funções
from functions.basic_functions import *

# Inicializando session state
ss = st.session_state

# Definindo session states
if "page" not in ss:
    if "title" not in ss:
        ss.title = "Simulador PPC"

    if "page_set" not in ss:
        ss.page_set = ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola Amortecedor", "Pêndulo Simples Amortecido", "Sistema Eletromecanico", "Tanque com Aquecimento", "Motor Bomba de um Poço BCS"]

    ss.page = "inicio"
    st.rerun()

# Definindo página
def page_inicio():

    # Definir Título
    ss.title = "Simulador PPC"
    
    # Cabeçalho
    col1, col2, col3 = st.columns([13, 1, 4])

    with col1:
        st.markdown(f"""
            <h1 style='text-align: left; margin: 0;'>{ss.title}</h1>
        """, unsafe_allow_html=True)

    with col3:
        pages_selection = st.selectbox(
            "", 
            [page for page in ss.page_set if normalize_title(page) != normalize_title(ss.page)],
            index=None,
            placeholder="Escolha um Cenário"
        )

        if pages_selection is not None:
            if normalize_title(pages_selection) != ss.page:
                ss.page = normalize_title(pages_selection)
                st.rerun()

    # Linha para separar o cabeçalho
    st.markdown("""
        <hr style="border: 1px solid #bbb; margin-top: 1rem; margin-bottom: 1rem;" />
    """, unsafe_allow_html=True)
    
    # CSS do bloco central
    st.markdown("""
        <style>
            .custom-container {
                width: 60%;
                height: 300px;
                margin: 80px auto;
                display: flex;
                justify-content: center;
                align-items: center;
                border: 2px solid #444;
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.0); /* fundo transparente */
                color: #000;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
            }

            .stApp {
                background-color: #dcdcdc;
            }

            header, footer {
                visibility: hidden;
            }
        </style>
    """, unsafe_allow_html=True)

    # Renderiza o container com a mensagem
    st.markdown("<div class='custom-container'>SELECIONE UM CENÁRIO PARA INICIAR O SIMULADOR</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center; font-size: 18px;'>
            Visite o repositório no 
            <a href='https://github.com/bruno-bmendes/PPC-Project' target='_blank'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

