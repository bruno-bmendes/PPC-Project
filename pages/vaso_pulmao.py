# Importando Bibliotecas
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import base64

# Importando Funções
from functions.basic_functions import *
from functions.streamlit_functions import *

# Inicializando session state
ss = st.session_state

# Definindo session states
if "page" not in ss:
    if "title" not in ss:
        ss.title = "Simulador PPC"

    if "page_set" not in ss:
        ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola"]

    ss.page = "vaso_pulmao"
    st.rerun()

# Definindo página
def vaso_pulmao():

    # Declarando Variáveis
    if "vp_v" not in ss:
        ss.vp_v = 1.0
    if "vp_mm" not in ss:
        ss.vp_mm = 0.029
    if "vp_t" not in ss:
        ss.vp_t = 300
    if "vp_k1" not in ss:
        ss.vp_k1 = 0.5
    if "vp_k2" not in ss:
        ss.vp_k2 = 0.4
    if "vp_p0" not in ss:
        ss.vp_p0 = 2e5
    if "vp_p1" not in ss:
        ss.vp_p1 = 5e5
    if "vp_p2" not in ss:
        ss.vp_p2 = 1e5
    if "vp_r" not in ss:
        ss.vp_r = 8.314
    if "vp_t_final" not in ss:
        ss.vp_t_final = 200

    # Definir Título
    ss.title = "Sistema Vaso Pulmão"
    
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
    space_line()
    
    # Botão de Info
    @st.dialog("Desenvolvendo a Equação")
    def info():

        st.markdown("""
            Podemos declarar inicialmente as seguintes premissas:

                - Sistema fechado com uma entrada e uma saída, conectadas a pressões P1 e P2;

                - O volume do vaso é constante;

                - Não há reação química interna, ou seja, a massa total só varia por entrada e saída;
                
                - O gás é ideal e as propriedades do fluido são constantes;

                - A pressão dentro do vaso é uniforme;

            Sabemos que o acumulo de massa em relação ao tempo, é dado pela Vazão de Entrada (F1) menos a Vazão de Saída (F2)
        """)

        st.latex(r'''
            \frac{dm}{dt} = F_1(t) - F_2(t)
        ''')

        st.markdown("E como F é dado como uma constante k multiplicada pela raiz quadrada da pressão, temos que:")

        st.latex(r'''
            \frac{dm}{dt} = k_1 \sqrt{P_1(t) - P(t)} - k_2 \sqrt{P(t) - P_2(t)}
        ''')

        st.markdown("Como estamos trabalhando com gases ideais, podemos considerar a lei universal:")

        st.latex(r'''
            PV = nRT
        ''')

        st.markdown("Podemos transformar o Numero de mols na equação:")

        st.latex(r'''
            P \cdot V = \frac{m}{MM} \cdot R \cdot T \quad (\text{onde } MM = \text{massa molecular})
        ''')

        st.markdown("E isolando a massa na equação:")

        st.latex(r'''
            m = \frac{P \cdot V \cdot MM}{R \cdot T}
        ''')

        st.markdown("""
            Nesse sistema, de acordo com as premissas adotadas, os únicos valores não constantes são a Pressão e a Temperatura.

            Ou ainda, se considerarmos uma variação pequena na Pressão, podemos considerar o sistema Isotérmico. Com isso, temos o sistema apenas com a pressão variando com o tempo:
        """)
        
        st.latex(r"""
            \frac{V\,mm}{R\,T}\,\frac{dP}{dt}
            = k_1\sqrt{P_1(t)-P(t)} \;-\; k_2\sqrt{\,P(t)-P_2(t)\,}
        """)

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    st.latex(r'''
        \frac{VMM}{RT} \frac{dP}{dt} = k_1 \sqrt{P_1(t) - P(t)} - k_2 \sqrt{P(t) - P_2(t)}
    ''')

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            # Equação diferencial
            def dP_dt(t, P):
                return (ss.vp_r * ss.vp_t / (ss.vp_v * ss.vp_mm)) * (
                    ss.vp_k1 * np.sqrt(max(ss.vp_p1 - P[0], 0)) -
                    ss.vp_k2 * np.sqrt(max(P[0] - ss.vp_p2, 0))
                )

            # Definindo tempo de simulação
            t_span = (0, ss.vp_t_final)
            t_eval = np.linspace(t_span[0], t_span[1], 200)

            # Resolvendo Equação Diferencial
            sol = solve_ivp(dP_dt, t_span, [ss.vp_p0], t_eval=t_eval)

            fig = go.Figure()

            # Curva de pressão no tempo
            fig.add_trace(go.Scatter(
                x=sol.t,
                y=sol.y[0] / 1e5,
                mode='lines',
                name='Pressão no vaso (P)',
                line=dict(color='black')
            ))

            # Linha de P1 (entrada)
            fig.add_trace(go.Scatter(
                x=sol.t,
                y=[ss.vp_p1 / 1e5] * len(sol.t),
                mode='lines',
                name='P1 (entrada)',
                line=dict(color='red', dash='dash')
            ))

            # Linha de P2 (saída)
            fig.add_trace(go.Scatter(
                x=sol.t,
                y=[ss.vp_p2 / 1e5] * len(sol.t),
                mode='lines',
                name='P2 (saída)',
                line=dict(color='blue', dash='dash')
            ))

            # Layout do gráfico
            fig.update_layout(
                title="Simulação Isotérmica do Vaso Pulmão",
                xaxis_title="Tempo (s)",
                yaxis_title="Pressão (bar)",
                template="simple_white",
                hovermode="x unified",
                legend=dict(title="", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )

            # Exibe no Streamlit
            st.plotly_chart(fig, use_container_width=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    # Equação diferencial
    def dP_dt(t, P):
        return (ss.vp_r * ss.vp_t / (ss.vp_v * ss.vp_mm)) * (
            ss.vp_k1 * np.sqrt(max(ss.vp_p1 - P[0], 0)) -
            ss.vp_k2 * np.sqrt(max(P[0] - ss.vp_p2, 0))
        )

    # Definindo tempo de simulação
    t_span = (0, ss.vp_t_final)
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    # Resolvendo Equação Diferencial
    sol = solve_ivp(dP_dt, t_span, [ss.vp_p0], t_eval=t_eval)

    fig = go.Figure()

    # Curva de pressão no tempo
    fig.add_trace(go.Scatter(
        x=sol.t,
        y=sol.y[0] / 1e5,
        mode='lines',
        name='Pressão no vaso (P)',
        line=dict(color='black')
    ))

    # Linha de P1 (entrada)
    fig.add_trace(go.Scatter(
        x=sol.t,
        y=[ss.vp_p1 / 1e5] * len(sol.t),
        mode='lines',
        name='P1 (entrada)',
        line=dict(color='red', dash='dash')
    ))

    # Linha de P2 (saída)
    fig.add_trace(go.Scatter(
        x=sol.t,
        y=[ss.vp_p2 / 1e5] * len(sol.t),
        mode='lines',
        name='P2 (saída)',
        line=dict(color='blue', dash='dash')
    ))

    # Layout do gráfico
    fig.update_layout(
        title="Simulação Isotérmica do Vaso Pulmão",
        xaxis_title="Tempo (s)",
        yaxis_title="Pressão (bar)",
        template="simple_white",
        hovermode="x unified",
        legend=dict(title="", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    # Exibe no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Separar Esquema e Variáveis
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("pages/images/esquema_vaso_pulmao.png")

    with col2:
        col3, col4 = st.columns([1, 1])
        with col3:
            # Volume
            with st.container(border=True):
                st.markdown(f"V = {ss.vp_v:.2f} m³")
                v = st.slider(
                    "", 0.1, 10.0, 1.0, step=0.1, key="sl_vp_v"
                )
                if v != ss.vp_v:
                    ss.vp_v = v
                    st.rerun()

            # Massa Molar
            with st.container(border=True):
                st.markdown(f"MM = {ss.vp_mm:.2f} kg/mol")
                mm = st.slider(
                    "", 0.001, 1.0, 0.029, step=0.001, key="sl_vp_mm"
                )
                if mm != ss.vp_mm:
                    ss.vp_mm = mm
                    st.rerun()

            # Constante R
            with st.container(border=True):
                st.markdown(f"R = {ss.vp_r:.2f} J/(mol*K)")

            # Temperatura
            with st.container(border=True):
                st.markdown(f"T = {ss.vp_t:.2f} K")
                t = st.slider(
                    "", 1.0, 600.0, 300.0, step=1.0, key="sl_vp_t"
                )
                if t != ss.vp_t:
                    ss.vp_t = t
                    st.rerun()

            # Pressão Inicial
            with st.container(border=True):
                st.markdown(f"P0 = {(ss.vp_p0/1e5):.2f} bar")
                p0 = st.slider(
                    "", 0.1, 10.0, 2.0, step=0.1, key="sl_vp_p0"
                )
                if p0 != (ss.vp_p0 / 1e5):
                    ss.vp_p0 = p0 * 1e5
                    st.rerun()

        with col4:
            # Constante de Material k1
            with st.container(border=True):
                st.markdown(f"k1 = {ss.vp_k1:.2f}")
                k1 = st.slider(
                    "", 0.1, 3.0, 0.5, step=0.1, key="sl_vp_k1"
                )
                if k1 != ss.vp_k1:
                    ss.vp_k1 = k1
                    st.rerun()
                    
            # Constante de Material k2
            with st.container(border=True):
                st.markdown(f"k2 = {ss.vp_k2:.2f}")
                k2 = st.slider(
                    "", 0.1, 3.0, 0.4, step=0.1, key="sl_vp_k2"
                )
                if k2 != ss.vp_k2:
                    ss.vp_k2 = k2
                    st.rerun()

            # Pressão 1
            with st.container(border=True):
                st.markdown(f"P1 = {(ss.vp_p1/1e5):.2f} bar")
                p1 = st.slider(
                    "", 0.1, 10.0, 5.0, step=0.1, key="sl_vp_p1"
                )
                if p1 != (ss.vp_p1 / 1e5):
                    ss.vp_p1 = p1 * 1e5
                    st.rerun()

            # Pressão 2
            with st.container(border=True):
                st.markdown(f"P2 = {(ss.vp_p2/1e5):.2f} bar")
                p2 = st.slider(
                    "", 0.1, 10.0, 1.0, step=0.1, key="sl_vp_p2"
                )
                if p2 != (ss.vp_p2 / 1e5):
                    ss.vp_p2 = p2 * 1e5
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicatios
    st.markdown("""O vaso pulmão é usado para suavizar oscilações de pressão em sistemas de gás. Ele acumula fluido quando a pressão de entrada é alta e libera quando a saída exige, garantindo um escoamento mais estável.""")

    st.markdown("""A pressão no vaso (linha amarela) tende a um valor de equilíbrio entre a pressão de entrada (vermelha) e a de saída (azul), neste momento, as vazões de entrada e saída se igualam.""")
    st.markdown("""Para esse sistema consideramos o volume do vaso, a massa molar do gás constantes. Além disso consideramos um sistema isotérmico. Se a variação de temperatura fosse considerada, precisaríamos de uma equação de energia para fechar o sistema, tornando-o mais complexo (não isotérmico).""")

    space_line()

    st.markdown("""
        🔸 **Aumentando P₁ (pressão de entrada):** o vaso recebe mais fluido, fazendo a curva amarela subir mais rapidamente até um novo equilíbrio mais alto.           
    """)

    st.markdown("""
        🔸 **Diminuindo P₂ (pressão de saída):** a resistência à saída aumenta, o acúmulo dentro do vaso cresce e o equilíbrio se desloca para uma pressão interna maior.            
    """)

    st.markdown("""
        🔸 **Alterando P(0):** muda o ponto de partida da simulação; o sistema ainda tende ao mesmo equilíbrio, mas com trajetórias diferentes.            
    """)

    st.markdown("""
        🔸 **Maior diferença entre P₁ e P₂:** o sistema responde de forma mais intensa, com transientes mais rápidos e maior fluxo de massa.           
    """)