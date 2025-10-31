# Importando Bibliotecas
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

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

    ss.page = "circuito_rc"
    st.rerun()

# Definindo página
def circuito_rc():

    # Declarando Variáveis
    if "rc_e" not in ss:
        ss.rc_e = 5.0
    if "rc_r" not in ss:
        ss.rc_r = 1000.0
    if "rc_c" not in ss:
        ss.rc_c = 100e-6
    if "rc_tmax" not in ss:
        ss.rc_tmax = 10

    # Definir Título
    ss.title = "Circuito RC"
    
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

                - Sistema com materiais ideais, sem preda de carga. 

                - Os parâmetros são invariantes no tempo.
        """)

        st.markdown("""
            Podemos iniciar usando a Lei das Malhas de Kirchhoff:            
        """)

        st.latex(r"\varepsilon = V_r + V_c")

        st.markdown("""
            Sabendo que os valores das tensões do Resistor e Capacitor são respectivamente:
        """)

        st.latex(r"V_r = R \cdot i \ \text{e} \ V_c = \frac{1}{C} \cdot q")

        st.markdown("Dessa forma:")

        st.latex(r"\varepsilon = R \cdot i + \frac{1}{C} \cdot q")

        st.markdown("Além disso, sabemos que a corrente _i_ é a variação da carga _q_ com o tempo, sendo assim, deixamos a equação apenas com uma variável indefinida, oferecendo o grau de liberdade necessário para a resolução:")

        st.latex(r"\varepsilon = R \cdot \frac{dq}{dt} + \frac{1}{C} \cdot q")

        space_line()

        st.markdown("A partir dessa equação chegamos em duas soluções, uma que nos dará a variação da carga _i_ no tempo, e outra que trará a variação da tensão _Vc_ no tempo:")
        
        st.latex(r"i(t) = \frac{\varepsilon}{R} e^{-t/(R C)}")
        st.latex(r"V_C(t) = \varepsilon \left( 1 - e^{-t/(R C)} \right)")

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    st.latex(r"\varepsilon = R \cdot \frac{dq}{dt} + \frac{1}{C} \cdot q")

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            # Constante de Tempo
            tau = ss.rc_r * ss.rc_c

            # Vetor de tempo
            t = np.linspace(0, 5*tau, 200)

            # Equações
            i = (ss.rc_e / ss.rc_r) * np.exp(-t / tau)              # Corrente
            Vc = ss.rc_e * (1 - np.exp(-t / tau))                   # Tensão no capacitor

            # Gráficos
            col1, col2 = st.columns([1, 1])

            # Gráfico 1: Corrente
            with col1:
                fig_i = go.Figure()

                fig_i.add_trace(go.Scatter(
                    x=t, 
                    y=i, 
                    mode='lines',
                    name='i(t) = (ε/R)·e^(-t/RC)',
                    line=dict(color='red')
                ))

                # Linha horizontal de referência em ε/R
                fig_i.add_hline(
                    y=ss.rc_e/ss.rc_r,
                    line=dict(color='gray', dash='dash'),
                    annotation_text='ε/R',
                    annotation_position='top left'
                )

                fig_i.update_layout(
                    title="Corrente no circuito RC durante a carga do capacitor",
                    xaxis_title="Tempo (s)",
                    yaxis_title="Corrente (A)",
                    template="plotly_white"
                )

                st.plotly_chart(fig_i, use_container_width=True)

            with col2:
                fig_vc = go.Figure()

                fig_vc.add_trace(go.Scatter(
                    x=t,
                    y=Vc,
                    mode='lines',
                    name='Vc(t) = ε·(1 - e^(-t/RC))',
                    line=dict(color='green')
                ))

                # Linha horizontal de referência em ε
                fig_vc.add_hline(
                    y=ss.rc_e,
                    line=dict(color='gray', dash='dash'),
                    annotation_text='ε',
                    annotation_position='top left'
                )

                fig_vc.update_layout(
                    title="Tensão no capacitor durante a carga",
                    xaxis_title="Tempo (s)",
                    yaxis_title="Tensão no capacitor (V)",
                    template="plotly_white"
                )

                st.plotly_chart(fig_vc, use_container_width=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    # Constante de Tempo
    tau = ss.rc_r * ss.rc_c

    # Vetor de tempo
    t = np.linspace(0, ss.rc_tmax, 200)

    # Equações
    i = (ss.rc_e / ss.rc_r) * np.exp(-t / tau)              # Corrente
    Vc = ss.rc_e * (1 - np.exp(-t / tau))                   # Tensão no capacitor

    # Gráficos
    col1, col2 = st.columns([1, 1])

    # Gráfico 1: Corrente
    with col1:
        fig_i = go.Figure()

        fig_i.add_trace(go.Scatter(
            x=t, 
            y=i, 
            mode='lines',
            name='i(t) = (ε/R)·e^(-t/RC)',
            line=dict(color='red')
        ))

        # Linha horizontal de referência em ε/R
        fig_i.add_hline(
            y=ss.rc_e/ss.rc_r,
            line=dict(color='gray', dash='dash'),
            annotation_text='ε/R',
            annotation_position='top left'
        )

        fig_i.update_layout(
            title="Corrente no circuito RC durante a carga do capacitor",
            xaxis_title="Tempo (s)",
            yaxis_title="Corrente (A)",
            template="plotly_white"
        )

        st.plotly_chart(fig_i, use_container_width=True)

    with col2:
        fig_vc = go.Figure()

        fig_vc.add_trace(go.Scatter(
            x=t,
            y=Vc,
            mode='lines',
            name='Vc(t) = ε·(1 - e^(-t/RC))',
            line=dict(color='green')
        ))

        # Linha horizontal de referência em ε
        fig_vc.add_hline(
            y=ss.rc_e,
            line=dict(color='gray', dash='dash'),
            annotation_text='ε',
            annotation_position='top left'
        )

        fig_vc.update_layout(
            title="Tensão no capacitor durante a carga",
            xaxis_title="Tempo (s)",
            yaxis_title="Tensão no capacitor (V)",
            template="plotly_white"
        )

        st.plotly_chart(fig_vc, use_container_width=True)

    # Separar Esquema e Variáveis
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("pages/images/esquema_circuito_rc.png")

    with col2:
        col3, col4 = st.columns([1, 1])
        with col3:
            # Fonte
            with st.container(border=True):
                st.markdown(f"E = {ss.rc_e:.2f} V")
                e = st.slider(
                    "", 1.0, 50.0, ss.rc_e, step=1.0, key="sl_rc_e"
                )
                if e != ss.rc_e:
                    ss.rc_e = e
                    st.rerun()

            # Resistor
            with st.container(border=True):
                st.markdown(f"R = {ss.rc_r:.2f} Ω")
                r = st.slider(
                    "", 100.0, 10000.0, ss.rc_r, step=10.0, key="sl_rc_r"
                )
                if r != ss.rc_r:
                    ss.rc_r = r
                    st.rerun()

        with col4:
            # Capacitor
            with st.container(border=True):
                st.markdown(f"C = {(ss.rc_c / 10e-6):.2f} * 10⁻⁶ F")
                temp_c = ss.rc_c / 10e-6
                def change_c():
                    ss.rcl_c = c * 10e-6
                c = st.slider(
                    "", 10, 1000, 100, step=10, key="sl_rc_c", on_change=change_c
                )
                if (c * 10e-6) != ss.rc_c:
                    ss.rc_c = c * 10e-6
                    st.rerun()

            with st.container(border=True):
                st.markdown(f"t = {ss.rc_tmax:.2f} s")
                tmax = st.slider(
                    "", 1.0, 200.0, float(ss.rc_tmax), step=1.0, key="sl_rc_tmax"
                )
                if tmax != ss.rc_tmax:
                    ss.rc_tmax = tmax
                    st.rerun()

            with st.container(border=True):
                st.markdown(f"τ = {tau:.2f} s")

    # Linha divisória
    space_line()

    # Textos explicatios
    st.markdown("""
        O circuito é formado por uma fonte de tensão (ε), um resistor (R) e um capacitor (C) ligados em série. A chave (S) controla o momento em que o capacitor começa a carregar ou descarregar.            
    """)

    st.markdown("""
        Ao fechar a chave, a corrente começa a fluir, parte da energia é dissipada no resistor e parte é armazenada no capacitor, o capacitor acumula carga elétrica nas placas até atingir a tensão da fonte.            
    """)

    st.markdown("""
        A constante de tempo (τ = RC) mede o ritmo de resposta do circuito. Em 𝑡 = 𝜏, o capacitor carrega 63% da tensão total (ou descarrega até 37%). Após cerca de 5τ, o circuito atinge o regime permanente (99,3% do regime final).
    """)

    st.markdown("""
        Funções:
            \nResistor: dissipa energia em calor (efeito Joule).
            \nCapacitor: armazena energia no campo elétrico.
            \nFonte: fornece energia elétrica ao sistema.            
    """)

    space_line()

    st.markdown("""
        **Aumentando a Capacitância (C):**  
        A constante de tempo (τ = R·C) aumenta, fazendo o capacitor carregar e descarregar mais lentamente.  
        A curva de tensão (Vᴄ) fica mais suave e o sistema armazena mais energia.            
    """)

    st.markdown("""
        **Diminuindo a Capacitância (C):**  
        O capacitor carrega e descarrega mais rapidamente, com uma curva mais inclinada no início.  
        O sistema responde mais rápido, mas armazena menos energia.            
    """)

    st.markdown("""
        **Aumentando a Resistência (R):**  
        O tempo de resposta do circuito aumenta (τ = R·C maior), reduzindo a corrente inicial.  
        O capacitor demora mais para atingir a tensão final, e o resistor dissipa mais energia em calor.            
    """)

    st.markdown("""
        **Diminuindo a Resistência (R):**  
        O capacitor carrega mais rapidamente, com maior corrente inicial.  
        A resposta é mais rápida, porém há maior dissipação instantânea de energia.            
    """)

    st.markdown("""
        **Aumentando a Tensão da Fonte (ε):**  
        O nível final de tensão no capacitor aumenta proporcionalmente, sem alterar o formato da curva.  
        O capacitor atinge um valor de equilíbrio mais alto.            
    """)

    st.markdown("""
        **Diminuindo a Tensão da Fonte (ε):**  
        O nível final de tensão no capacitor diminui, mantendo o mesmo tempo de resposta.  
        A curva de carga se torna mais baixa, com o mesmo comportamento exponencial.            
    """)