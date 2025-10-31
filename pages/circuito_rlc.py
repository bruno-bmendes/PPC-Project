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
def circuito_rlc():

    # Declarando Variáveis
    if "rcl_e" not in ss:
        ss.rcl_e = 100.0
    if "rcl_r" not in ss:
        ss.rcl_r = 5.0
    if "rcl_c" not in ss:
        ss.rcl_c = 10000e-6 
    if "rcl_l" not in ss:
        ss.rcl_l = 1000e-3
    if "rcl_tmax" not in ss:
        ss.rcl_tmax = 4

    # Definir Título
    ss.title = "Circuito RLC"
    
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

        st.latex(r"\varepsilon = V_r + V_c + V_L")

        st.markdown("""
            Sabendo que os valores das tensões do Resistor, Capacitor e Indutor são respectivamente:
        """)

        st.latex(r"V_C = \frac{q}{C}, \quad V_R = Ri, \quad V_L = L \frac{di}{dt}")

        st.markdown("Portanto ficamos com a equação:")

        st.latex(r"\varepsilon = \frac{q}{C} + Ri + L \frac{di}{dt}")

        st.markdown(r"Sabendo que $i = \frac{dq}{dt}$, temos:")

        st.latex(r"\varepsilon = \frac{1}{C}q + R\frac{dq}{dt} + L\frac{d^2q}{dt^2}")

        st.markdown(r"Ou Dividindo tudo por $L$:")

        st.latex(r"\frac{\varepsilon(t)}{L} = \frac{d^2 q(t)}{dt^2} + \frac{R}{L} \frac{dq(t)}{dt} + \frac{1}{LC} q(t)")

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    st.latex(r"\frac{\varepsilon(t)}{L} = \frac{d^2 q(t)}{dt^2} + \frac{R}{L} \frac{dq(t)}{dt} + \frac{1}{LC} q(t)")


    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            # Definindo intervalo de tempo
            t = np.linspace(0, 4, 1000)

            def rlc_response(R, L, C, E, t):
                alpha = R / (2 * L)
                omega_0 = 1 / np.sqrt(L * C)
                disc = alpha**2 - omega_0**2

                if disc < 0:  # subamortecido
                    omega_d = np.sqrt(omega_0**2 - alpha**2)
                    q = C * E * (1 - np.exp(-alpha*t) * (np.cos(omega_d*t) + (alpha/omega_d)*np.sin(omega_d*t)))
                elif disc == 0:  # crítico
                    q = C * E * (1 - np.exp(-alpha*t) * (1 + alpha*t))
                else:  # superamortecido
                    r1 = -alpha + np.sqrt(disc)
                    r2 = -alpha - np.sqrt(disc)
                    A = C * E * r2 / (r2 - r1)
                    B = -C * E * r1 / (r2 - r1)
                    q = C * E * (1 - A*np.exp(r1*t) - B*np.exp(r2*t))
                return q

            # respostas
            q = rlc_response(ss.rcl_r, ss.rlc_l, ss.rlc_c, ss.rlc_e, t)

            # Gráfico de Carga
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=q_sub, mode='lines', name='Subamortecido', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=t, y=q_crit, mode='lines', name='Criticamente Amortecido', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=t, y=q_super, mode='lines', name='Superamortecido', line=dict(color='red')))

            fig.update_layout(
                title='Resposta da Carga q(t) em Circuitos RLC',
                xaxis_title='Tempo (s)',
                yaxis_title='Carga (q)',
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    # Definindo intervalo de tempo
    t = np.linspace(0, ss.rcl_tmax, 1000)

    def rlc_response(R, L, C, E, t):
        alpha = R / (2 * L)
        omega_0 = 1 / np.sqrt(L * C)
        disc = alpha**2 - omega_0**2

        if disc < 0:  # subamortecido
            omega_d = np.sqrt(omega_0**2 - alpha**2)
            q = C * E * (1 - np.exp(-alpha*t) * (np.cos(omega_d*t) + (alpha/omega_d)*np.sin(omega_d*t)))
        elif disc == 0:  # crítico
            q = C * E * (1 - np.exp(-alpha*t) * (1 + alpha*t))
        else:  # superamortecido
            r1 = -alpha + np.sqrt(disc)
            r2 = -alpha - np.sqrt(disc)
            A = C * E * r2 / (r2 - r1)
            B = -C * E * r1 / (r2 - r1)
            q = C * E * (1 - A*np.exp(r1*t) - B*np.exp(r2*t))
        return q

    # respostas
    q = rlc_response(ss.rcl_r, ss.rcl_l, ss.rcl_c, ss.rcl_e, t)

    # Gráfico de Carga
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=q, mode='lines', line=dict(color='green')))

    fig.update_layout(
        title='Resposta da Carga q(t) em Circuitos RLC',
        xaxis_title='Tempo (s)',
        yaxis_title='Carga (q)',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("pages/images/esquema_circuito_rlc.png")

    with col2:
        col3, col4 = st.columns([1, 1])
        with col3:
            # Fonte
            with st.container(border=True):
                st.markdown(f"E = {ss.rcl_e:.2f} V")
                e = st.slider(
                    "", 10.0, 500.0, ss.rcl_e, step=1.0
                )
                if e != ss.rcl_e:
                    ss.rcl_e = e
                    st.rerun()

            # Resistor
            with st.container(border=True):
                st.markdown(f"R = {ss.rcl_r:.2f} Ω")
                r = st.slider(
                    "", 1.0, 100.0, ss.rcl_r, step=1.0
                )
                if r != ss.rcl_r:
                    ss.rcl_r = r
                    st.rerun()

            with st.container(border=True):
                st.markdown(f"R² = {ss.rcl_r**2:.2f} Ω²")
                st.markdown(f"4L/C = {(4 * ss.rcl_l / ss.rcl_c):.2f} s²")

        with col4:
            # Capacitor
            with st.container(border=True):
                st.markdown(f"C = {(ss.rcl_c / 10e-6):.2f} * 10⁻⁶ F")
                temp_c = ss.rcl_c / 10e-6
                def change_c():
                    ss.rcl_c = c * 10e-6
                c = st.slider(
                    "", 100.0, 50000.0, float(temp_c), step=10.0, on_change=change_c
                )
                if (c * 10e-6) != ss.rcl_c:
                    ss.rcl_c = c * 10e-6
                    st.rerun()

            # Indutor
            with st.container(border=True):
                st.markdown(f"L = {(ss.rcl_l / 10e-3):.2f} * 10⁻³ H")
                temp_l = ss.rcl_l / 10e-3
                def change_l():
                    ss.rcl_l = l * 10e-3
                l = st.slider(
                    "", 10.0, 10000.0, float(temp_l), step=10.0, on_change=change_l
                )
                if (l * 10e-3) != ss.rcl_l:
                    ss.rcl_l = l * 10e-3
                    st.rerun()

            # Tempo Máximo
            with st.container(border=True):
                st.markdown(f"Tempo Máximo = {ss.rcl_tmax:.2f} s")
                tmax = st.slider(
                    "", 0.5, 400.0, float(ss.rcl_tmax), step=0.5
                )
                if tmax != ss.rcl_tmax:
                    ss.rcl_tmax = tmax
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicatios
    st.markdown("""
        Ele representa um sistema de segunda ordem — mais complexo que o RC, pois há dois elementos de armazenamento de energia:
            O capacitor, que armazena energia em campo elétrico,
            O indutor, que armazena energia em campo magnético.            
    """)

    st.markdown("""
        Essa é uma equação diferencial linear de 2ª ordem, SISO, invariante no tempo e forçada (pois há uma fonte 𝜀).

        Quando a fonte é constante (degrau DC), o circuito entra em um regime transitório e depois estacionário.
             
        O comportamento é determinado pela relação entre R, L e C, e o tipo de amortecimento depende de:   
    """)

    st.markdown("""
        O circuito RLC pode apresentar três tipos de comportamento, dependendo da relação entre a resistência, a indutância e a capacitância: 
            
            No regime subamortecido, a energia oscila entre o capacitor e o indutor, gerando variações periódicas que diminuem com o tempo até atingir o equilíbrio.
            
            No regime criticamente amortecido, o circuito atinge o valor final de forma rápida e estável, sem oscilações — é a resposta mais eficiente possível. 
            
            Já no regime superamortecido, a resistência é dominante e a tensão cresce lentamente, aproximando-se do valor final sem ultrapassá-lo.            
    """)

    st.markdown("""
        Quando o circuito é ligado, a fonte injeta energia que se divide entre L e C.

        O indutor resiste à mudança de corrente, atrasando o carregamento do capacitor.

        O capacitor acumula energia, enquanto o resistor dissipa parte dela.

        O balanço entre essas energias define se o sistema vai oscilar, suavizar ou responder lentamente.            
    """)

    st.markdown("""
        **Subamortecido:** 
        Ocorre quando R² < 4L/C. A energia oscila entre o indutor e o capacitor, produzindo uma resposta com oscilações decrescentes até o equilíbrio.

        **Criticamente amortecido:** 
        Acontece quando R² = 4L/C. O sistema atinge o valor final da forma mais rápida possível, sem oscilar — é a resposta mais estável e eficiente.

        **Superamortecido:** 
        Ocorre quando R² > 4L/C. O resistor domina o comportamento do circuito, que cresce lentamente e se aproxima do valor final sem ultrapassá-lo.  
    """)

    space_line()

    # Sugestões
    st.markdown("""
        Ao alterar os parâmetros do circuito RLC, o gráfico de carga (ou tensão no capacitor) muda de forma previsível, refletindo o efeito de cada componente no comportamento dinâmico do sistema:

        Resistência (R): quanto maior o valor de R, maior a dissipação de energia e, portanto, o amortecimento. Um aumento de R faz o gráfico ficar mais “lento” e menos oscilatório, podendo chegar ao regime superamortecido. Diminuir R reduz o amortecimento, fazendo a carga oscilar mais em torno do valor final antes de estabilizar.

        Capacitância (C): aumentar C faz o capacitor armazenar mais carga, tornando o sistema mais “pesado” e lento para responder — o gráfico fica mais suave e o tempo para atingir o equilíbrio aumenta. Diminuir C deixa o sistema mais ágil, com respostas mais rápidas e maiores oscilações.

        Indutância (L): o indutor resiste à variação de corrente. Aumentar L também retarda a resposta, alongando o tempo das oscilações e tornando-as mais suaves. Diminuir L deixa o sistema mais rápido e com oscilações mais frequentes.

        Fonte (E): a tensão da fonte define o nível final de carga do capacitor. Aumentar E eleva proporcionalmente a amplitude do gráfico, sem alterar a forma da curva. Diminuir E reduz a escala da resposta, mantendo o mesmo comportamento dinâmico.

        Tempo máximo (t_max): esse parâmetro não altera o comportamento físico do circuito, mas apenas o intervalo visualizado no gráfico. Aumentar t_max permite observar todo o processo de amortecimento até o regime permanente, enquanto reduzir t_max limita a visualização apenas ao início do transiente.            
    """)
    