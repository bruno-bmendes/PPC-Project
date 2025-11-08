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

    ss.page = "sistema_massa_mola"
    st.rerun()

# Definindo página
def sistema_massa_mola():

    # Declarando Variáveis
    if "mm_m" not in ss:
        ss.mm_m = 1.0
    if "mm_k" not in ss:
        ss.mm_k = 10.0
    if "mm_c" not in ss:
        ss.mm_c = 3.0
    if "mm_z0" not in ss:
        ss.mm_z0 = 0.1
    if "mm_zdot0" not in ss:
        ss.mm_zdot0 = 0.0
    if "mm_tmax" not in ss:
        ss.mm_tmax = 50.0

    # Definir Título
    ss.title = "Sistema Massa Mola"
    
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
            Considerando a Lei de Newton da Conservação do Movimento:
        """)

        st.latex(r"""
            m \frac{d^2 z(t)}{dt^2} = \sum_i F_i(t)
        """)

        st.markdown("""
            Analisando o sistema aplicamos as Forças presentes (Externa, Atrito, Elástica, Amortecimento):
        """)

        st.latex(r"""
            m \frac{d^2 z}{dt^2} = F_{ext} - F_{am} - F_{k}
        """)

        st.markdown("""
            Conhecendo as Forças aplicamos na equação:            
        """)

        st.latex(r"""
            m \frac{d^2 z(t)}{dt^2} + c \frac{dz(t)}{dt} + k z(t) = 0
        """)

        st.markdown("""
            A equação pode ser reescrita como:
        """)

        st.latex(r"""
            \ddot{z}(t) + 2\zeta\omega_n\dot{z}(t) + \omega_n^2 z(t) = 0
        """)

        st.markdown("### Regimes de amortecimento")

        st.markdown("**1. Subamortecido (0 < ζ < 1):**")

        st.latex(r"""
            z(t) = e^{-\zeta\omega_n t}\left( A_1\cos(\omega_d t) + A_2\sin(\omega_d t) \right)
        """)

        st.latex(r"""
            \omega_d = \omega_n\sqrt{1-\zeta^2}
        """)

        st.markdown("**2. Criticamente amortecido (ζ = 1):**")

        st.latex(r"""
            z(t) = (A_1 + A_2 t)e^{-\omega_n t}
        """)

        st.markdown("**3. Superamortecido (ζ > 1):**")

        st.latex(r"""
            z(t) = A_1 e^{s_1 t} + A_2 e^{s_2 t}
        """)

        st.latex(r"""
            s_{1,2} = -\omega_n\left(\zeta \mp \sqrt{\zeta^2 - 1}\right)
        """)

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    st.latex(r"""
        m \frac{d^2 z(t)}{dt^2} + c \frac{dz(t)}{dt} + k z(t) = 0
    """)

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            # Cálculo de parâmetros derivados
            wn = np.sqrt(ss.mm_k/ss.mm_m)               # frequência natural não amortecida (rad/s)
            zeta = ss.mm_c / (2*np.sqrt(ss.mm_k*ss.mm_m))     # fator de amortecimento (adimensional)
            wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0   # frequência amortecida (rad/s)

            # Tempo
            t = np.linspace(0, ss.mm_tmax, 1000)    # tempo (s)

            # Solução (subamortecido)
            if zeta < 1:
                A = ss.mm_z0                      # amplitude inicial (m)
                B = (ss.mm_zdot0 + zeta*wn*ss.mm_z0) / wd   # constante dependente das condições iniciais (m)
                z = np.exp(-zeta*wn*t) * (A*np.cos(wd*t) + B*np.sin(wd*t))   # posição (m)

            # Criticamente amortecido
            elif np.isclose(zeta, 1):
                A1 = ss.mm_z0                     # constante (m)
                A2 = ss.mm_zdot0 + wn*ss.mm_z0          # constante (m/s)
                z = (A1 + A2*t) * np.exp(-wn*t)   # posição (m)

            # Superamortecido
            else:
                s1 = -wn*(zeta - np.sqrt(zeta**2 - 1))   # raiz característica (1/s)
                s2 = -wn*(zeta + np.sqrt(zeta**2 - 1))   # raiz característica (1/s)
                A1 = (ss.mm_zdot0 - s2*ss.mm_z0)/(s1 - s2)           # constante (m)
                A2 = ss.mm_z0 - A1                             # constante (m)
                z = A1*np.exp(s1*t) + A2*np.exp(s2*t)    # posição (m)
            # Criação do gráfico
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=t, 
                y=z, 
                mode='lines',
                name=f'ζ = {zeta:.2f}',
                line=dict(width=2)
            ))

            # Layout do gráfico
            fig.update_layout(
                title='Resposta temporal z(t) - Sistema massa-mola-amortecedor',
                xaxis_title='Tempo (s)',
                yaxis_title='Posição z(t) (m)',
                template='plotly_white',
                width=800,
                height=400,
                legend=dict(x=0.8, y=1.1)
            )

            # Exibição no Streamlit
            st.plotly_chart(fig, use_container_width=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    # Cálculo de parâmetros derivados
    wn = np.sqrt(ss.mm_k/ss.mm_m)               # frequência natural não amortecida (rad/s)
    zeta = ss.mm_c / (2*np.sqrt(ss.mm_k*ss.mm_m))     # fator de amortecimento (adimensional)
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0   # frequência amortecida (rad/s)

    # Tempo
    t = np.linspace(0, ss.mm_tmax, 1000)    # tempo (s)

    # Solução (subamortecido)
    if zeta < 1:
        A = ss.mm_z0                      # amplitude inicial (m)
        B = (ss.mm_zdot0 + zeta*wn*ss.mm_z0) / wd   # constante dependente das condições iniciais (m)
        z = np.exp(-zeta*wn*t) * (A*np.cos(wd*t) + B*np.sin(wd*t))   # posição (m)

    # Criticamente amortecido
    elif np.isclose(zeta, 1):
        A1 = ss.mm_z0                     # constante (m)
        A2 = ss.mm_zdot0 + wn*ss.mm_z0          # constante (m/s)
        z = (A1 + A2*t) * np.exp(-wn*t)   # posição (m)

    # Superamortecido
    else:
        s1 = -wn*(zeta - np.sqrt(zeta**2 - 1))   # raiz característica (1/s)
        s2 = -wn*(zeta + np.sqrt(zeta**2 - 1))   # raiz característica (1/s)
        A1 = (ss.mm_zdot0 - s2*ss.mm_z0)/(s1 - s2)           # constante (m)
        A2 = ss.mm_z0 - A1                             # constante (m)
        z = A1*np.exp(s1*t) + A2*np.exp(s2*t)    # posição (m)
    # Criação do gráfico
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, 
        y=z, 
        mode='lines',
        name=f'ζ = {zeta:.2f}',
        line=dict(width=2)
    ))

    # Layout do gráfico
    fig.update_layout(
        title='Resposta temporal z(t) - Sistema massa-mola-amortecedor',
        xaxis_title='Tempo (s)',
        yaxis_title='Posição z(t) (m)',
        template='plotly_white',
        width=800,
        height=400,
        legend=dict(x=0.8, y=1.1)
    )

    # Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("pages/images/sistema_massa_mola.png")

    with col2:
        col3, col4 = st.columns([1, 1])
        with col3:
            # Massa
            with st.container(border=True):
                st.markdown(f"m = {ss.mm_m:.2f} kg")
                m = st.slider(
                    "", 0.5, 50.0, ss.mm_m, step=0.5
                )
                if m != ss.mm_m:
                    ss.mm_m = m
                    st.rerun()

            # Constante da Mola
            with st.container(border=True):
                st.markdown(f"k = {ss.mm_k:.2f} N/m")
                k = st.slider(
                    "", 1.0, 100.0, ss.mm_k, step=1.0
                )
                if k != ss.mm_k:
                    ss.mm_k = k
                    st.rerun()

            # Velocidade Inicial
            with st.container(border=True):
                st.markdown(f"v0 = {ss.mm_zdot0:.2f} m")
                zdot0 = st.slider(
                    "", 0.0, 20.0, ss.mm_zdot0, step=0.5
                )
                if zdot0 != ss.mm_zdot0:
                    ss.mm_zdot0 = zdot0
                    st.rerun()

            with st.container(border=True):
                st.markdown(f"ωn​ = {wn:.2f} rad/s")
                st.markdown(f"ζ = {zeta:.2f}")
                st.markdown(f"ωd = {wd:.2f} rad/s")

        with col4:
            # Constante da Mola
            with st.container(border=True):
                st.markdown(f"c = {ss.mm_c:.2f} N·s/m")
                c = st.slider(
                    "", 1.0, 20.0, ss.mm_c, step=0.5
                )
                if c != ss.mm_c:
                    ss.mm_c = c
                    st.rerun()

            # Posição Inicial
            with st.container(border=True):
                st.markdown(f"z0 = {ss.mm_z0:.2f} m")
                z0 = st.slider(
                    "", 0.1, 20.0, ss.mm_z0, step=0.1
                )
                if z0 != ss.mm_z0:
                    ss.mm_z0 = z0
                    st.rerun()

            # Tempo Máximo
            with st.container(border=True):
                st.markdown(f"Tempo Máximo = {ss.mm_tmax:.2f} s")
                tmax = st.slider(
                    "", 1.0, 400.0, float(ss.mm_tmax), step=1.0
                )
                if tmax != ss.mm_tmax:
                    ss.mm_tmax = tmax
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicativos
    st.markdown("""
        Ele representa um sistema massa-mola-amortecedor — um modelo clássico de segunda ordem em dinâmica, 
        pois há dois elementos que armazenam energia:
            A mola, que armazena energia potencial elástica,
            A massa, que armazena energia cinética associada ao movimento.            
    """)

    st.markdown("""
        Essa é uma equação diferencial linear de 2ª ordem, SISO, invariante no tempo e não forçada (pois não há uma força externa atuando).

        Quando o sistema é deslocado de sua posição de equilíbrio e liberado, ele entra em um regime transitório 
        até retornar à posição de repouso. 
            
        O comportamento é determinado pela relação entre massa, constante de mola e coeficiente de amortecimento, e o tipo de resposta depende do valor do fator de amortecimento (ζ).   
    """)

    st.markdown("""
        O sistema massa-mola-amortecedor pode apresentar três tipos de comportamento, dependendo da relação entre a constante da mola (k), a massa (m) e o coeficiente de amortecimento (c): 
                
            No regime subamortecido, a energia alterna entre a mola e a massa, produzindo oscilações que diminuem com o tempo até atingir o equilíbrio.
                
            No regime criticamente amortecido, o sistema retorna à posição de equilíbrio de forma rápida e estável, sem oscilações — é a resposta mais eficiente possível. 
                
            Já no regime superamortecido, o amortecimento é dominante e o sistema retorna lentamente à posição de equilíbrio, sem oscilar.            
    """)

    st.markdown("""
        Quando o sistema é perturbado, parte da energia é armazenada na mola, enquanto outra parte é dissipada pelo amortecedor. 

        A mola tende a puxar a massa de volta para o ponto de equilíbrio, enquanto o amortecedor resiste ao movimento, 
        reduzindo gradualmente a energia cinética até que o sistema pare completamente.

        O equilíbrio entre a rigidez da mola, a massa e o amortecimento define se o sistema oscila, se suaviza ou se movimenta lentamente até o repouso.            
    """)

    st.markdown("""
        **Subamortecido:** 
        Ocorre quando o amortecimento é pequeno (ζ < 1). 
        O sistema apresenta oscilações decrescentes, onde a amplitude diminui com o tempo até estabilizar.

        **Criticamente amortecido:** 
        Acontece quando o amortecimento atinge o valor crítico (ζ = 1). 
        O sistema retorna ao equilíbrio no menor tempo possível, sem oscilações — é a resposta mais rápida e estável.

        **Superamortecido:** 
        Ocorre quando o amortecimento é elevado (ζ > 1). 
        O sistema retorna lentamente à posição de equilíbrio, sem oscilações, devido à predominância da força de amortecimento.  
    """)

    space_line()

    # Sugestões
    st.markdown("""
        Ao alterar os parâmetros do sistema massa-mola-amortecedor, o gráfico de deslocamento z(t) muda de forma previsível, 
        refletindo o efeito de cada componente no comportamento dinâmico do sistema:

        Massa (m): quanto maior a massa, maior a inércia e mais lenta a resposta. Um aumento de m reduz a frequência natural, 
        fazendo o sistema oscilar mais devagar.

        Constante da mola (k): aumentar k torna a mola mais rígida, elevando a frequência natural e deixando o sistema mais “rápido” 
        e com oscilações mais frequentes. Diminuir k reduz a velocidade das oscilações e torna o sistema mais flexível.

        Coeficiente de amortecimento (c): quanto maior o valor de c, maior a dissipação de energia e, portanto, maior o amortecimento. 
        Um aumento de c reduz as oscilações, podendo levar ao regime superamortecido. Diminuir c faz o sistema oscilar mais antes de estabilizar.

        Tempo máximo (t_max): esse parâmetro não altera o comportamento físico do sistema, mas apenas o intervalo de tempo mostrado no gráfico. 
        Aumentar t_max permite observar todo o processo de amortecimento até o repouso, enquanto reduzir t_max limita a visualização apenas 
        ao início do movimento.            
    """)