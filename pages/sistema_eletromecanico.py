# Importando Bibliotecas
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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
        ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola Amortecedor", "Pêndulo Simples Amortecido", "Sistema Eletromecanico"]

    ss.page = "sistema_eletromecanico"
    st.rerun()

# Definindo página
def sistema_eletromecanico():

    # Declarando Variáveis
    if "mdc_J" not in ss:
        ss.mdc_J = 0.01      # inércia [kg.m²]
    if "mdc_k1" not in ss:
        ss.mdc_k1 = 0.1      # constante de torque
    if "mdc_k2" not in ss:
        ss.mdc_k2 = 0.1      # constante de f.c.e.m.
    if "mdc_R" not in ss:
        ss.mdc_R = 1.0       # resistência [ohm]
    if "mdc_b" not in ss:
        ss.mdc_b = 0.01      # atrito viscoso
    if "mdc_E0" not in ss:
        ss.mdc_E0 = 24.0     # tensão de entrada (degrau)
    if "mdc_tmax" not in ss:
        ss.mdc_tmax = 2.0    # tempo total de simulação
    if "mdc_theta0" not in ss:
        ss.mdc_theta0 = 0.0    # ângulo inicial
    if "mdc_omega0" not in ss:
        ss.mdc_omega0 = 0.0    # velocidade angular inicial

    # Definir Título
    ss.title = "Motor DC"
    
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
            Definindo as premissas do sistema:
            - Motor operando em regime linear, sem saturação;
            - Indutância elétrica desprezada;
            - Atrito modelado como viscoso;
            - Inércia mecânica constante;
            - Campo magnético constante e coeficientes k1 e k2 fixos;
            - Ausência de cargas externas que dependam de tempo ou posição;
            - Temperatura constante, resistência fixa;
            - Comutação e efeitos de ripple ignorados.
        """)

        st.markdown("Começaremos destrinchando o sistema a partir da Segunda Lei de Newton:")

        st.latex(r"\sum F = ma")

        st.markdown("Se formos considerar a equação análoga para o movimento rotacional, chegamos na Lei de Newton–Euler:")

        st.latex(r"\sum T = J \alpha")

        st.markdown("Definindo a aceleração angular como a segunda derivada do ângulo, e que precisamos considerar o Torque Resultante no sistema, chegamos na equação:")

        st.latex(r"J \frac{d^2 \theta}{dt^2} = T_r")

        st.markdown("Considerando que no sistema o Torque Resultante será dado pelo Torque gerado pelo motor mais o Torque Resistivo, causado pelo atrito, temos que:")

        st.latex(r"J \frac{d^2 \theta}{dt^2} = T_r")

        st.markdown("Para o Torque gerado pelo mototr, sabemos que Tg é proporcional à corrente que passa no enrolamento da armadura, então se adicionarmos uma constante k1, teremos que:")

        st.latex(r"T_g(t) = k_1\, i(t)")

        st.markdown("""
            Já o Torque Resistivo é inversamente proporcional à velocidade ângular (Se o motor gira devagar → o atrito é pequeno, se gira rápido → o atrito cresce proporcionalmente.). Considerando também a constante de atrito viscoso b, teremos:
        """)

        st.latex(r"T_f(t) = b\, \omega(t)")

        st.markdown("Dessa forma ficamos com a primeira parte da equação:")

        st.latex(r"J \frac{d^2 \theta(t)}{dt^2} = k_1\, i(t) - b\, \omega(t)")

        st.markdown("Para a parte elétroca, começaremos explicitando a Lei das Malhas considerando a Resistência e a Força Eletromotriz:")

        st.latex(r"E(t) = V_{emf}(t) + V_R(t)")

        st.markdown("Sabemos que a força contra-eletromotriz é proporcional à velocidade angular do motor, logo:")

        st.latex(r"V_{emf}(t) = k_2\, \omega(t)")

        st.markdown("Já a queda de tensão no resistor da armadura é dada pela Lei de Ohm:")

        st.latex(r"V_R(t) = R\, i(t)")

        st.markdown("Substituindo essas relações na Lei das Malhas, vem:")

        st.latex(r"E(t) = k_2\, \omega(t) + R\, i(t)")

        st.markdown("Isolando a corrente i(t), obtemos:")

        st.latex(r"i(t) = \frac{E(t) - k_2\, \omega(t)}{R}")

        st.markdown("Lembrando que a velocidade angular é a derivada do ângulo:")

        st.latex(r"\omega(t) = \frac{d\theta(t)}{dt}")

        st.markdown("Agora substituímos a expressão de i(t) na equação mecânica já obtida:")

        st.latex(r"J \frac{d^2 \theta(t)}{dt^2} = k_1\, i(t) - b\, \omega(t)")

        st.markdown("Ficando:")

        st.latex(r"J \frac{d^2 \theta(t)}{dt^2} = k_1 \left( \frac{E(t) - k_2\, \omega(t)}{R} \right) - b\, \omega(t)")

        st.markdown("Substituindo ω(t) pela derivada de θ(t):")

        st.latex(r"J \frac{d^2 \theta(t)}{dt^2} = k_1 \left( \frac{E(t) - k_2\, \frac{d\theta(t)}{dt}}{R} \right) - b\, \frac{d\theta(t)}{dt}")

        st.markdown("Distribuindo k1 e agrupando os termos em dθ(t)/dt, chegamos à equação resultante do movimento:")

        st.latex(
            r"J \frac{d^2 \theta(t)}{dt^2}"
            r" = \frac{k_1}{R} E(t)"
            r" - \left(\frac{k_1 k_2}{R} + b\right) \frac{d\theta(t)}{dt}"
        )
       
    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    st.latex(
        r"J \frac{d^2 \theta(t)}{dt^2}"
        r" = \frac{k_1}{R} E(t)"
        r" - \left(\frac{k_1 k_2}{R} + b\right) \frac{d\theta(t)}{dt}"
    )

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            # Constante de tempo e velocidade em regime permanente
            a = (ss.mdc_k1 * ss.mdc_k2) / ss.mdc_R + ss.mdc_b      # coeficiente da velocidade
            tau = ss.mdc_J / a                                     # constante de tempo mecânica
            omega_ss = (ss.mdc_k1 * ss.mdc_E0 / ss.mdc_R) / a      # velocidade em regime permanente

            # Vetor de tempo
            t = np.linspace(0, ss.mdc_tmax, 1000)

            with tab1:
                theta = (
                    ss.mdc_theta0
                    + omega_ss * t
                    + (ss.mdc_omega0 - omega_ss) * tau * (1 - np.exp(-t / tau))
                )

                fig_theta = px.line(
                    x=t,
                    y=theta,
                    labels={"x": "Tempo (s)", "y": "Posição angular θ(t) [rad]"},
                    title="Posição angular do motor DC ao longo do tempo"
                )

                st.plotly_chart(fig_theta, use_container_width=True)

            with tab2:
                omega = omega_ss + (ss.mdc_omega0 - omega_ss) * np.exp(-t / tau)

                fig_omega = px.line(
                    x=t,
                    y=omega,
                    labels={"x": "Tempo (s)", "y": "Velocidade angular ω(t) [rad/s]"},
                    title="Velocidade angular do motor DC ao longo do tempo"
                )

                st.plotly_chart(fig_omega, use_container_width=True)

            with tab3:
                alpha = (omega_ss - ss.mdc_omega0) / tau * np.exp(-t / tau)

                fig_alpha = px.line(
                    x=t,
                    y=alpha,
                    labels={"x": "Tempo (s)", "y": "Aceleração angular α(t) [rad/s²]"},
                    title="Aceleração angular do motor DC ao longo do tempo"
                )

                st.plotly_chart(fig_alpha, use_container_width=True)

            with tab4:
                omega = omega_ss + (ss.mdc_omega0 - omega_ss) * np.exp(-t / tau)

                i = (ss.mdc_E0 - ss.mdc_k2 * omega) / ss.mdc_R

                fig_i = px.line(
                    x=t,
                    y=i,
                    labels={"x": "Tempo (s)", "y": "Corrente i(t) [A]"},
                    title="Corrente elétrica do motor DC ao longo do tempo"
                )

                st.plotly_chart(fig_i, use_container_width=True)

            with tab5:
                omega = omega_ss + (ss.mdc_omega0 - omega_ss) * np.exp(-t / tau)

                i = (ss.mdc_E0 - ss.mdc_k2 * omega) / ss.mdc_R

                Tg = ss.mdc_k1 * i

                Tf = ss.mdc_b * omega

                Tr = Tg - Tf

                # Gráfico com Plotly + Streamlit
                fig_Tr = px.line(
                    x=t,
                    y=Tr,
                    labels={"x": "Tempo (s)", "y": "Torque resultante T_r(t) [N·m]"},
                    title="Torque resultante do motor DC ao longo do tempo"
                )

                st.plotly_chart(fig_Tr, use_container_width=True)      
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Posição angular θ(t)",
        "Velocidade angular ω(t)",
        "Aceleração angular α(t)",
        "Corrente i(t)",
        "Torque resultante Tₙ(t)"
    ])

    # Constante de tempo e velocidade em regime permanente
    a = (ss.mdc_k1 * ss.mdc_k2) / ss.mdc_R + ss.mdc_b      # coeficiente da velocidade
    tau = ss.mdc_J / a                                     # constante de tempo mecânica
    omega_ss = (ss.mdc_k1 * ss.mdc_E0 / ss.mdc_R) / a      # velocidade em regime permanente

    # Vetor de tempo
    t = np.linspace(0, ss.mdc_tmax, 1000)

    with tab1:
        theta = (
            ss.mdc_theta0
            + omega_ss * t
            + (ss.mdc_omega0 - omega_ss) * tau * (1 - np.exp(-t / tau))
        )

        fig_theta = px.line(
            x=t,
            y=theta,
            labels={"x": "Tempo (s)", "y": "Posição angular θ(t) [rad]"},
            title="Posição angular do motor DC ao longo do tempo"
        )

        st.plotly_chart(fig_theta, use_container_width=True)

    with tab2:
        omega = omega_ss + (ss.mdc_omega0 - omega_ss) * np.exp(-t / tau)

        fig_omega = px.line(
            x=t,
            y=omega,
            labels={"x": "Tempo (s)", "y": "Velocidade angular ω(t) [rad/s]"},
            title="Velocidade angular do motor DC ao longo do tempo"
        )

        st.plotly_chart(fig_omega, use_container_width=True)

    with tab3:
        alpha = (omega_ss - ss.mdc_omega0) / tau * np.exp(-t / tau)

        fig_alpha = px.line(
            x=t,
            y=alpha,
            labels={"x": "Tempo (s)", "y": "Aceleração angular α(t) [rad/s²]"},
            title="Aceleração angular do motor DC ao longo do tempo"
        )

        st.plotly_chart(fig_alpha, use_container_width=True)

    with tab4:
        omega = omega_ss + (ss.mdc_omega0 - omega_ss) * np.exp(-t / tau)

        i = (ss.mdc_E0 - ss.mdc_k2 * omega) / ss.mdc_R

        fig_i = px.line(
            x=t,
            y=i,
            labels={"x": "Tempo (s)", "y": "Corrente i(t) [A]"},
            title="Corrente elétrica do motor DC ao longo do tempo"
        )

        st.plotly_chart(fig_i, use_container_width=True)

    with tab5:
        omega = omega_ss + (ss.mdc_omega0 - omega_ss) * np.exp(-t / tau)

        i = (ss.mdc_E0 - ss.mdc_k2 * omega) / ss.mdc_R

        Tg = ss.mdc_k1 * i

        Tf = ss.mdc_b * omega

        Tr = Tg - Tf

        # Gráfico com Plotly + Streamlit
        fig_Tr = px.line(
            x=t,
            y=Tr,
            labels={"x": "Tempo (s)", "y": "Torque resultante T_r(t) [N·m]"},
            title="Torque resultante do motor DC ao longo do tempo"
        )

        st.plotly_chart(fig_Tr, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("pages/images/sistema_eletromecanico.png")

    with col2:

        col3, col4 = st.columns([1, 1]) 
        with col3:
            # Inércia J
            with st.container(border=True):
                st.markdown(f"J = {ss.mdc_J:.4f} kg·m²")
                J = st.number_input(
                    "",
                    #value=float(ss.mdc_J),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_J"
                )
                if J != ss.mdc_J:
                    ss.mdc_J = J
                    st.rerun()

            # k1
            with st.container(border=True):
                st.markdown(f"k₁ = {ss.mdc_k1:.4f}")
                k1 = st.number_input(
                    "",
                    #value=float(ss.mdc_k1),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_k1"
                )
                if k1 != ss.mdc_k1:
                    ss.mdc_k1 = k1
                    st.rerun()

            # k2
            with st.container(border=True):
                st.markdown(f"k₂ = {ss.mdc_k2:.4f}")
                k2 = st.number_input(
                    "",
                    #value=float(ss.mdc_k2),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_k2"
                )
                if k2 != ss.mdc_k2:
                    ss.mdc_k2 = k2
                    st.rerun()

            # θ0
            with st.container(border=True):
                st.markdown(f"θ₀ = {ss.mdc_theta0:.4f} rad")
                theta0 = st.number_input(
                    "",
                    #value=float(ss.mdc_theta0),
                    label_visibility="hidden",
                    key="mdc_theta0"
                )
                if theta0 != ss.mdc_theta0:
                    ss.mdc_theta0 = theta0
                    st.rerun()

        with col4:
            # Resistência R
            with st.container(border=True):
                st.markdown(f"R = {ss.mdc_R:.4f} Ω")
                R = st.number_input(
                    "",
                    #value=float(ss.mdc_R),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_R"
                )
                if R != ss.mdc_R:
                    ss.mdc_R = R
                    st.rerun()

            # Atrito viscoso b
            with st.container(border=True):
                st.markdown(f"b = {ss.mdc_b:.4f} N·m·s/rad")
                b = st.number_input(
                    "",
                    #value=float(ss.mdc_b),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_b"
                )
                if b != ss.mdc_b:
                    ss.mdc_b = b
                    st.rerun()

            # Tensão E0
            with st.container(border=True):
                st.markdown(f"E₀ = {ss.mdc_E0:.2f} V")
                E0 = st.number_input(
                    "",
                    #value=float(ss.mdc_E0),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_E0"
                )
                if E0 != ss.mdc_E0:
                    ss.mdc_E0 = E0
                    st.rerun()

            # Tempo máximo t_max
            with st.container(border=True):
                st.markdown(f"t_max = {ss.mdc_tmax:.2f} s")
                tmax = st.number_input(
                    "",
                    #value=float(ss.mdc_tmax),
                    label_visibility="hidden",
                    min_value=0.0,
                    key="mdc_tmax"
                )
                if tmax != ss.mdc_tmax:
                    ss.mdc_tmax = tmax
                    st.rerun()

            # ω0
            with st.container(border=True):
                st.markdown(f"ω₀ = {ss.mdc_omega0:.4f} rad/s")
                omega0 = st.number_input(
                    "",
                    #value=float(ss.mdc_omega0),
                    label_visibility="hidden",
                    key="mdc_omega0"
                )
                if omega0 != ss.mdc_omega0:
                    ss.mdc_omega0 = omega0
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicativos
    st.markdown("""
        O motor DC modelado neste sistema representa um dispositivo eletromecânico capaz de converter energia elétrica em movimento de rotação. 
        Sua dinâmica resulta da interação entre as grandezas elétricas e mecânicas, que se influenciam mutuamente durante o funcionamento. 

        No lado elétrico, a tensão aplicada ao motor gera uma corrente na armadura, e essa corrente é responsável por produzir torque por meio da interação entre o campo magnético e os enrolamentos. 
        Esse torque acelerador, proporcional à corrente, constitui a força motriz que coloca o eixo em movimento. 
        Ao mesmo tempo, conforme o rotor gira, surge uma força contra-eletromotriz proporcional à velocidade angular, que se opõe à tensão aplicada e limita a corrente à medida que o motor se aproxima do regime permanente.

        No lado mecânico, o eixo do motor possui inércia, o que faz com que mudanças na velocidade não ocorram instantaneamente. 
        A dinâmica de rotação é descrita por uma equação diferencial que relaciona a aceleração angular ao torque líquido disponível, que é a diferença entre o torque gerado e o torque resistivo devido ao atrito viscoso. 
        Esse atrito cresce com a velocidade e atua dissipando energia, estabilizando o sistema à medida que o motor se aproxima de um valor final de rotação.

        O conjunto dessas interações resulta em um sistema de primeira ordem para a velocidade angular, que apresenta comportamento típico de resposta transitória seguido de regime permanente. 
        Após a aplicação de um degrau de tensão, o motor inicialmente acelera rapidamente, pois a corrente é máxima; conforme a velocidade aumenta, a força contra-eletromotriz reduz a corrente e, consequentemente, o torque gerado. 
        O atrito viscoso completa o equilíbrio, fazendo com que a velocidade se estabilize em um valor finito determinado pelos parâmetros físicos do motor.

        A posição angular surge como a integral da velocidade, formando um sistema de segunda ordem quando observada diretamente ao longo do tempo. 
        Mudanças no ângulo inicial ou na velocidade inicial afetam apenas o comportamento transitório, enquanto o regime permanente depende exclusivamente das características elétricas e mecânicas do motor.

        A resposta total do sistema é, portanto, fruto da combinação entre os efeitos eletromagnéticos que produzem torque, a resistência decorrente das perdas mecânicas e a própria inércia do rotor. 
        Esse equilíbrio define como o motor acelera, desacelera e converge para sua velocidade final, ilustrando de forma clara o acoplamento entre energia elétrica e movimento mecânico em sistemas dinâmicos de conversão.
    """)

    space_line()

    # Sugestões
    st.markdown("""
        Ao alterar os parâmetros do motor DC, os gráficos de θ(t), ω(t) e das demais grandezas associadas respondem de forma coerente com a física do sistema, 
        mostrando como cada termo da equação influencia diretamente a dinâmica eletromecânica:

        Momento de inércia (J): ao aumentar J, o eixo do motor fica "mais pesado" para acelerar. Isso alonga a constante de tempo mecânica, faz com que a velocidade 
        demore mais para atingir o regime permanente e torna a curva de θ(t) mais suave e gradual. Valores menores de J deixam o sistema mais "esperto", 
        com aceleração mais intensa e resposta mais rápida.

        Constante de torque (k1): valores maiores de k1 aumentam o torque gerado para a mesma corrente, o que intensifica a aceleração inicial e eleva a velocidade 
        em regime permanente. O motor responde de forma mais vigorosa ao degrau de tensão, enquanto valores menores de k1 resultam em um motor mais "fraco", 
        com menor torque e menor ω(t) final.

        Constante de f.c.e.m. (k2): ao aumentar k2, a força contra eletromotriz cresce mais rapidamente com a velocidade, limitando a corrente e reduzindo a velocidade 
        em regime permanente. O motor tende a atingir um ω(t) final menor, com uma ação de "autolimitação" mais forte. Valores menores de k2 permitem velocidades finais 
        maiores, já que a f.c.e.m. se opõe menos à tensão aplicada.

        Resistência da armadura (R): resistências maiores reduzem a corrente para uma mesma tensão de entrada, o que diminui o torque gerado e a velocidade em regime 
        permanente. O motor arranca com menor corrente de partida e responde de forma mais lenta. Resistências menores permitem correntes mais altas, maior torque inicial 
        e maior ω(t), porém à custa de maior consumo elétrico.

        Coeficiente de atrito viscoso (b): quanto maior b, maior o torque resistivo proporcional à velocidade, o que reduz a velocidade em regime permanente e faz o motor 
        "frear" mais rapidamente. O sistema tende a atingir o regime estacionário mais cedo, porém com uma ω(t) final menor. Valores pequenos de b deixam o motor mais 
        "livre", com menos perdas mecânicas e maior velocidade final.

        Tensão de entrada (E0): aumentar E0 eleva a corrente disponível na partida, aumentando o torque gerado e a velocidade em regime permanente. Os gráficos de ω(t) 
        e θ(t) mostram rampas mais inclinadas e um patamar final mais alto. Reduzir E0 faz o motor responder de forma mais tímida, com menor aceleração e menor valor 
        final de velocidade.

        Tempo máximo (t_max): esse parâmetro não altera o comportamento físico do motor, apenas o intervalo exibido nos gráficos. Aumentar t_max permite visualizar toda 
        a trajetória até o regime permanente, enquanto reduzir t_max limita a análise ao trecho inicial da resposta transitória.

        Ângulo inicial (θ₀): ao ajustar θ₀, o gráfico de posição angular θ(t) é deslocado verticalmente, preservando o formato da curva. A dinâmica de velocidade não muda, 
        apenas o ponto de partida da posição, como se escolhêssemos um novo zero de referência para o eixo.

        Velocidade angular inicial (ω₀): ao definir uma velocidade inicial diferente de zero, o motor pode começar acima ou abaixo da velocidade de regime. 
        Se ω₀ for maior que a velocidade em regime, a curva de ω(t) decai até o valor final, se ω₀ for menor, a velocidade cresce até esse patamar. 
        Isso altera o formato transitório de θ(t) e ω(t), mas o ponto de chegada continua determinado pelos parâmetros físicos do sistema e por E0.
    """)