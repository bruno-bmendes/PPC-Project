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
        ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola Amortecedor", "Pêndulo Simples Amortecido", "Sistema Eletromecanico", "Tanque com Aquecimento", "Motor Bomba de um Poço BCS"]

    ss.page = "pendulo_simples_amortecido"
    st.rerun()

# Definindo página
def pendulo_simples_amortecido():

    # Declarando Variáveis
    if "ps_g" not in ss:
        ss.ps_g = 9.81
    if "ps_l" not in ss:
        ss.ps_l = 1.0
    if "ps_theta0" not in ss:
        ss.ps_theta0 = 0.5
    if "ps_omega0" not in ss:
        ss.ps_omega0 = 0.0
    if "ps_tmax" not in ss:
        ss.ps_tmax = 10.0

    # Definir Título
    ss.title = "Pêndulo Simples Amortecido (Linear e Não Linear)"
    
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
    @st.dialog("Desenvolvendo a Equação", width="large")
    def info():

        st.markdown("""
            Podemos declarar inicialmente as seguintes premissas:

                - A massa 𝑚 está concentrada na ponta da haste, ou seja, trataremos como uma massa puntiforme

                - A haste é rígida, não alonga e não deforma

                - O movimento é em um único plano, então basta um ângulo 𝜃 para descrever o sistema

                - O comprimento 𝐿, a massa 𝑚, a gravidade 𝑔, o amortecimento 𝑘 e o momento de inércia 𝐽 são constantes

                - O atrito que vamos considerar está concentrado no pivô e é proporcional à velocidade angular, por isso usamos um coeficiente viscoso 𝑘

                - Não há outra força externa empurrando o pêndulo, só gravidade e amortecimento
        """)

        st.markdown("""
            Para esse sistema consideraremos o método de Lagrange, e para isso devemos determinar a energia cinética K e a energia potencial P:
        """)

        st.latex(r"\mathcal{L} = K - P")

        st.markdown("""
            Inicialmente calculamos K. Para isso, precisamos primeiro calcular a posição da massa. Se a haste tem comprimento 𝐿, as coordenadas da massa são:
        """)

        st.latex(r"""
            \begin{aligned}
            x &= L \sin\theta \\
            y &= -L \cos\theta
            \end{aligned}
        """)

        st.markdown("""
            Para obter a velocidade, derivamos x e y no tempo:            
        """)

        st.latex(r"""
            \begin{aligned}
            \dot{x} &= \frac{dx}{dt} = L \cos\theta \, \dot{\theta} \\
            \dot{y} &= \frac{dy}{dt} = L \sin\theta \, \dot{\theta}
            \end{aligned}
        """)


        st.markdown("""
            E a velocidade escalar:            
        """)

        st.latex(r"v^2 = \dot{x}^2 + \dot{y}^2")

        st.markdown("""
            Substituindo:
        """)

        st.latex(r"""
            \begin{aligned}
            v^2 &= (L \cos\theta \, \dot{\theta})^2 + (L \sin\theta \, \dot{\theta})^2 \\
            v^2 &= L^2 \dot{\theta}^2 (\cos^2\theta + \sin^2\theta)
            \end{aligned}
        """)

        st.markdown("""
            Sabemos que cos²𝜃 + sin²𝜃 = 1, então:            
        """)

        st.latex(r"""
            \begin{aligned}
            v^2 &= L^2 \dot{\theta}^2 \\
            v &= L \dot{\theta}
            \end{aligned}
        """)

        st.markdown("Determinamos primeiro a energia cinética da massa Kt:")

        st.latex(r"""
            K_t = \frac{1}{2} m v^2 
                = \frac{1}{2} m (L \dot{\theta})^2
                = \frac{1}{2} m L^2 \dot{\theta}^2
        """)

        st.markdown("""
            Se o pivô tem momento de inércia 𝐽, sua energia cinética de rotação é:            
        """)

        st.latex(r"K_r = \frac{1}{2} J \dot{\theta}^2")

        st.markdown("Somamos as duas para obter a energia cinética total:")

        st.latex(r"K = K_t + K_r = \frac{1}{2}(J + mL^2)\dot{\theta}^2")

        st.markdown("""
            Em sequencia vamos calcular a energia potencial gerada pela gravidade. A energia potencial gravitacional é 𝑃 = 𝑚𝑔ℎ, onde ℎ é a altura do ponto de massa.            

            Vamos escolher como referência de energia potencial zero a posição de equilíbrio, quando o pêndulo está para baixo, com 𝜃 = 0.

            Quando o pêndulo faz um ângulo 𝜃, a massa sobe uma certa altura em relação à posição de referência.

            A coordenada vertical é 𝑦 = −𝐿cos𝜃. Na posição de equilíbrio, 𝜃 = 0 e 𝑦0 = −𝐿.

            O aumento de altura em relação ao equilíbrio é:
        """)

        st.latex(r"""
            h(\theta) = y_0 - y 
            = (-L) - (-L \cos\theta) 
            = -L + L\cos\theta 
            = L(\cos\theta - 1)
        """)

        st.markdown("Se preferirmos escrever como um valor positivo quando sobe:")

        st.latex(r"h(\theta) = L(1 - \cos\theta)")

        st.markdown("A energia potencial então é:")

        st.latex(r"P = m g h(\theta) = m g L (1 - \cos\theta)")

        st.markdown("""
            Note que:
            Se 𝜃 = 0 então 𝑃 = 0
            Se 𝜃 afasta do zero, cos𝜃 diminui e 𝑃 aumenta            
        """)

        st.markdown("Juntando K e P teremos a função central do Método de Lagrange:")

        st.latex(r"""
            \mathcal{L}(\theta, \dot{\theta}) = K - P
        """)

        st.latex(r"""
            \mathcal{L} = \frac{1}{2}(J + mL^2)\dot{\theta}^2 - mgL(1 - \cos\theta)
        """)

        st.markdown("O amortecimento é viscoso, ou seja, o torque de atrito é proporcional à velocidade angular e contrário ao movimento:")

        st.latex(r"\tau_{am} = -k\,\dot{\theta}")

        st.markdown("Na formulação de Lagrange, essa torque entra como força generalizada associada à coordenada 𝜃:")

        st.latex(r"Q = \tau_{am} = -k\,\dot{\theta}")

        st.markdown("""
            Então na equação de Lagrange o lado direito será −𝑘𝜃˙.
                    
            A equação de Lagrange para a coordenada generalizada 𝜃 é:
        """)

        st.latex(r"""
            \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\theta}}\right)
            - \frac{\partial \mathcal{L}}{\partial \theta}
            = Q
        """)
        
        st.markdown("""
            com 𝑄 = −𝑘𝜃˙.

            Vamos calcular cada pedaço da equação, começando por ∂𝐿/∂𝜃˙.
        """)

        st.latex(r"\mathcal{L} = \frac{1}{2}(J + mL^2)\dot{\theta}^2 - mgL(1 - \cos\theta)")

        st.markdown("""
            A parte que depende de 𝜃˙ é só o primeiro termo. 
            
            Derivando em relação a 𝜃˙:            
        """)

        st.latex(r"\frac{\partial \mathcal{L}}{\partial \dot{\theta}} = (J + mL^2)\dot{\theta}")

        st.markdown("Em sequência, a derivada temporal de ∂𝐿/∂𝜃˙:")

        st.latex(r"""
            \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\theta}}\right)
            = \frac{d}{dt}\left((J + mL^2)\dot{\theta}\right)
        """)

        st.markdown("Como 𝐽 e 𝐿 são constantes:")

        st.latex(r"""
            \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\theta}}\right)
            = (J + mL^2)\,\ddot{\theta}
        """)

        st.markdown("""
            Esse termo representa a parte de inércia do sistema.

            Por fim calculamos a parte de ∂𝐿/∂𝜃:
        """)

        st.latex(r"""
            \frac{\partial \mathcal{L}}{\partial \theta}
            = -mgL \frac{d}{d\theta}(1 - \cos\theta)
        """)

        st.latex(r"""
            \frac{d}{d\theta}(1 - \cos\theta)
            = 0 - (-\sin\theta) = \sin\theta
        """)

        st.latex(r"""
            \frac{\partial \mathcal{L}}{\partial \theta}
            = -mgL \sin\theta
        """)

        st.markdown("Substituimos portanto os termos na equação de Lagrange:")

        st.latex(r"""
            \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\theta}}\right)
            - \frac{\partial \mathcal{L}}{\partial \theta}
            = Q
        """)

        st.latex(r"""
            (J + mL^2)\ddot{\theta} - (-mgL\sin\theta)
            = -k\,\dot{\theta}
        """)

        st.markdown("""
            Dessa forma, chegamos na equação não linear do pêndulo simples amortecido:        
        """)

        st.latex(r"(J + mL^2)\ddot{\theta} + k\,\dot{\theta} + mgL\sin\theta = 0")

        st.markdown("Já que estamos considerando o Pêndulo Ideal, podemos desconsiderar o Momento de Inércia (J) e o Coeficiente de Amortecimento Viscoso (k), já que não consideramos a resistência do ar, nem o atrito no pivô, e portanto não há nenhum mecanismo dissipativo no sistema. Dessa forma, a equação para o sistema que trabalhamos seria:")

        st.latex(r"mL^2\ddot{\theta} + mgL\sin\theta = 0")

        st.markdown("Ou dividindo tudo por mL²:")

        st.latex(r"\ddot{\theta} + \frac{g}{L}\sin\theta = 0")

        st.markdown("Se quisermos linearizar o sistema, para situações onde o ângulo 𝜃 é pequeno, podemos aproximar sin𝜃 = 𝜃, e portanto:")
        
        st.latex(r"\ddot{\theta} + \frac{g}{L}\theta = 0")

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    st.latex(r"\ddot{\theta} + \frac{g}{L}\sin\theta = 0")

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            # Intervalo de simulação
            t_span = (0, 10)
            t_eval = np.linspace(t_span[0], t_span[1], 2000)

            def pendulo_nao_linear(t, y):
                theta, omega = y
                dtheta = omega
                domega = -(g / L) * np.sin(theta)
                return [dtheta, domega]
            
            def pendulo_linear(t, y):
                theta, omega = y
                dtheta = omega
                domega = -(g / L) * theta
                return [dtheta, domega]
            
            sol_nl = solve_ivp(pendulo_nao_linear, t_span, [theta0, omega0], t_eval=t_eval)
            sol_lin = solve_ivp(pendulo_linear, t_span, [theta0, omega0], t_eval=t_eval)

            tab1, tab2 = st.tabs(["Equação Não Linear", "Equação Linearizada"])

            # =========================================================
            # TABELA 1: NÃO LINEAR
            # =========================================================
            with tab1:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("### θ(t)  rad  –  Não Linear")

                    fig_theta_nl = go.Figure()
                    fig_theta_nl.add_trace(go.Scatter(
                        x=sol_nl.t,
                        y=sol_nl.y[0],
                        mode="lines",
                        name="θ(t)"
                    ))
                    fig_theta_nl.update_layout(
                        xaxis_title="Tempo (s)",
                        yaxis_title="θ(t) rad",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_theta_nl, use_container_width=True)

                with col2:
                    st.write("### ω(t)  rad/s  –  Não Linear")

                    fig_omega_nl = go.Figure()
                    fig_omega_nl.add_trace(go.Scatter(
                        x=sol_nl.t,
                        y=sol_nl.y[1],
                        mode="lines",
                        name="ω(t)"
                    ))
                    fig_omega_nl.update_layout(
                        xaxis_title="Tempo (s)",
                        yaxis_title="ω(t) rad/s",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_omega_nl, use_container_width=True)


            # =========================================================
            # TABELA 2: LINEARIZADA
            # =========================================================
            with tab2:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.write("### θ(t)  rad  –  Linearizado")

                    fig_theta_lin = go.Figure()
                    fig_theta_lin.add_trace(go.Scatter(
                        x=sol_lin.t,
                        y=sol_lin.y[0],
                        mode="lines",
                        name="θ(t)"
                    ))
                    fig_theta_lin.update_layout(
                        xaxis_title="Tempo (s)",
                        yaxis_title="θ(t) rad",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_theta_lin, use_container_width=True)

                with col2:
                    st.write("### ω(t)  rad/s  –  Linearizado")

                    fig_omega_lin = go.Figure()
                    fig_omega_lin.add_trace(go.Scatter(
                        x=sol_lin.t,
                        y=sol_lin.y[1],
                        mode="lines",
                        name="ω(t)"
                    ))
                    fig_omega_lin.update_layout(
                        xaxis_title="Tempo (s)",
                        yaxis_title="ω(t) rad/s",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_omega_lin, use_container_width=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    # Intervalo de simulação
    t_span = (0, ss.ps_tmax)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)

    def pendulo_nao_linear(t, y):
        theta, omega = y
        dtheta = omega
        domega = -(ss.ps_g / ss.ps_l) * np.sin(theta)
        return [dtheta, domega]
    
    def pendulo_linear(t, y):
        theta, omega = y
        dtheta = omega
        domega = -(ss.ps_g / ss.ps_l) * theta
        return [dtheta, domega]
    
    sol_nl = solve_ivp(pendulo_nao_linear, t_span, [ss.ps_theta0, ss.ps_omega0], t_eval=t_eval)
    sol_lin = solve_ivp(pendulo_linear, t_span, [ss.ps_theta0, ss.ps_omega0], t_eval=t_eval)

    col1, col2 = st.columns([1, 1])

    # =========================================================
    # COLUNA 1: θ(t) – Linear vs Não Linear
    # =========================================================
    with col1:
        st.write("### θ(t) – Linear vs Não Linear")

        fig_theta = go.Figure()

        fig_theta.add_trace(go.Scatter(
            x=sol_nl.t,
            y=sol_nl.y[0],
            mode="lines",
            name="θ(t) – Não Linear"
        ))

        fig_theta.add_trace(go.Scatter(
            x=sol_lin.t,
            y=sol_lin.y[0],
            mode="lines",
            name="θ(t) – Linearizado"
        ))

        fig_theta.update_layout(
            xaxis_title="Tempo (s)",
            yaxis_title="θ(t) rad",
            template="plotly_white"
        )

        st.plotly_chart(fig_theta, use_container_width=True)


    # =========================================================
    # COLUNA 2: ω(t) – Linear vs Não Linear
    # =========================================================
    with col2:
        st.write("### ω(t) – Linear vs Não Linear")

        fig_omega = go.Figure()

        fig_omega.add_trace(go.Scatter(
            x=sol_nl.t,
            y=sol_nl.y[1],
            mode="lines",
            name="ω(t) – Não Linear"
        ))

        fig_omega.add_trace(go.Scatter(
            x=sol_lin.t,
            y=sol_lin.y[1],
            mode="lines",
            name="ω(t) – Linearizado"
        ))

        fig_omega.update_layout(
            xaxis_title="Tempo (s)",
            yaxis_title="ω(t) rad/s",
            template="plotly_white"
        )

        st.plotly_chart(fig_omega, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("pages/images/pendulo_simples.png")

    with col2:
        col3, col4 = st.columns([1, 1])
        with col3:
            # Comprimento
            with st.container(border=True):
                st.markdown(f"L = {ss.ps_l:.2f} m")
                l = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ps_l"
                )
                if l != ss.ps_l:
                    ss.ps_l = l
                    st.rerun() 

            with st.container(border=True):
                # Entrada para theta0
                st.markdown(f"θ₀ = {ss.ps_theta0:.2f} rad")
                theta0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ps_theta0"
                )
                if theta0 != ss.ps_theta0:
                    ss.ps_theta0 = theta0
                    st.rerun()

            with st.container(border=True):
                st.markdown(f"g​ = {ss.ps_g:.2f} m/s²")

        with col4:
            with st.container(border=True):
                st.markdown(f"ω₀ = {ss.ps_omega0:.2f} rad/s")
                omega0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ps_omega0"
                )
                if omega0 != ss.ps_omega0:
                    ss.ps_omega0 = omega0
                    st.rerun()

            # Tempo Máximo
            with st.container(border=True):
                st.markdown(f"t_max = {ss.ps_tmax:.2f} s")
                tmax = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ps_tmax"
                )
                if tmax != ss.ps_tmax:
                    ss.ps_tmax = tmax
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicativos
    st.markdown("""
    O pêndulo simples amortecido representa um sistema dinâmico de segunda ordem, composto por uma massa concentrada presa a uma haste de comprimento fixo. Ele é capaz de armazenar energia de duas formas:

        A componente gravitacional, que fornece um torque restaurador proporcional ao seno do ângulo de deslocamento,
        e a energia cinética associada ao movimento angular da massa.

    O comportamento do pêndulo é descrito por uma equação diferencial de segunda ordem, que relaciona o ângulo, sua velocidade angular e o efeito do amortecimento. 
    Essa é uma EDO não linear, pois o termo responsável pela força restauradora depende de sin(θ), o que introduz um comportamento mais completo em amplitudes maiores. 
    Para deslocamentos pequenos, a aproximação sin(θ) ≈ θ leva a um modelo linearizado que simplifica a análise e preserva as principais características do movimento.

    Quando o pêndulo é afastado da vertical e liberado, ele inicia um movimento transitório, em que parte da energia é convertida entre cinética e potencial gravitacional, enquanto outra parte é perdida devido ao amortecimento. 
    O amortecedor atua resistindo ao movimento, reduzindo gradualmente a amplitude das oscilações até que a massa pare completamente na posição de equilíbrio.

    A forma como a energia se distribui entre o torque gravitacional, a inércia da massa e o amortecimento define o padrão de resposta do sistema ao longo do tempo. 
    Para pequenas amplitudes, o modelo linearizado apresenta oscilações harmônicas com redução progressiva, enquanto o modelo completo, não linear, revela variações mais ricas quando o deslocamento inicial é significativo, já que o seno do ângulo altera a taxa de retorno à posição de equilíbrio.

    À medida que o sistema perde energia, as oscilações vão diminuindo e o pêndulo converge para a posição vertical. 
    A interação entre gravidade, movimento angular e dissipação determina como o sistema responde à perturbação inicial, compondo uma dinâmica que combina elementos restauradores e dissipativos até atingir o repouso final.
    """)

    space_line()

    # Sugestões
    st.markdown("""
        Ao alterar os parâmetros do pêndulo simples amortecido, o gráfico de θ(t) e ω(t) responde de forma coerente com a física do sistema, 
        revelando como cada termo influencia diretamente o movimento angular:

        Comprimento da haste (L): ao aumentar L, o pêndulo se torna mais lento, pois a frequência natural diminui. 
        Um pêndulo mais longo oscila de forma mais suave e com períodos maiores, enquanto reduzir L torna o movimento mais rápido.

        Gravidade (g): valores maiores de g intensificam o torque restaurador, aumentando a velocidade com que o pêndulo retorna à posição de equilíbrio. 
        Isso eleva a frequência natural do movimento. Reduzir g diminui essa força restauradora e torna as oscilações mais lentas.

        Coeficiente de amortecimento (c): quanto maior o amortecimento, mais energia é dissipada a cada ciclo, diminuindo a amplitude das oscilações. 
        Aumentar c faz o pêndulo perder velocidade angular mais rapidamente, reduzindo a oscilação visível no gráfico. 
        Já valores menores de c permitem oscilações mais longas antes de o sistema retornar ao repouso.

        Tempo máximo (t_max): esse parâmetro não altera o comportamento físico do pêndulo, mas apenas o intervalo exibido no gráfico. 
        Aumentar t_max permite visualizar toda a evolução do movimento até a parada completa, enquanto reduzir t_max limita a análise ao trecho inicial 
        do deslocamento angular.
    """)