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
        ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola Amortecedor", "Pêndulo Simples Amortecido", "Sistema Eletromecanico", "Tanque com Aquecimento", "Motor Bomba de um Poço BCS"]

    ss.page = "motor_bomba_de_um_poco_bcs"
    st.rerun()

# Definindo página
def motor_bomba_de_um_poco_bcs():

    # Tempo de simulação
    if "bcs_tmax" not in ss:
        ss.bcs_tmax = 30.0

    # Parâmetros mecânicos/elétricos/térmicos
    if "bcs_J" not in ss:
        ss.bcs_J = 0.01      # inércia [kg*m^2]
    if "bcs_b" not in ss:
        ss.bcs_b = 0.1       # atrito viscoso [N*m*s]
    if "bcs_k1" not in ss:
        ss.bcs_k1 = 0.05     # constante de torque [N*m/A]
    if "bcs_L" not in ss:
        ss.bcs_L = 0.5       # indutância [H]
    if "bcs_R0" not in ss:
        ss.bcs_R0 = 2.0      # resistência em T0 [ohm]
    if "bcs_alpha" not in ss:
        ss.bcs_alpha = 0.004 # coef. temp. resistência [1/K]
    if "bcs_k2" not in ss:
        ss.bcs_k2 = 0.05     # constante FEM (back-emf) [V*s/rad]
    if "bcs_Ct" not in ss:
        ss.bcs_Ct = 10.0     # capacitância térmica [J/K]
    if "bcs_hA" not in ss:
        ss.bcs_hA = 1.5      # h*A [W/K]
    if "bcs_T_inf" not in ss:
        ss.bcs_T_inf = 293.15 # ambiente [K]

    # Parâmetros do sinal E(t)
    if "bcs_E_min" not in ss:
        ss.bcs_E_min = 10.0  # [V]
    if "bcs_E_max" not in ss:
        ss.bcs_E_max = 14.0  # [V]
    if "bcs_dt_E" not in ss:
        ss.bcs_dt_E = 0.2    # a cada dt_E s sorteia um novo valor

    # Condições iniciais
    if "bcs_w0" not in ss:
        ss.bcs_w0 = 0.0
    if "bcs_i0" not in ss:
        ss.bcs_i0 = 0.0

    # Definir Título
    ss.title = "Motor Bomba de um Poço BCS"
    
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
            - Os materiais não sofrem desgaste, mantendo suas propriedades constantes ao longo do tempo;
            - O torque resistivo é considerado apenas como torque viscoso, proporcional à velocidade angular;
            - A geração de calor do sistema é limitada às perdas ôhmicas no resistor;
            - A resistência elétrica varia linearmente com a temperatura no intervalo de operação analisado;
            - A indutância do motor é constante e independe da corrente e da temperatura;
            - As constantes eletromecânicas do motor são consideradas constantes, não havendo efeitos de saturação magnética;
            - O sistema mecânico é modelado com um único grau de liberdade rotacional, representando motor e bomba como um conjunto rígido;
            - A capacidade térmica do motor é considerada concentrada, admitindo uma única temperatura representativa do sistema;
            - A troca térmica com o ambiente ocorre apenas por convecção, com coeficiente de convecção e área de troca constantes;
            - A temperatura do ambiente é considerada constante ao longo do tempo;
            - A temperatura inicial do sistema é igual a temperatura ambiente.
        """)

        st.markdown("""
            O sistema é um motor dc que alimenta a bomba de um poço de petróleo. Ele funciona nas seguintes etapas:

            1. Uma fonte externa fornece uma tensão para o motor, gerando uma corrente elétrica, que por sua vez, gera um campo eletromagnético;

            2. A força eletromagnética promove um torque, e consequentemente, a rotação no rotor (armadura) do motor, que está mecanicamente acoplado à bomba, fazendo ela funcionar;

            3. A geração de energia, e a energia dissipada para o ambiente promovem uma variação térmica no sistema. Essa variação de temperatura muda a resistência causando uma alteração no próprio comportamento elétrico do sistema. A resistência muda de acordo com a equação:            
        """)

        st.latex(
            r"R(T) = R_0 \left[ 1 + \alpha \left( T - T_0 \right) \right]"
        )

        st.markdown("""
            Em que:  
                R é a resistência;  
                R0 é a resistência em uma temperatura de referência;  
                α é o coeficiente térmico do material;  
                T é a temperatura;  
                T0 é a temperatura de referência.  

            Essa relação sai da Dependência da Resistividade Elétrica:            
        """)

        st.latex(
            r"R = \rho \frac{L}{A}"
        )

        st.markdown("""
            onde:
                ρ é a resistividade do material,
                L é o comprimento do fio,
                A é a área da seção transversal.

            Nesse caso L e A são praticamente constantes, e o que muda com a temperatura é a resistividade ρ.

            Para metais condutores, a resistividade vai ser dada por:            
        """)

        st.latex(
            r"\rho(T) = \rho_0 \left[ 1 + \alpha \left( T - T_0 \right) \right]"
        )
    
        st.markdown("proporcionando a relação citada:")

        st.latex(
            r"R(T) = R_0 \left[ 1 + \alpha \left( T - T_0 \right) \right]"
        )

        st.markdown("""
            A primeira EDO que vamos desenvolver é a elétrica. Ela parte da Lei das Malhas, que vai mostrar que:
        """)

        st.latex(
            r"E(t) = V_R(t) + V_L(t) + V_{emf}(t)"
        )

        st.markdown("""
            Sendo:
                E a tensão de entrada;
                VR é a tensão do resistor;
                VL é a tensão do indutor;
                Vemf é a força contraeletromotriz.

            Cada uma dessas variáveis pode ser destrinchada:  
        """)

        st.latex(
            r"V_R(t) = R\, i(t)"
        )

        st.markdown("""
            onde:
                R é resistência elétrica;
                i(t) é a corrente elétrica.
        """)

        st.latex(
            r"V_L(t) = L\,\frac{di(t)}{dt}"
        )

        st.markdown("""
            onde:
                L é a indutância            
        """)

        st.latex(
            r"V_{emf}(t) = k_e\,\omega(t)"
        )

        st.markdown("""
            onde:
                ke é a constante eletromecânica do motor;
                ω(t) é a velocidade angular do eixo do motor.

            Dessa forma ficamos com a equação:           
        """)

        st.latex(
            r"E(t) = R\,i(t) + L\,\frac{di(t)}{dt} + k_e\,\omega(t)"
        )

        st.markdown("ou isolando a derivada:")

        st.latex(
            r"L\,\frac{di(t)}{dt} = E(t) - R\,i(t) - k_e\,\omega(t)"
        )

        st.markdown("como definimos que a Resistência varia com a Temperatura, vamos substituir o R da equação pelo termo calculado anteriormente.")

        st.latex(
            r"L\,\frac{di(t)}{dt} = E(t) - R_0 \left[ 1 + \alpha \left( T - T_0 \right) \right] i(t) - k_e\,\omega(t)"
        )

        st.markdown("""
            A  segunda EDO que vamos calcular é a mecânica. Ela vai partir da Segunda Lei de Newton para a Rotação. 

            Do mesmo jeito que temos para sistemas lineares a base:            
        """)

        st.latex(
            r"\sum F = m a"
        )

        st.markdown("para sistemas rotacionais vamos ter a relação:")

        st.latex(
            r"\sum T = J\,\alpha"
        )

        st.markdown("""
            onde:
                T vai ser o torque (somatória dos torques ou torque resultante);
                J é o momento de inércia;
                α é a aceleração angular

            Como a aceleração angular pode ser descrita pela derivada da velocidade angular no tempo, podemos reescrever a relação como:            
        """)

        st.latex(
            r"\sum T = J\,\frac{d\omega(t)}{dt}"
        )

        st.markdown("Nesse caso temos dois torques a serem considerados, o torque gerado pelo motor, e o torque de resistencia, como uma espécie de atrito:")

        st.latex(
            r"\sum T = T_m(t) - T_r(t)"
        )

        st.markdown("Nesse caso temos que o torque do motor será:")

        st.latex(
            r"T_m = k_t\,i(t)"
        )

        st.markdown("""
            onde:
                kt​ é a constante de torque do motor (dada de acordo com o material).

            E o torque resistivo será:            
        """)

        st.latex(
            r"T_r(t) = b\,\omega(t)"
        )

        st.markdown("""
            onde:
                b é o coeficiente de atrito viscoso

            Nesse caso chegamos na equação:            
        """)

        st.latex(
            r"J\,\frac{d\omega(t)}{dt} = k_t\,i(t) - b\,\omega(t)"
        )

        st.markdown("A última EDO é a térmica. Vamos partir da premissa do balanço de energia térmica, falando que:")

        st.latex(
            r"\text{Acúmulo} = \text{Geração} - \text{Perda}"
        )

        st.markdown("O acúmulo é basicamente quanto calor está sendo guardado no motor ao longo do tempo, então pode ser descrito por sua Capacidade Térmica (Ct) multiplicada pela Taxa de Variação de Temperatura:")

        st.latex(
            r"\text{Acúmulo} = C_t\,\frac{dT}{dt}"
        )

        st.markdown("""
            A geração de calor é explicada pela Lei de Joule, que diz que:

            “Quando uma corrente elétrica passa por um condutor com resistência, parte da energia elétrica é transformada em calor.”

            Ela fala basicamente que o calor gerado é a Potência Dissipada, a Potência Elétrica é no geral:            
        """)

        st.latex(
            r"P = V\,i"
        )

        st.markdown("Em que nesse caso, a tensçao é a tensão de resistencia, e portanto:")

        st.latex(
            r"P_{diss} = V_R\,i = (R\,i)\,i = i^2 R"
        )

        st.markdown("Sendo assim, a geração de calor será dada por:")

        st.latex(
            r"i^2\,R(T)"
        )

        st.markdown("E como a resistencia varia com a temperatura:")

        st.latex(
            r"i^2(t)\,R_0 \left[ 1 + \alpha \left( T(t) - T_0 \right) \right]"
        )

        st.markdown("""
            A perda que é o último termo da EDO, será modelado levando em consideração a Lei de Resfriamento de Newton para convecção, que vai falar que o ambiente vai roubar calor do sistema de acordo com:

            Um Coeficiente de Convecção (h) que vai medir o quão eficiente o ambiente é para “roubar calor” do motor;
            A Área de Contato (A), porque quanto maior a área de contato com o ar, mais “superfície” existe para o calor escapar, então a perda aumenta;
            E a Diferença de Temperatura do Sistema com o Ambiente (T−T∞​), porque quanto maior a diferença de temperatura entre o corpo e o ambiente, mais rápido ele perde calor.

            Dessa forma temos o termo:            
        """)

        st.latex(
            r"hA\,(T - T_\infty)"
        )

        st.markdown("E aplicando os três na condição inicial:")

        st.latex(
            r"C_t\,\frac{dT}{dt} = i^2 R_0 \left[ 1 + \alpha (T - T_0) \right] - hA (T - T_\infty)"
        )

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("EDO Elétrica:")
        st.latex(
            r"L\,\frac{di(t)}{dt} = E(t) - R_0 \left[ 1 + \alpha \left( T - T_0 \right) \right] i(t) - k_e\,\omega(t)"
        )

    with col2:
        st.markdown("EDO Mecânica:")
        st.latex(
            r"J\,\frac{d\omega(t)}{dt} = k_t\,i(t) - b\,\omega(t)"
        )

    with col3:
        st.markdown("EDO Térmica:")
        st.latex(
            r"C_t\,\frac{dT}{dt} = i^2 R_0 \left[ 1 + \alpha (T - T_0) \right] - hA (T - T_\infty)"
        )

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
            t_eval = np.linspace(0.0, ss.bcs_tmax, 2000)

            seed = 42 

            t_E = np.arange(0.0, ss.bcs_tmax + ss.bcs_dt_E, ss.bcs_dt_E)
            rng = np.random.default_rng(seed)
            E_vals = rng.uniform(ss.bcs_E_min, ss.bcs_E_max, size=len(t_E))

            def E_of_t(t):
                # sinal em degraus: pega o último valor sorteado antes de t
                idx = np.searchsorted(t_E, t, side="right") - 1
                idx = np.clip(idx, 0, len(E_vals) - 1)
                return E_vals[idx]

            def motor_termo_odes(t, y):
                w, i, T = y

                R_T = ss.bcs_R0 * (1.0 + ss.bcs_alpha * (T - ss.bcs_T_inf))           # R(T)
                dw_dt = (ss.bcs_k1 * i - ss.bcs_b * w) / ss.bcs_J                   # J*w_dot = k1*i - b*w

                E_t = E_of_t(t)                                # <<< agora E varia no tempo
                di_dt = (E_t - R_T * i - ss.bcs_k2 * w) / ss.bcs_L           # L*i_dot = E(t) - R(T)*i - k2*w

                dT_dt = (i**2 * R_T - ss.bcs_hA * (T - ss.bcs_T_inf)) / ss.bcs_Ct   # Ct*T_dot = i^2*R(T) - hA(T-Tinf)

                return [dw_dt, di_dt, dT_dt]
            
            T_init = ss.bcs_T_inf

            sol = solve_ivp(
                motor_termo_odes,
                (0.0, ss.bcs_tmax),
                [ss.bcs_w0, ss.bcs_i0, T_init],
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-10
            )

            t = sol.t
            w = sol.y[0]
            i = sol.y[1]
            T = sol.y[2]

            # (Opcional) reconstruir E(t) no mesmo grid do gráfico
            E_plot = np.array([E_of_t(tt) for tt in t])

            col1, col2 = st.columns([1, 1])
            with col1:
                # 1) ω(t)
                fig1, ax1 = plt.subplots()
                ax1.plot(t, w)
                ax1.set_xlabel("Tempo [s]")
                ax1.set_ylabel("Velocidade angular ω [rad/s]")
                ax1.set_title("EDO mecânica: ω(t)")
                ax1.grid(True)
                st.pyplot(fig1, clear_figure=True)

                # 2) i(t)
                fig2, ax2 = plt.subplots()
                ax2.plot(t, i)
                ax2.set_xlabel("Tempo [s]")
                ax2.set_ylabel("Corrente i [A]")
                ax2.set_title("EDO elétrica: i(t)")
                ax2.grid(True)
                st.pyplot(fig2, clear_figure=True)

            with col2:
                # 3) T(t)
                fig3, ax3 = plt.subplots()
                ax3.plot(t, T)
                ax3.set_xlabel("Tempo [s]")
                ax3.set_ylabel("Temperatura T [K]")
                ax3.set_title("EDO térmica: T(t)")
                ax3.grid(True)
                st.pyplot(fig3, clear_figure=True)

                # 4) E(t) (entrada em degraus)
                fig4, ax4 = plt.subplots()
                ax4.step(t, E_plot, where="post")
                ax4.set_xlabel("Tempo [s]")
                ax4.set_ylabel("Tensão E(t) [V]")
                ax4.set_title("Entrada aleatória: E(t)")
                ax4.grid(True)
                st.pyplot(fig4, clear_figure=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    t_eval = np.linspace(0.0, ss.bcs_tmax, 2000)

    seed = 42 

    t_E = np.arange(0.0, ss.bcs_tmax + ss.bcs_dt_E, ss.bcs_dt_E)
    rng = np.random.default_rng(seed)
    E_vals = rng.uniform(ss.bcs_E_min, ss.bcs_E_max, size=len(t_E))

    def E_of_t(t):
        # sinal em degraus: pega o último valor sorteado antes de t
        idx = np.searchsorted(t_E, t, side="right") - 1
        idx = np.clip(idx, 0, len(E_vals) - 1)
        return E_vals[idx]

    def motor_termo_odes(t, y):
        w, i, T = y

        R_T = ss.bcs_R0 * (1.0 + ss.bcs_alpha * (T - ss.bcs_T_inf))           # R(T)
        dw_dt = (ss.bcs_k1 * i - ss.bcs_b * w) / ss.bcs_J                   # J*w_dot = k1*i - b*w

        E_t = E_of_t(t)                                # <<< agora E varia no tempo
        di_dt = (E_t - R_T * i - ss.bcs_k2 * w) / ss.bcs_L           # L*i_dot = E(t) - R(T)*i - k2*w

        dT_dt = (i**2 * R_T - ss.bcs_hA * (T - ss.bcs_T_inf)) / ss.bcs_Ct   # Ct*T_dot = i^2*R(T) - hA(T-Tinf)

        return [dw_dt, di_dt, dT_dt]
    
    T_init = ss.bcs_T_inf

    sol = solve_ivp(
        motor_termo_odes,
        (0.0, ss.bcs_tmax),
        [ss.bcs_w0, ss.bcs_i0, T_init],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    t = sol.t
    w = sol.y[0]
    i = sol.y[1]
    T = sol.y[2]

    # (Opcional) reconstruir E(t) no mesmo grid do gráfico
    E_plot = np.array([E_of_t(tt) for tt in t])

    col1, col2 = st.columns([1, 1])
    with col1:
        # 1) ω(t)
        fig1, ax1 = plt.subplots()
        ax1.plot(t, w)
        ax1.set_xlabel("Tempo [s]")
        ax1.set_ylabel("Velocidade angular ω [rad/s]")
        ax1.set_title("EDO mecânica: ω(t)")
        ax1.grid(True)
        st.pyplot(fig1, clear_figure=True)

        # 2) i(t)
        fig2, ax2 = plt.subplots()
        ax2.plot(t, i)
        ax2.set_xlabel("Tempo [s]")
        ax2.set_ylabel("Corrente i [A]")
        ax2.set_title("EDO elétrica: i(t)")
        ax2.grid(True)
        st.pyplot(fig2, clear_figure=True)

    with col2:
        # 3) T(t)
        fig3, ax3 = plt.subplots()
        ax3.plot(t, T)
        ax3.set_xlabel("Tempo [s]")
        ax3.set_ylabel("Temperatura T [K]")
        ax3.set_title("EDO térmica: T(t)")
        ax3.grid(True)
        st.pyplot(fig3, clear_figure=True)

        # 4) E(t) (entrada em degraus)
        fig4, ax4 = plt.subplots()
        ax4.step(t, E_plot, where="post")
        ax4.set_xlabel("Tempo [s]")
        ax4.set_ylabel("Tensão E(t) [V]")
        ax4.set_title("Entrada aleatória: E(t)")
        ax4.grid(True)
        st.pyplot(fig4, clear_figure=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("pages/images/motor_bomba_bcs.png")

    with col2:

        col3, col4 = st.columns([1, 1])

        # =========================
        # COLUNA 3
        # =========================
        with col3:

            # Inércia J
            with st.container(border=True):
                st.markdown("Momento de Inércia")
                st.markdown(f"J = {ss.bcs_J:.4f} kg·m²")
                J = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_J"
                )
                if J != ss.bcs_J:
                    ss.bcs_J = J
                    st.rerun()

            # Atrito viscoso b
            with st.container(border=True):
                st.markdown("Atrito Viscoso")
                st.markdown(f"b = {ss.bcs_b:.4f} N·m·s/rad")
                b = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_b"
                )
                if b != ss.bcs_b:
                    ss.bcs_b = b
                    st.rerun()

            # Constante de torque k1
            with st.container(border=True):
                st.markdown("Constante de Torque do Motor")
                st.markdown(f"k₁ = {ss.bcs_k1:.4f} N·m/A")
                k1 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_k1"
                )
                if k1 != ss.bcs_k1:
                    ss.bcs_k1 = k1
                    st.rerun()

            # Indutância L
            with st.container(border=True):
                st.markdown("Indutância da Armadura")
                st.markdown(f"L = {ss.bcs_L:.4f} H")
                L = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_L"
                )
                if L != ss.bcs_L:
                    ss.bcs_L = L
                    st.rerun()

            # Constante FEM k2
            with st.container(border=True):
                st.markdown("Constante de Força Eletromotriz (Back-EMF)")
                st.markdown(f"k₂ = {ss.bcs_k2:.4f} V·s/rad")
                k2 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_k2"
                )
                if k2 != ss.bcs_k2:
                    ss.bcs_k2 = k2
                    st.rerun()

            with st.container(border=True):
                st.markdown("Tensão mínima da fonte")
                st.markdown(f"E_min = {ss.bcs_E_min:.2f} V")
                E_min = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    max_value=ss.bcs_E_max,
                    key="bcs_E_min"
                )
                if E_min != ss.bcs_E_min:
                    ss.bcs_E_min = E_min
                    st.rerun()

            with st.container(border=True):
                st.markdown("Tensão máxima da fonte")
                st.markdown(f"E_max = {ss.bcs_E_max:.2f} V")
                E_max = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=ss.bcs_E_min,
                    key="bcs_E_max"
                )
                if E_max != ss.bcs_E_max:
                    ss.bcs_E_max = E_max
                    st.rerun()

        # =========================
        # COLUNA 4
        # =========================
        with col4:

            # Resistência R0
            with st.container(border=True):
                st.markdown("Resistência Elétrica na Temperatura de Referência")
                st.markdown(f"R₀ = {ss.bcs_R0:.4f} Ω")
                R0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_R0"
                )
                if R0 != ss.bcs_R0:
                    ss.bcs_R0 = R0
                    st.rerun()

            # Coeficiente térmico alpha
            with st.container(border=True):
                st.markdown("Coeficiente Térmico da Resistência")
                st.markdown(f"α = {ss.bcs_alpha:.5f} 1/K")
                alpha = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    format="%.5f",
                    key="bcs_alpha"
                )
                if alpha != ss.bcs_alpha:
                    ss.bcs_alpha = alpha
                    st.rerun()

            # Capacitância térmica Ct
            with st.container(border=True):
                st.markdown("Capacitância Térmica do Sistema")
                st.markdown(f"Cₜ = {ss.bcs_Ct:.4f} J/K")
                Ct = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_Ct"
                )
                if Ct != ss.bcs_Ct:
                    ss.bcs_Ct = Ct
                    st.rerun()

            # hA
            with st.container(border=True):
                st.markdown("Troca de Calor com o Ambiente (h·A)")
                st.markdown(f"h·A = {ss.bcs_hA:.4f} W/K")
                hA = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_hA"
                )
                if hA != ss.bcs_hA:
                    ss.bcs_hA = hA
                    st.rerun()

            # Temperatura ambiente
            with st.container(border=True):
                st.markdown("Temperatura do Ambiente")
                st.markdown(f"T∞ = {ss.bcs_T_inf:.2f} K")
                T_inf = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="bcs_T_inf"
                )
                if T_inf != ss.bcs_T_inf:
                    ss.bcs_T_inf = T_inf
                    st.rerun()

            # Tempo máximo
            with st.container(border=True):
                st.markdown("Tempo Total de Simulação")
                st.markdown(f"t_max = {ss.bcs_tmax:.2f} s")
                tmax = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="bcs_tmax"
                )
                if tmax != ss.bcs_tmax:
                    ss.bcs_tmax = tmax
                    st.rerun()

            # ω0
            with st.container(border=True):
                st.markdown("Velocidade Angular Inicial")
                st.markdown(f"ω₀ = {ss.bcs_w0:.4f} rad/s")
                w0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="bcs_w0"
                )
                if w0 != ss.bcs_w0:
                    ss.bcs_w0 = w0
                    st.rerun()

            # i0
            with st.container(border=True):
                st.markdown("Corrente Inicial")
                st.markdown(f"i₀ = {ss.bcs_i0:.4f} A")
                i0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="bcs_i0"
                )
                if i0 != ss.bcs_i0:
                    ss.bcs_i0 = i0
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicativos
    st.markdown("""
        De acordo com a Resolução ANP nº 749/2018, campo de petróleo ou de gás natural maduro é o que atingiu produção maior ou igual a vinte e cinco anos ou cuja produção acumulada corresponda a, pelo menos, 70% do volume a ser produzido previsto, considerando as reservas provadas. Conforme a produção progride, a pressão do reservatório frequentemente diminui, necessitando do uso de técnicas de elevação artificial para manter e aumentar as taxas de produção. A elevação artificial funciona reduzindo a pressão do fundo do poço ou fornecendo energia adicional aos fluidos produzidos, permitindo que eles fluam mais livremente para a superfície (Esimtech, 2024). Alguns dos métodos de elevação mais utilizados são o BM (Bombeio Mecânico), BCS (Bombeio Centrifugo Submerso), BCP (Bombeio por Cavidades Progressivas), Gas Lift, entre outros.

        O estudo de caso desse artigo será realizado em cima de um poço BCS. O método de elevação de bombeio centrifugo submerso consiste em uma bomba centrífuga elétrica de múltiplos estágios instalada dentro do poço, próxima à zona produtora, acionada por um motor elétrico descido junto à bomba. As ESP (Electric Submersible Pumps) conseguem elevar grandes volumes de fluido de poços profundos e são indicadas para poços de maior vazão ou alto corte de água.
        
        Essa bomba funciona com o auxílio de um motor que fica também localizado no fundo do poço.

        Para operar em condições tão críticas, é indispensável que exista um monitoramento constante desses equipamentos, para isso as empresas usam instrumentos de telemetria para acompanhar em tempo real o comportamento de indicadores como a temperatura do motor, a corrente, a tensão, a potência, entre outros. Mas além de ter os instrumentos de telemetria, é necessário saber se eles estão calibrados e o sistema está funcionando como o esperado, por isso é de suma importância conhecer o comportamento ideal desses sistemas. Nesse artigo iremos explorar o comportamento ideal e a modelagem matemática de um motor DC, e comparar com os dados coletados do poço.
    """)

    space_line()

    # Sugestões
    st.markdown("""
        Momento de inércia (J): ao aumentar J, o conjunto rotor + bomba fica mais “pesado” para acelerar. Isso reduz a aceleração angular para o mesmo torque,
        fazendo ω(t) subir mais devagar e suavizando o transitório. Valores menores de J deixam a resposta mais rápida, com ω(t) chegando ao patamar final em menos tempo.
        Como a velocidade demora mais para subir quando J é grande, a f.c.e.m. (k₂·ω) cresce mais lentamente, o que pode manter a corrente mais alta por mais tempo no início,
        influenciando também o aquecimento.

        Coeficiente de atrito viscoso (b): b representa perdas mecânicas proporcionais à velocidade.
        Quanto maior b, maior o torque resistivo para um mesmo ω, então o motor atinge uma ω(t) final menor e tende a estabilizar mais cedo, só que em um patamar mais baixo.
        Valores menores de b deixam o conjunto mais “livre”, com maior ω final, mas também podem elevar a potência mecânica exigida em regime se houver carga (no seu modelo,
        a carga é essencialmente viscosa).

        Constante de torque (k₁): aumentar k₁ significa que, para a mesma corrente, o motor gera mais torque. Na prática, ω(t) acelera mais forte no início,
        e o sistema vence mais facilmente os torques resistivos, elevando ω em regime. Já com k₁ menor, o motor fica “mais fraco”, precisa de mais corrente para gerar o mesmo torque,
        e pode terminar com ω(t) mais baixa (especialmente se b for alto).

        Indutância da armadura (L): L governa o quanto a corrente consegue variar rapidamente.
        Valores maiores de L “amortecem” mudanças bruscas em i(t), então a corrente demora mais para se ajustar quando E(t) muda em degraus,
        o que deixa i(t) mais suave e pode atrasar o torque (k₁·i) no começo. Com L menor, a corrente reage mais rápido aos degraus de tensão e à f.c.e.m.,
        fazendo i(t) acompanhar as variações de E(t) de forma mais imediata, geralmente com transitórios mais rápidos.

        Resistência de referência (R₀): R₀ é o “freio elétrico” da corrente. Aumentar R₀ reduz i(t) para uma mesma tensão E(t),
        diminuindo torque (k₁·i) e reduzindo ω(t) em regime. Como a geração térmica vem de i²·R(T), aumentar R₀ pode ter dois efeitos competindo:
        pode reduzir i, mas também aumenta a parcela resistiva, no fim, o comportamento depende do quanto a corrente cai. Com R₀ menor,
        o motor puxa mais corrente, gera mais torque e costuma aumentar ω final, porém tende a aquecer mais por conta de i².

        Coeficiente térmico da resistência (α): α controla o quanto a resistência cresce com a temperatura. Quanto maior α, mais forte é o “feedback térmico”:
        T(t) sobe, R(T) aumenta, a corrente i(t) cai, o torque diminui e a dinâmica do motor se “auto limita”.
        Isso costuma reduzir ω(t) e suavizar a corrente após algum tempo, podendo criar um comportamento em que o sistema acelera bem no começo,
        mas perde fôlego conforme aquece. Com α pequeno, a resistência fica quase constante, então a parte elétrica muda pouco com a temperatura,
        e o acoplamento T → i fica bem mais fraco.

        Constante de f.c.e.m. (k₂): k₂ define o quanto a velocidade gera oposição elétrica via back-emf.
        Aumentar k₂ faz a f.c.e.m. crescer mais rápido com ω, reduzindo i(t) e limitando ω(t) em regime, o motor “se segura” mais cedo.
        Com k₂ menor, a oposição elétrica por velocidade é menor, então o motor tende a alcançar ω(t) maiores, geralmente com mais corrente em regime
        e, portanto, mais aquecimento.

        Capacitância térmica (Cₜ): Cₜ é a “inércia térmica”. Quanto maior Cₜ, mais difícil é a temperatura mudar,
        então T(t) sobe mais devagar e fica mais “lenta” para responder a mudanças em i(t). Isso também atrasa o aumento de R(T),
        mantendo a corrente mais próxima do que seria sem aquecimento por mais tempo. Com Cₜ pequeno, T(t) reage rápido,
        R(T) muda cedo, e o efeito térmico aparece mais rapidamente na corrente e na velocidade.

        Troca térmica com o ambiente (h·A): h·A mede o quanto o sistema consegue perder calor para o ambiente.
        Aumentar h·A puxa T(t) para perto de T∞ mais rapidamente, reduzindo o pico e o patamar de temperatura. Isso tende a manter R(T) menor,
        o que pode aumentar i(t) e, indiretamente, sustentar ω(t) um pouco mais alto ao longo do tempo. Se h·A é pequeno, o motor dissipa pouco,
        T(t) sobe mais, R(T) cresce mais, e o motor “perde corrente” com o passar do tempo por conta do aquecimento.

        Temperatura ambiente (T∞): T∞ é o “piso térmico” do sistema. Se T∞ aumenta, o motor parte mais quente (no seu modelo T_init = T∞),
        o que já começa com R(T) maior, reduzindo i(t) e torque desde o início. Com T∞ menor, o motor começa com resistência menor,
        puxa mais corrente no começo e acelera mais forte, mas também pode aquecer mais rápido dependendo do balanço entre geração e dissipação.

        Tensão mínima e máxima (E_min e E_max): esses parâmetros controlam o intervalo dos degraus aleatórios de E(t).
        Aumentar E_min/E_max desloca a entrada para níveis mais altos, elevando a corrente média, o torque médio e, em geral, ω(t) e T(t).
        Diminuir esses valores deixa o motor mais “comportado”, com i(t) menor, menos torque e menos aquecimento. A largura do intervalo (E_max - E_min)
        também importa: intervalos mais largos geram uma entrada mais “agitada”, o que aumenta as oscilações de i(t), ω(t) e pode criar mais variação térmica.

        Passo de atualização da entrada (dt_E): dt_E controla de quanto em quanto tempo você sorteia um novo valor de tensão.
        Se dt_E é pequeno, E(t) muda frequentemente, então i(t) e ω(t) ficam mais “nervosos”, tentando acompanhar os degraus.
        Se dt_E é grande, E(t) fica mais tempo constante, e o sistema tem mais chance de quase estabilizar entre uma mudança e outra,
        deixando as curvas mais “em blocos” e com transientes mais completos.

        Tempo máximo (t_max): não altera a física, só o quanto do filme você está assistindo.
        Se t_max é curto, você vê só o começo do transitório, se é longo, dá para enxergar como o acoplamento térmico vai mudando o comportamento ao longo do tempo,
        especialmente quando T(t) já subiu o suficiente para impactar R(T).

        Condições iniciais de velocidade e corrente (ω₀ e i₀): ω₀ desloca o ponto inicial de ω(t).
        Se ω₀ começa alto, a f.c.e.m. (k₂·ω) já nasce alta, reduzindo a corrente imediatamente, e ω(t) pode cair até um patamar mais baixo.
        Se ω₀ começa baixo, a corrente inicial tende a ser maior (dependendo de E(t)), e ω(t) cresce até o regime.
        Já i₀ define o torque inicial, então um i₀ maior pode dar um “empurrão” no começo, mas também aumenta a geração térmica instantânea (i²·R),
        fazendo T(t) responder mais rápido logo no início.
    """)