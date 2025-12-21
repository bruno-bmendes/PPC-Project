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
        ss.page_set = ["Início", "Vaso Pulmão", "Circuito RC", "Circuito RLC", "Sistema Massa Mola Amortecedor", "Pêndulo Simples Amortecido", "Sistema Eletromecanico", "Tanque com Aquecimento"]

    ss.page = "tanque_com_aquecimento"
    st.rerun()

# Definindo página
def tanque_com_aquecimento():

    # Declarando Variáveis
    if "ta_tmax" not in ss:
        ss.ta_tmax = 2.0  # tempo total de simulação
    if "ta_A" not in ss:
        ss.ta_A = 1.0  # m^2  (área seção transversal do tanque)
    if "ta_k" not in ss:
        ss.ta_k = 0.15  # (m^3/s)/sqrt(m) (Torricelli ajustado de V e A)
    if "ta_rho" not in ss:
        ss.ta_rho = 1000.0  # kg/m^3 (densidade do fluido)
    if "ta_cp" not in ss:
        ss.ta_cp = 4180.0  # J/(kg.K) (calor específico do fluido)

    # Vapor/condensação:
    if "ta_lambda_c" not in ss:
        ss.ta_lambda_c = 2.26e6  # J/kg (calor latente)

    # PONTO DE OPERAÇÃO (equilíbrio)
    if "ta_qi0" not in ss:
        ss.ta_qi0 = 0.10  # m^3/s (vazão de entrada)
    if "ta_Ti0" not in ss:
        ss.ta_Ti0 = 300.0 # K
    if "ta_V0" not in ss:
        ss.ta_V0  = ss.ta_A * ( (ss.ta_qi0 / ss.ta_k) ** 2 )  # de 0 = qi0 - k*sqrt(V0/A) => V0 = A*(qi0/k)^2
    if "ta_T0" not in ss:
        ss.ta_T0 = 310.0  # K
    if "ta_qc0" not in ss:
        ss.ta_qc0 = - ss.ta_rho*ss.ta_cp * ss.ta_qi0 * (ss.ta_Ti0 - ss.ta_T0) / ss.ta_lambda_c     # kg/s

    # PERTURBAÇÕES (degrau) nas entradas
    if "ta_t0" not in ss:
        ss.ta_t0 = 0.0
    if "ta_tf" not in ss:
        ss.ta_tf = 500.0
    if "ta_t_step" not in ss:
        ss.ta_t_step = 0.0

    if "ta_dqi" not in ss:
        ss.ta_dqi = 0.02      # m^3/s
    if "ta_dTi" not in ss:
        ss.ta_dTi = 5.0       # K
    if "ta_dqc" not in ss:
        ss.ta_dqc = 0.0       # (kg/s)

    # Definir Título
    ss.title = "Tanque com Aquecimento"
    
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
            Definindo as premissas do sistema:
            - Fluido ideal (incompressível, homogêneo);
            - Energia cinética e energia potencial desprezíveis;
            - Balanço de momento desprezível;
            - Vazão de entrada constante;
            - Densidade e calor específico constantes;
            - Sem geração interna de calor;
            - O aquecimento do tanque ocorre por condensação de vapor, sendo o calor transferido ao fluido proporcional à vazão mássica de vapor condensado e ao calor latente de condensação, assumido constante, poranto Tc pode ser desconsiderado.
        """)

        st.markdown("O sistema é dividido em 2 EDOs que moldam seu comportamento. A primeira EDO a ser observada é a do comportamento mecânico, olhando para a variação do volume do tanque durante o tempo.")

        st.markdown("Para definirmos a primeira EDO, partimos da premissa básica em que a variação de volume é igual a vazão de entrada menos a vazão de saída:")

        st.latex(r"\frac{dV}{dt} = q_i - q_{out}")

        st.markdown("Conforme a Lei de Torricelli de escoamento por gravidade, podemos inferir que a vazão de saída é proporcional à raiz quadrada da altura do líquido, sendo ajustada por uma constante 'k' que vai considerar as incertezas:")

        st.latex(r"\frac{dV}{dt} = q_i - k\sqrt{h}")

        st.markdown("Ainda podemos admitir a altura da coluna de fluido como a razão da volume ocupado pela área da seção transversal do tanque:")

        st.latex(r"\frac{dV}{dt} = q_i - k\sqrt{\frac{V}{A}}")

        st.markdown("A segunda EDO descreve o acúmulo de energia no fluido do tanque. Para sua formulação, utiliza-se a definição do calor específico a pressão constante, que corresponde à variação da entalpia específica com a temperatura, sob a hipótese de pressão constante:")

        st.latex(r"c_p \equiv \left(\frac{\partial h}{\partial T}\right)_p")

        st.markdown("Como assumimos a premissa de que cp não varia com a temperatura, podemos chegar na definição:")

        st.latex(r"dh = c_p\, dT")

        st.markdown("Integrando de um estado de referência 𝑇0 até 𝑇:")

        st.latex(r"h(T) - h(T_0) = \int_{T_0}^{T} c_p\, dT \approx c_p\,(T - T_0)")

        st.markdown("Se considerarmos a constante:")

        st.latex(r"h(T) \approx c_p T + \text{constante}")

        st.markdown("Como o tanque tem massa m=ρV e energia carregada por unidade de massa = h, temos a energia total associada ao líquido armazenado:")

        st.latex(r"E_{\text{armazenada}} = \rho V\, h(T)")

        st.markdown("Substituindo a entalpia pela relação declarada anteriormente:")

        st.latex(r"E_{\text{armazenada}} = \rho V\, (c_p T + \text{const})")

        st.markdown("Com isso, podemos derivar a equação no tempo:")

        st.latex(
            r"\frac{dE_{\text{armazenada}}}{dt} = \frac{d}{dt}\left(\rho V\,(c_p T + \text{const})\right)"
        )

        st.markdown("Nesse caso como cp e ρ são constantes, e a derivada de 'const' é zero:")

        st.latex(r"\frac{dE_{\text{armazenada}}}{dt} = \rho c_p \frac{d}{dt}(VT)")

        st.markdown("Como V e T variam com o tempo, podemos aplicar a regra do produto:")

        st.latex(
            r"\frac{dE_{\text{armazenada}}}{dt} = \rho c_p \left( V\frac{dT}{dt} + T\frac{dV}{dt} \right)"
        )

        st.markdown("Agora para fazermos o balanço de energia:")

        st.latex(
            r"\underbrace{\frac{dE_{\text{armazenada}}}{dt}}_{\text{acúmulo}}"
            r" = "
            r"\underbrace{\dot{E}_{in}}_{\text{entra com o líquido}}"
            r" - "
            r"\underbrace{\dot{E}_{out}}_{\text{sai com o líquido}}"
            r" + "
            r"\underbrace{\dot{Q}_{vapor}}_{\text{calor do vapor}}"
        )

        st.markdown("Para a energia que entra com a alimentação: A vazão mássica de entrada é 𝜌*𝑞𝑖 e a entalpia específica de entrada é 𝑐𝑝*𝑇𝑖, então:")

        st.latex(r"\dot{E}_{in} = \rho q_i c_p T_i")

        st.markdown("Para a energia que sai com a descarga: A vazão mássica de saída é 𝜌*𝑞𝑜𝑢𝑡 e a entalpia específica de saída é 𝑐𝑝*𝑇 (mistura perfeita, sai na temperatura do tanque):")

        st.latex(r"\dot{E}_{out} = \rho q_{out} c_p T")

        st.markdown("Para o calor fornecido pelo vapor: Se condensa a taxa mássica 𝑞𝑐 e libera calor latente 𝜆𝑐:")

        st.latex(r"\dot{Q}_{vapor} = q_c \lambda_c")

        st.markdown("Dessa forma chegamos na equação montada:")

        st.latex(
            r"\rho c_p \left( V \frac{dT}{dt} + T \frac{dV}{dt} \right) = \rho q_i c_p T_i - \rho q_{out} c_p T + \rho q_c \lambda_c"
        )

        st.markdown("A partir disso, podemo susar o balanço de massa da primeira EDO para substituir o qout​:")

        st.latex(
            r"\frac{dV}{dt} = q_i - q_{out} \;\Rightarrow\; q_{out} = q_i - \frac{dV}{dt}"
        )

        st.markdown("Substitui isso no termo de saída:")

        st.latex(
            r"-\rho q_{out} c_p T"
            r" = -\rho \left(q_i - \frac{dV}{dt}\right) c_p T"
            r" = -\rho q_i c_p T + \rho c_p T \frac{dV}{dt}"
        )

        st.markdown("e,")

        st.latex(
            r"""
            \begin{aligned}
            \rho c_p \left( V \frac{dT}{dt} + T \frac{dV}{dt} \right)
            &= \rho q_i c_p T_i \\
            &\quad + \left( -\rho q_i c_p T + \rho c_p T \frac{dV}{dt} \right)
            + \rho q_c \lambda_c
            \end{aligned}
            """
        )

        st.markdown("Chegando na equação final:")

        st.latex(
            r"\rho c_p V \frac{dT}{dt} = \rho q_i c_p (T_i - T) + \rho q_c \lambda_c"
        )

        st.markdown("No intuito de simplificar o sistema, inicia-se o processo de linearização. A primeira etapa do processo de linearização é definir as matrizes dos estados e entradas do modelo:")

        st.markdown("• **Estados:**")
        st.latex(r"\mathbf{x} = \begin{bmatrix} V \\ T \end{bmatrix}")

        st.markdown("• **Entradas:**")
        st.latex(r"\mathbf{u} = \begin{bmatrix} q_i \\ T_i \\ q_c \end{bmatrix}")

        st.markdown(
            r"O segundo passo é encontrar o equilíbrio $(\mathbf{x}^*, \mathbf{u}^*)$, isto é, um ponto em que o sistema fica estacionário com entradas constantes:"
        )

        st.latex(r"\dot{V} = 0")
        st.latex(r"\dot{T} = 0")

        st.markdown(
            r"Esse $(V^*, T^*)$ é o **centro** em torno do qual você vai aproximar o sistema não linear por um sistema linear. Depois disso, o próximo passo é definir as EDOs citadas anteriormente nesse ponto de equilibrio:"
        )

        st.markdown("Para a EDO mecânica:")

        st.latex(r"\dot{V} = q_i - k\sqrt{\frac{V}{A}}")

        st.latex(r"0 = q_i^* - k\sqrt{\frac{V^*}{A}}")

        st.markdown("Para a EDO térmica, isolando a derivada, e definindo a constante Tc como :")

        st.latex(
            r"\dot{T} = \frac{\rho q_i c_p (T_i - T) + \rho q_c \lambda_c}{\rho V c_p}"
        )

        st.latex(
            r"0 = \frac{\rho q_i^* c_p (T_i^* - T^*) + \rho q_c^* \lambda_c}{\rho V^* c_p}"
        )

        st.markdown("Nesse ponto de equilibrio, vamos declarar a matriz Jacobiana, que se dá pela regra:")

        st.latex(
            r"\mathbf{A} = \left["
            r"\begin{array}{cc}"
            r"\frac{\partial f_1}{\partial V} & \frac{\partial f_1}{\partial T} \\"
            r"\frac{\partial f_2}{\partial V} & \frac{\partial f_2}{\partial T}"
            r"\end{array}"
            r"\right]_{V_0,\,T_0}"
        )

        st.markdown("A Jacobiana mostra o quanto cada variável do sistema influencia a evolução das outras, quando você faz uma pequena perturbação perto do equilíbrio. Para isso, vamos calcular cada termo da matriz:")

        st.latex(
            r"A_{11} = \frac{\partial f_1}{\partial V} = \frac{\partial}{\partial V} \left[ q_i - k \sqrt{\frac{V}{A}} \right]"
        )

        st.latex(r"A_{11} = -\frac{k}{2\sqrt{A_{\text{tank}}}\sqrt{V_0}}")

        st.markdown(
            "Esse primeiro termo mostra como variações de volume afetam a taxa de saída. O sinal negativo indica que, quanto maior o volume (maior nível), maior a vazão de saída e menor a taxa de acumulação."
        )

        st.latex(
            r"A_{12} = \frac{\partial f_1}{\partial T}"
            r" = \frac{\partial}{\partial T}\left[ q_i - k\sqrt{\frac{V}{A}} \right]"
        )

        st.latex(r"A_{12} = 0")

        st.markdown(
            "O segundo termo mostra que a temperatura não altera diretamente o balanço de massa, apenas o nível/volume importa."
        )

        st.latex(
            r"A_{21} = \frac{\partial f_2}{\partial V}"
            r" = \frac{\partial}{\partial V}"
            r"\left[ \frac{\rho q_i c_p (T_i - T) + \rho q_c \lambda_c}{\rho c_p V} \right]"
        )

        st.latex(
            r"A_{21} = -\frac{\rho q_i c_p (T_i - T_0) + \rho q_c \lambda_c}{\rho c_p V_0^{2}}"
        )

        st.markdown(
            "O terceiro termo mostra como o volume atua como “capacidade térmica”. Um volume maior dilui o efeito das perturbações, deixando a temperatura variar mais devagar. O sinal negativo mostra que aumentar V reduz a taxa de variação de T."
        )

        st.latex(
            r"A_{22} = \frac{\partial f_2}{\partial T}"
            r" = \frac{\partial}{\partial T}"
            r"\left[ \frac{\rho q_i c_p (T_i - T) + \rho q_c \lambda_c}{\rho c_p V} \right]"
        )
        
        st.latex(r"A_{22} = -\frac{q_i}{V_0}")

        st.markdown(
            "O último termo representa o amortecimento térmico devido à corrente de entrada. Quanto maior a vazão de entrada $q_i$ (para um mesmo volume), mais rapidamente a temperatura é “puxada” para o valor de referência, aumentando a taxa de retorno ao equilíbrio."
        )

        st.markdown("O próximo passo é realizar a operação de A*Δx, com:")

        st.latex(
            r"\Delta \mathbf{x} = \begin{bmatrix} \Delta V \\ \Delta T \end{bmatrix}"
        )

        st.latex(
            r"""
            A\,\Delta \mathbf{x} =
            \begin{bmatrix}
            -\dfrac{k}{2\sqrt{A}\sqrt{V_0}}\,\Delta V \\[10pt]
            -\dfrac{\rho q_i c_p (T_i - T_0) + \rho_c q_c \lambda_c}{\rho c_p V_0^{2}}\,\Delta V
            \;-\; \dfrac{q_i}{V_0}\,\Delta T
            \end{bmatrix}
            """
        )

        st.markdown("A próxima etapa é calcular a matriz B, composta pelas derivadas da função f1 e f2 em relação as variáveis de entrada qi, Ti e qc, avaliadas pelo ponto de equilíbrio (V(0), T(0)):")

        st.latex(
            r"\mathbf{B} = \left["
            r"\begin{array}{ccc}"
            r"\dfrac{\partial f_1}{\partial q_i} & \dfrac{\partial f_1}{\partial T_i} & \dfrac{\partial f_1}{\partial q_c} \\"
            r"\dfrac{\partial f_2}{\partial q_i} & \dfrac{\partial f_2}{\partial T_i} & \dfrac{\partial f_2}{\partial q_c}"
            r"\end{array}"
            r"\right]_{V_0,\,T_0}"
        )

        st.markdown("Na mesma forma como foi feito na Jacobiana, vamos declarar cada termo da matriz B:")

        st.latex(r"B_{11} = \frac{\partial f_1}{\partial q_i}")

        st.latex(r"B_{11} = 1")

        st.markdown(
            "$\\Delta q_i$ afeta diretamente a taxa de acumulação de volume ($\\Delta \\dot{V}$)."
        )

        st.latex(r"B_{12} = \frac{\partial f_1}{\partial T_i}")

        st.latex(r"B_{12} = 0")

        st.markdown(
            r"Temperatura de Entrada ($T_i$) não afeta Balanço de Massa ($\dot{V}$)."
        )

        st.latex(r"B_{13} = \frac{\partial f_1}{\partial q_c}")

        st.latex(r"B_{13} = 0")

        st.markdown(
            r"Vazão da Camisa ($\Delta q_c$) não afeta Balanço de Massa ($\dot{V}$)."
        )

        st.latex(r"B_{21} = \frac{\partial f_2}{\partial q_i}")

        st.latex(r"B_{21} = \frac{T_i - T_0}{V_0}")

        st.markdown(
            r"$\Delta q_i$ afeta a taxa de variação da Temperatura ($\Delta \dot{T}$)."
        )

        st.latex(r"B_{22} = \frac{\partial f_2}{\partial T_i}")

        st.latex(r"B_{22} = \frac{q_{i0}}{V_0}")

        st.markdown(
            r"$\Delta T_i$ afeta a taxa de $\Delta \dot{T}$."
        )

        st.latex(r"B_{23} = \frac{\partial f_2}{\partial q_c}")

        st.latex(
            r"B_{23} = \frac{\rho_c \lambda_c}{\rho V_0 c_p}"
        )

        st.markdown(
            r"$\Delta q_c$ é a principal entrada de calor e afeta diretamente $\Delta \dot{T}$."
        )

        st.markdown("Dessa forma, aplicamos a multiplicação de B pela matriz Δu, sendo:")

        st.latex(
            r"\Delta \mathbf{u} = \begin{bmatrix} \Delta q_i \\ \Delta T_i \\ \Delta q_c \end{bmatrix}"
        )

        st.latex(
            r"\mathbf{B}\,\Delta \mathbf{u} = "
            r"\begin{bmatrix}"
            r"1 \cdot \Delta q_i \\[6pt]"
            r"\dfrac{T_{i0}-T_0}{V_0}\,\Delta q_i"
            r" + \dfrac{q_{i0}}{V_0}\,\Delta T_i"
            r" + \dfrac{\rho_c \lambda_c}{\rho V_0 c_p}\,\Delta q_c"
            r"\end{bmatrix}"
        )

        st.markdown("Assim, utilizando as matrizes, chegamos nas duas EDOs linearizadas:")

        st.latex(
            r"\frac{d(\Delta V)}{dt}"
            r"\approx"
            r"-\frac{k}{2\sqrt{A}\sqrt{V_0}}\,\Delta V"
            r"+ 1\cdot \Delta q_i"
        )

        st.markdown("e:")

        st.latex(
            r"""
            \begin{aligned}
            \frac{d(\Delta T)}{dt}
            &\approx
            -\frac{\rho q_i c_p (T_i - T_0) + \rho_c q_c \lambda_c}
                {\rho c_p V_0^{2}}\,\Delta V
            - \frac{q_i}{V_0}\,\Delta T \\[6pt]
            &\quad
            + \frac{T_{i0} - T_0}{V_0}\,\Delta q_i
            + \frac{q_i}{V_0}\,\Delta T_i
            + \frac{\rho_c \lambda_c}{\rho V_0 c_p}\,\Delta q_c
            \end{aligned}
            """
        )

        st.markdown("Com as EDOs linearizadas podemos chegar nas equações de transferência, a partir da Transformada de Laplace.")

        st.markdown(
            "A Transformada de Laplace é usada para simplificar a análise do sistema. "
            "Ela transforma equações diferenciais no tempo em equações algébricas no domínio s, "
            "onde derivadas viram multiplicações por s."
        )

        st.markdown(
            "Com isso, a dinâmica do sistema pode ser descrita por funções de transferência, "
            "que relacionam entradas e saídas e permitem analisar estabilidade, polos e resposta dinâmica "
            "de forma mais simples."
        )

        st.markdown("Aplicando Laplace na EDO mecânica linearizada, teremos as transformações:")

        st.latex(
            r"\mathcal{L}\left\{\frac{d}{dt}\right\} = s"
        )

        st.latex(
            r"\mathcal{L}\{\Delta V\} = \Delta V(s)"
        )

        st.latex(
            r"\mathcal{L}\{\Delta T\} = \Delta T(s)"
        )

        st.latex(
            r"\mathcal{L}\{\Delta q_i\} = \Delta q_i(s)"
        )

        st.latex(
            r"\mathcal{L}\{\Delta T_i\} = \Delta T_i(s)"
        )

        st.latex(
            r"\mathcal{L}\{\Delta q_c\} = \Delta q_c(s)"
        )

        st.markdown("Portanto a EDO mecânica ficará no formato:")

        st.latex(
            r"s\,\Delta V(s) \approx -\frac{k}{2\sqrt{A}\sqrt{V_0}}\,\Delta V(s) + \Delta q_i(s)"
        )

        st.markdown("Enquanto a EDO térmica ficará no formato:")

        st.latex(
            r"\begin{aligned}"
            r"s\,\Delta T(s) \approx "
            r"&-\frac{\rho q_{i0} c_p (T_{i0} - T_0) + \rho_c q_{c0} \lambda_c}{\rho c_p V_0^{2}}\,\Delta V(s)"
            r" - \frac{q_{i0}}{V_0}\,\Delta T(s) \\"
            r"&+ \frac{T_{i0} - T_0}{V_0}\,\Delta q_i(s)"
            r" + \frac{q_{i0}}{V_0}\,\Delta T_i(s)"
            r" + \frac{\rho_c \lambda_c}{\rho V_0 c_p}\,\Delta q_c(s)"
            r"\end{aligned}"
        )

        st.markdown(
            "Após aplicar a Transformada de Laplace às EDOs linearizadas, as equações passam a ser algébricas. "
            "O próximo passo é reorganizar os termos de forma a isolar as variáveis de estado no domínio de Laplace, "
            "obtendo uma expressão matricial padrão que facilite a análise do sistema."
        )

        st.latex(
            r"(sI - A)\,\Delta X(s) = B\,\Delta U(s)"
        )

        st.markdown(
            "Relembrando que: "
            "I é a matriz identidade, s é a variável complexa da Transformada de Laplace, "
            "A é Jacobiana, "
            "ΔX(s) representa o vetor das variáveis de estado no domínio de Laplace, "
            "B é a matriz de entradas e "
            "ΔU(s) é o vetor das perturbações ou entradas do sistema. "
            "Essa forma é a base para a obtenção direta da matriz de funções de transferência G(s)."
        )

        st.markdown("Partimos da premissa que o sistema começa no modelo:")

        st.latex(
            r"s\,\Delta X(s) = A\,\Delta X(s) + B\,\Delta U(s)"
        )

        st.markdown("Para chegarmos na função objetivo, a primeira etapa é subtrair o termo da Jacobiana em ambos os lados:")

        st.latex(
           r"s\,\Delta X(s) - A\,\Delta X(s) = B\,\Delta U(s)"
        )

        st.markdown("A EDO mecânica ficará no formato:")

        st.latex(
            r"\left(s + \frac{k}{2\sqrt{A}\sqrt{V_0}}\right)\,\Delta V(s) = \Delta q_i(s)"
        )

        st.markdown("Enquanto a EDO térmica:")

        st.latex(
            r"\begin{aligned}"
            r"\left(s + \frac{q_{i0}}{V_0}\right)\,\Delta T(s)"
            r" + \frac{\rho q_{i0} c_p (T_{i0} - T_0) + \rho_c q_{c0} \lambda_c}{\rho c_p V_0^{2}}\,\Delta V(s)"
            r"&= \\"
            r"&\hspace{-6.0em}\frac{T_{i0} - T_0}{V_0}\,\Delta q_i(s)"
            r" + \frac{q_{i0}}{V_0}\,\Delta t_i(s)"
            r" + \frac{\rho_c \lambda_c}{\rho V_0 c_p}\,\Delta q_c(s)"
            r"\end{aligned}"
        )

        st.markdown("O próximo passo é introduzir a matriz identidade pra que cheguemos no formato:")

        st.latex(
            r"(sI - A)\,\Delta X(s) = B\,\Delta U(s)"
        )

        st.markdown("Portanto teremos a matriz:")

        st.latex(
            r"sI - A = "
            r"\begin{bmatrix}"
            r"s + \frac{k}{2 \sqrt{A}\sqrt{V_0}} & 0 \\"
            r"\frac{\rho q_{i0} c_p (T_{i0} - T_0) + \rho_c q_{c0} \lambda_c}{\rho c_p V_0^{2}}"
            r" & s + \frac{q_{i0}}{V_0}"
            r"\end{bmatrix}"
        )

        st.markdown("O próximo passo é chegarmos no formato:")

        st.latex(
            r"\Delta X(s) = (sI - A)^{-1} B\,\Delta U(s)"
        )

        st.markdown("Sendo assim, chegaremos nas equações:")

        st.latex(
            r"\Delta V(s)"
            r" = \frac{\Delta q_i(s)}{\,s + \frac{k}{2\sqrt{A}\sqrt{V_0}}\,}"
        )

        st.markdown("e,")

        st.latex(
            r"\Delta T(s) = "
            r"\frac{\alpha\,\frac{T_{i0} - T_0}{V_0} - \gamma}{\alpha\,\beta}\,\Delta q_i(s)"
            r" + \frac{q_{i0}/V_0}{\beta}\,\Delta T_i(s)"
            r" + \frac{\rho_c \lambda_c}{\rho V_0 c_p\,\beta}\,\Delta q_c(s)"
        )

        st.markdown("Com:")

        st.latex(
            r"\alpha = s + \frac{k}{2\sqrt{A}\sqrt{V_0}}"
        )

        st.latex(
            r"\beta = s + \frac{q_{i0}}{V_0}"
        )

        st.latex(
            r"\gamma = \frac{\rho q_{i0} c_p (T_{i0} - T_0) + \rho_c q_{c0} \lambda_c}{\rho c_p V_0^{2}}"
        )

        st.markdown("E por fim, queremos chegar na relação que mostra como uma função de saida se comporta conforme a função de entrada:")

        st.latex(
            r"\frac{\Delta X(s)}{\Delta U(s)} = (sI - A)^{-1} B"
        )

        st.markdown("E podemos chamar cada uma dessas funções de transferências de:")

        st.latex(
            r"G(s) = \frac{\Delta X(s)}{\Delta U(s)}"
        )

        st.markdown("Para isso vamos resolver o lado direito da equação e chegar na matriz G(s) que traz todas as funções de transferência:")

        st.latex(
            r"G(s) = "
            r"\begin{bmatrix}"
            r"G_{V/q_i}(s) & G_{V/T_i}(s) & G_{V/q_c}(s) \\"
            r"G_{T/q_i}(s) & G_{T/T_i}(s) & G_{T/q_c}(s)"
            r"\end{bmatrix}"
        )

        st.markdown("Onde teremos que:")

        st.latex(
            r"G_{11}(s) = \frac{\Delta V(s)}{\Delta q_i(s)}"
            r" = \frac{1}{\,s + \frac{k}{2\sqrt{A}\sqrt{V_0}}\,}"
        )

        st.latex(
            r"G_{12}(s) = \frac{\Delta V(s)}{\Delta T_i(s)} = 0"
        )

        st.latex(
            r"G_{13}(s) = \frac{\Delta V(s)}{\Delta q_c(s)} = 0"
        )

        st.latex(
            r"G_{21}(s) = \frac{\Delta T(s)}{\Delta q_i(s)} = \frac{\dfrac{T_{i0} - T_0}{V_0}}{s + \dfrac{q_{i0}}{V_0}}"
        )

        st.latex(
            r"G_{22}(s) = \frac{\Delta T(s)}{\Delta T_i(s)}"
            r" = \frac{\dfrac{q_{i0}}{V_0}}{s + \dfrac{q_{i0}}{V_0}}"
        )

        st.latex(
            r"G_{23}(s) = \frac{\Delta T(s)}{\Delta q_c(s)}"
            r" = \frac{\dfrac{\lambda_c}{V_0 c_p}}{s + \dfrac{q_{i0}}{V_0}}"
        )

    # Layout principal
    col1, col2 = st.columns([12, 1])

    with col2:
        if st.button(":material/info:"):
            info()

    # Fórmula Inicial
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("EDO Mecânica")
        st.latex(r"\frac{dV}{dt} = q_i - k\sqrt{\frac{V}{A}}")

        st.markdown("EDO Mecânica Linearizada")
        st.latex(
            r"\frac{d(\Delta V)}{dt}"
            r"\approx"
            r"-\frac{k}{2A_{\text{tank}}\sqrt{h_0}}\,\Delta V"
            r"+ 1\cdot \Delta q_i"
        )
    with col2:
        st.markdown("EDO Térmica")
        st.latex(
            r"\rho c_p V \frac{dT}{dt} = \rho q_i c_p (T_i - T) + \rho q_c \lambda_c"
        )

        st.markdown("EDO Térmica Linearizada")
        st.latex( r"\frac{d(\Delta T)}{dt} \approx " r"-\frac{\rho q_i c_p (T_i - T_0) + \rho_c q_c \lambda_c T_c}{\rho c_p V_0^{2}}\,\Delta V" r" - \frac{q_i}{V_0}\,\Delta T" r" + \frac{T_{i0} - T_0}{V_0}\,\Delta q_i" r" + \frac{q_i}{V_0}\,\Delta T_i" r" + \frac{\rho_c \lambda_c T_c}{\rho V_0 c_p}\,\Delta q_c" )

    # Botão de Código
    @st.dialog("Código Utilizado")
    def code():

        st.code("""
             def inputs(t):
                if t < ss.ta_t_step:
                    return ss.ta_qi0, ss.ta_Ti0, ss.ta_qc0
                return ss.ta_qi0 + ss.ta_dqi, ss.ta_Ti0 + ss.ta_dTi, ss.ta_qc0 + ss.ta_dqc
            
            # NÃO LINEAR: EDOs originais
            def f_nl(t, x):
                V, T = x
                ss.ta_qi, ss.ta_Ti, ss.ta_qc = inputs(t)

                # mecânica (Torricelli)
                qout = ss.ta_k * np.sqrt(max(V, 0.0) / ss.ta_A)
                dVdt = ss.ta_qi - qout

                # térmica
                Qterm = (ss.ta_qc * ss.ta_lambda_c) / (ss.ta_rho * ss.ta_cp)   # equivalente em (m^3/s)*K? -> entra como termo de aquecimento/ V

                # dT/dt = qi*(Ti - T)/V + Qterm/V
                dTdt = 0.0
                if V > 1e-9:
                    dTdt = (ss.ta_qi * (ss.ta_Ti - T) + Qterm) / V

                return [dVdt, dTdt]
            
            # LINEARIZADO (em torno do equilíbrio)
            # Estados: ΔV, ΔT ; Entradas: Δqi, ΔTi, Δqc
            A11 = -ss.ta_k / (2.0 * np.sqrt(ss.ta_A) * np.sqrt(ss.ta_V0))
            A12 = 0.0

            # Para f2 = qi(Ti-T)/V + Qterm/V
            A22 = -ss.ta_qi0 / ss.ta_V0

            # A21 no equilíbrio zera (pois f2(V0,T0,u0)=0)
            # A21 = -( qi0*(Ti0-T0) + Qterm0 ) / V0^2 ; no equilíbrio isso dá 0
            Qterm0 = (ss.ta_qc0 * ss.ta_lambda_c) / (ss.ta_rho * ss.ta_cp)
            A21 = -(ss.ta_qi0 * (ss.ta_Ti0 - ss.ta_T0) + Qterm0) / (ss.ta_V0**2)

            # Matriz B (avaliada no ponto de operação)
            B11 = 1.0
            B12 = 0.0
            B13 = 0.0

            B21 = (ss.ta_Ti0 - ss.ta_T0) / ss.ta_V0
            B22 = ss.ta_qi0 / ss.ta_V0
            B23 = ss.ta_lambda_c / (ss.ta_rho * ss.ta_cp * ss.ta_V0)

            def du(t):
                if t < ss.ta_t_step:
                    return np.array([0.0, 0.0, 0.0])
                return np.array([ss.ta_dqi, ss.ta_dTi, ss.ta_dqc])
            
            def f_lin(t, dx):
                dV, dT = dx
                dqi_t, dTi_t, dqc_t = du(t)

                ddVdt = A11 * dV + B11 * dqi_t
                ddTdt = A21 * dV + A22 * dT + B21 * dqi_t + B22 * dTi_t + B23 * dqc_t
                return [ddVdt, ddTdt]
            
            t_eval = np.linspace(ss.ta_t0, ss.ta_tf, 2000)

            # Não linear: inicia no equilíbrio
            sol_nl = solve_ivp(f_nl, (ss.ta_t0, ss.ta_tf), [ss.ta_V0, ss.ta_T0], t_eval=t_eval, rtol=1e-7, atol=1e-9)

            # Linear: inicia em Δx = 0
            sol_lin = solve_ivp(f_lin, (ss.ta_t0, ss.ta_tf), [0.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-12)

            V_nl, T_nl = sol_nl.y
            dV_lin, dT_lin = sol_lin.y
            V_lin = ss.ta_V0 + dV_lin
            T_lin = ss.ta_T0 + dT_lin

            t = sol_nl.t

            # Gráfico mecânico (V)
            fig_V = go.Figure()
            fig_V.add_trace(go.Scatter(x=t, y=V_nl, mode="lines", name="Não linear: V(t)"))
            fig_V.add_trace(go.Scatter(x=t, y=V_lin, mode="lines", name="Linearizado: V(t)"))
            fig_V.add_vline(x=ss.ta_t_step, line_dash="dash")
            fig_V.update_layout(
                title="EDO mecânica: Volume V(t), não linear vs linearizado",
                xaxis_title="Tempo (s)",
                yaxis_title="Volume (m³)",
                legend_title="Modelo"
            )

            # Gráfico térmico (T)
            fig_T = go.Figure()
            fig_T.add_trace(go.Scatter(x=t, y=T_nl, mode="lines", name="Não linear: T(t)"))
            fig_T.add_trace(go.Scatter(x=t, y=T_lin, mode="lines", name="Linearizado: T(t)"))
            fig_T.add_vline(x=ss.ta_t_step, line_dash="dash")
            fig_T.update_layout(
                title="EDO térmica: Temperatura T(t), não linear vs linearizado",
                xaxis_title="Tempo (s)",
                yaxis_title="Temperatura (K)",
                legend_title="Modelo"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig_V)
            with col2:
                st.plotly_chart(fig_T)

            with st.container(border=True):
            st.title("Funções de Transferência")

            space_line()

            st.latex(
                r"G_{11}(s) = \frac{\Delta V(s)}{\Delta q_i(s)}"
                r" = \frac{1}{\,s + \frac{k}{2\sqrt{A}\sqrt{V_0}}\,}"
            )

            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # a = k/(2*sqrt(A)*sqrt(V0))
            a = ss.ta_k / (2.0 * np.sqrt(ss.ta_A) * np.sqrt(ss.ta_V0))

            # ΔV(t) com degrau em t_step
            dV = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero
            if abs(a) < 1e-15:
                dV[idx] = ss.ta_dqi * tau
            else:
                dV[idx] = (ss.ta_dqi / a) * (1.0 - np.exp(-a * tau))

            # Plot (Plotly)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dV,
                    mode="lines",
                    name="ΔV(t) (via G11, degrau em Δqi)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G11(s): ΔV(s)/Δqi(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔV(t) (m³)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            space_line()

            st.latex(
                r"G_{21}(s) = \frac{\Delta T(s)}{\Delta q_i(s)} = \frac{\dfrac{T_{i0} - T_0}{V_0}}{s + \dfrac{q_{i0}}{V_0}}"
            )

            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # b = qi0 / V0
            b = ss.ta_qi0 / ss.ta_V0

            # K = (Ti0 - T0) / V0
            K = (ss.ta_Ti0 - ss.ta_T0) / ss.ta_V0

            # ΔT(t) com degrau em t_step (entrada: Δqi)
            dT = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero (caso b == 0)
            if abs(b) < 1e-15:
                # Se b = 0, então G(s)=K/s, degrau -> rampa: ΔT(t)=K*Δqi*(t-t_step)
                dT[idx] = K * ss.ta_dqi * tau
            else:
                dT[idx] = (K * ss.ta_dqi / b) * (1.0 - np.exp(-b * tau))

            # Plot (Plotly)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dT,
                    mode="lines",
                    name="ΔT(t) (via G21, degrau em Δqi)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G21(s): ΔT(s)/Δqi(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔT(t) (K)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            space_line()
            
            st.latex(
                r"G_{22}(s) = \frac{\Delta T(s)}{\Delta T_i(s)}"
                r" = \frac{\dfrac{q_{i0}}{V_0}}{s + \dfrac{q_{i0}}{V_0}}"
            )
            
            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # b = qi0 / V0
            b = ss.ta_qi0 / ss.ta_V0

            # ΔT(t) com degrau em t_step (entrada: ΔTi)
            dT = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero (caso b == 0)
            if abs(b) < 1e-15:
                # Se b = 0, então G(s)=0, resposta fica zero
                dT[idx] = 0.0
            else:
                dT[idx] = ss.ta_dTi * (1.0 - np.exp(-b * tau))

            # Plot (Plotly)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dT,
                    mode="lines",
                    name="ΔT(t) (via G22, degrau em ΔTi)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G22(s): ΔT(s)/ΔTi(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔT(t) (K)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            space_line()

            st.latex(
                r"G_{23}(s) = \frac{\Delta T(s)}{\Delta q_c(s)}"
                r" = \frac{\dfrac{\lambda_c}{V_0 c_p}}{s + \dfrac{q_{i0}}{V_0}}"
            )

            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # b = qi0 / V0
            b = ss.ta_qi0 / ss.ta_V0

            # Kc = lambda_c / (V0 * cp)
            Kc = ss.ta_lambda_c / (ss.ta_V0 * ss.ta_cp)

            # ΔT(t) com degrau em t_step (entrada: Δqc)
            dT = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero
            if abs(b) < 1e-15:
                # caso degenerado: integrador
                dT[idx] = Kc * ss.ta_dqc * tau
            else:
                dT[idx] = (Kc * ss.ta_dqc / b) * (1.0 - np.exp(-b * tau))

            # Plot (Plotly)
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dT,
                    mode="lines",
                    name="ΔT(t) (via G23, degrau em Δqc)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G23(s): ΔT(s)/Δqc(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔT(t) (K)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)
        """)

    # Layout principal
    col1, col2, col3 = st.columns([11, 1, 1])

    with col3:
        if st.button("</>"):
            code()

    def inputs(t):
        """Retorna qi, Ti, qc (com degrau em t_step)."""
        if t < ss.ta_t_step:
            return ss.ta_qi0, ss.ta_Ti0, ss.ta_qc0
        return ss.ta_qi0 + ss.ta_dqi, ss.ta_Ti0 + ss.ta_dTi, ss.ta_qc0 + ss.ta_dqc
    
    # NÃO LINEAR: EDOs originais
    def f_nl(t, x):
        V, T = x
        ss.ta_qi, ss.ta_Ti, ss.ta_qc = inputs(t)

        # mecânica (Torricelli)
        qout = ss.ta_k * np.sqrt(max(V, 0.0) / ss.ta_A)
        dVdt = ss.ta_qi - qout

        # térmica
        Qterm = (ss.ta_qc * ss.ta_lambda_c) / (ss.ta_rho * ss.ta_cp)   # equivalente em (m^3/s)*K? -> entra como termo de aquecimento/ V

        # dT/dt = qi*(Ti - T)/V + Qterm/V
        dTdt = 0.0
        if V > 1e-9:
            dTdt = (ss.ta_qi * (ss.ta_Ti - T) + Qterm) / V

        return [dVdt, dTdt]
    
    # LINEARIZADO (em torno do equilíbrio)
    # Estados: ΔV, ΔT ; Entradas: Δqi, ΔTi, Δqc
    A11 = -ss.ta_k / (2.0 * np.sqrt(ss.ta_A) * np.sqrt(ss.ta_V0))
    A12 = 0.0

    # Para f2 = qi(Ti-T)/V + Qterm/V
    A22 = -ss.ta_qi0 / ss.ta_V0

    # A21 no equilíbrio zera (pois f2(V0,T0,u0)=0)
    # A21 = -( qi0*(Ti0-T0) + Qterm0 ) / V0^2 ; no equilíbrio isso dá 0
    Qterm0 = (ss.ta_qc0 * ss.ta_lambda_c) / (ss.ta_rho * ss.ta_cp)
    A21 = -(ss.ta_qi0 * (ss.ta_Ti0 - ss.ta_T0) + Qterm0) / (ss.ta_V0**2)

    # Matriz B (avaliada no ponto de operação)
    B11 = 1.0
    B12 = 0.0
    B13 = 0.0

    B21 = (ss.ta_Ti0 - ss.ta_T0) / ss.ta_V0
    B22 = ss.ta_qi0 / ss.ta_V0
    B23 = ss.ta_lambda_c / (ss.ta_rho * ss.ta_cp * ss.ta_V0)

    def du(t):
        """Δu(t) = [Δqi, ΔTi, Δqc] com degrau."""
        if t < ss.ta_t_step:
            return np.array([0.0, 0.0, 0.0])
        return np.array([ss.ta_dqi, ss.ta_dTi, ss.ta_dqc])
    
    def f_lin(t, dx):
        dV, dT = dx
        dqi_t, dTi_t, dqc_t = du(t)

        ddVdt = A11 * dV + B11 * dqi_t
        ddTdt = A21 * dV + A22 * dT + B21 * dqi_t + B22 * dTi_t + B23 * dqc_t
        return [ddVdt, ddTdt]
    
    t_eval = np.linspace(ss.ta_t0, ss.ta_tf, 2000)

    # Não linear: inicia no equilíbrio
    sol_nl = solve_ivp(f_nl, (ss.ta_t0, ss.ta_tf), [ss.ta_V0, ss.ta_T0], t_eval=t_eval, rtol=1e-7, atol=1e-9)

    # Linear: inicia em Δx = 0
    sol_lin = solve_ivp(f_lin, (ss.ta_t0, ss.ta_tf), [0.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-12)

    V_nl, T_nl = sol_nl.y
    dV_lin, dT_lin = sol_lin.y
    V_lin = ss.ta_V0 + dV_lin
    T_lin = ss.ta_T0 + dT_lin

    t = sol_nl.t

    # Gráfico mecânico (V)
    fig_V = go.Figure()
    fig_V.add_trace(go.Scatter(x=t, y=V_nl, mode="lines", name="Não linear: V(t)"))
    fig_V.add_trace(go.Scatter(x=t, y=V_lin, mode="lines", name="Linearizado: V(t)"))
    fig_V.add_vline(x=ss.ta_t_step, line_dash="dash")
    fig_V.update_layout(
        title="EDO mecânica: Volume V(t), não linear vs linearizado",
        xaxis_title="Tempo (s)",
        yaxis_title="Volume (m³)",
        legend_title="Modelo"
    )

    # Gráfico térmico (T)
    fig_T = go.Figure()
    fig_T.add_trace(go.Scatter(x=t, y=T_nl, mode="lines", name="Não linear: T(t)"))
    fig_T.add_trace(go.Scatter(x=t, y=T_lin, mode="lines", name="Linearizado: T(t)"))
    fig_T.add_vline(x=ss.ta_t_step, line_dash="dash")
    fig_T.update_layout(
        title="EDO térmica: Temperatura T(t), não linear vs linearizado",
        xaxis_title="Tempo (s)",
        yaxis_title="Temperatura (K)",
        legend_title="Modelo"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig_V)
    with col2:
        st.plotly_chart(fig_T)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("pages/images/tanque_encamisado.png")

        # Gráficos das funções de transferência
        with st.container(border=True):
            st.title("Funções de Transferência")

            space_line()

            st.latex(
                r"G_{11}(s) = \frac{\Delta V(s)}{\Delta q_i(s)}"
                r" = \frac{1}{\,s + \frac{k}{2\sqrt{A}\sqrt{V_0}}\,}"
            )

            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # a = k/(2*sqrt(A)*sqrt(V0))
            a = ss.ta_k / (2.0 * np.sqrt(ss.ta_A) * np.sqrt(ss.ta_V0))

            # ΔV(t) com degrau em t_step
            dV = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero
            if abs(a) < 1e-15:
                dV[idx] = ss.ta_dqi * tau
            else:
                dV[idx] = (ss.ta_dqi / a) * (1.0 - np.exp(-a * tau))

            # Plot (Plotly)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dV,
                    mode="lines",
                    name="ΔV(t) (via G11, degrau em Δqi)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G11(s): ΔV(s)/Δqi(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔV(t) (m³)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            space_line()

            st.latex(
                r"G_{21}(s) = \frac{\Delta T(s)}{\Delta q_i(s)} = \frac{\dfrac{T_{i0} - T_0}{V_0}}{s + \dfrac{q_{i0}}{V_0}}"
            )

            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # b = qi0 / V0
            b = ss.ta_qi0 / ss.ta_V0

            # K = (Ti0 - T0) / V0
            K = (ss.ta_Ti0 - ss.ta_T0) / ss.ta_V0

            # ΔT(t) com degrau em t_step (entrada: Δqi)
            dT = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero (caso b == 0)
            if abs(b) < 1e-15:
                # Se b = 0, então G(s)=K/s, degrau -> rampa: ΔT(t)=K*Δqi*(t-t_step)
                dT[idx] = K * ss.ta_dqi * tau
            else:
                dT[idx] = (K * ss.ta_dqi / b) * (1.0 - np.exp(-b * tau))

            # Plot (Plotly)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dT,
                    mode="lines",
                    name="ΔT(t) (via G21, degrau em Δqi)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G21(s): ΔT(s)/Δqi(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔT(t) (K)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            space_line()
            
            st.latex(
                r"G_{22}(s) = \frac{\Delta T(s)}{\Delta T_i(s)}"
                r" = \frac{\dfrac{q_{i0}}{V_0}}{s + \dfrac{q_{i0}}{V_0}}"
            )
            
            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # b = qi0 / V0
            b = ss.ta_qi0 / ss.ta_V0

            # ΔT(t) com degrau em t_step (entrada: ΔTi)
            dT = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero (caso b == 0)
            if abs(b) < 1e-15:
                # Se b = 0, então G(s)=0, resposta fica zero
                dT[idx] = 0.0
            else:
                dT[idx] = ss.ta_dTi * (1.0 - np.exp(-b * tau))

            # Plot (Plotly)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dT,
                    mode="lines",
                    name="ΔT(t) (via G22, degrau em ΔTi)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G22(s): ΔT(s)/ΔTi(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔT(t) (K)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            space_line()

            st.latex(
                r"G_{23}(s) = \frac{\Delta T(s)}{\Delta q_c(s)}"
                r" = \frac{\dfrac{\lambda_c}{V_0 c_p}}{s + \dfrac{q_{i0}}{V_0}}"
            )

            t = np.linspace(ss.ta_t0, ss.ta_tf, 1200)

            # b = qi0 / V0
            b = ss.ta_qi0 / ss.ta_V0

            # Kc = lambda_c / (V0 * cp)
            Kc = ss.ta_lambda_c / (ss.ta_V0 * ss.ta_cp)

            # ΔT(t) com degrau em t_step (entrada: Δqc)
            dT = np.zeros_like(t)
            idx = t >= ss.ta_t_step
            tau = t[idx] - ss.ta_t_step

            # Evita divisão por zero
            if abs(b) < 1e-15:
                # caso degenerado: integrador
                dT[idx] = Kc * ss.ta_dqc * tau
            else:
                dT[idx] = (Kc * ss.ta_dqc / b) * (1.0 - np.exp(-b * tau))

            # Plot (Plotly)
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=dT,
                    mode="lines",
                    name="ΔT(t) (via G23, degrau em Δqc)"
                )
            )

            fig.add_vline(x=ss.ta_t_step, line_dash="dash")

            fig.update_layout(
                title="Resposta ao degrau via G23(s): ΔT(s)/Δqc(s)",
                xaxis_title="Tempo (s)",
                yaxis_title="ΔT(t) (K)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:

        col3, col4 = st.columns([1, 1])

        with col3:
            # =========================
            # PARÂMETROS DO MODELO
            # =========================

            # Tempo total de simulação
            with st.container(border=True):
                st.markdown("Tempo Máximo")
                st.markdown(f"t_max = {ss.ta_tmax:.2f} s")
                ta_tmax = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_tmax"
                )
                if ta_tmax != ss.ta_tmax:
                    ss.ta_tmax = ta_tmax
                    st.rerun()

            # Área A
            with st.container(border=True):
                st.markdown("Área da seção transversal do tanque")
                st.markdown(f"A = {ss.ta_A:.4f} m²")
                ta_A = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_A"
                )
                if ta_A != ss.ta_A:
                    ss.ta_A = ta_A
                    st.rerun()

            # k (Torricelli)
            with st.container(border=True):
                st.markdown("Constante de escoamento (Lei de Torricelli)")
                st.markdown(f"k = {ss.ta_k:.4f} (m³/s)/√m")
                ta_k = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_k"
                )
                if ta_k != ss.ta_k:
                    ss.ta_k = ta_k
                    st.rerun()

            # rho
            with st.container(border=True):
                st.markdown("Densidade do fluido")
                st.markdown(f"ρ = {ss.ta_rho:.2f} kg/m³")
                ta_rho = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_rho"
                )
                if ta_rho != ss.ta_rho:
                    ss.ta_rho = ta_rho
                    st.rerun()

            # cp
            with st.container(border=True):
                st.markdown("Calor específico do fluido")
                st.markdown(f"cₚ = {ss.ta_cp:.2f} J/(kg·K)")
                ta_cp = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_cp"
                )
                if ta_cp != ss.ta_cp:
                    ss.ta_cp = ta_cp
                    st.rerun()

            # lambda_c
            with st.container(border=True):
                st.markdown("Calor latente de condensação")
                st.markdown(f"λ_c = {ss.ta_lambda_c:.2e} J/kg")
                ta_lambda_c = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_lambda_c"
                )
                if ta_lambda_c != ss.ta_lambda_c:
                    ss.ta_lambda_c = ta_lambda_c
                    st.rerun()

            # =========================
            # PONTO DE OPERAÇÃO
            # =========================

            # qi0
            with st.container(border=True):
                st.markdown("Vazão volumétrica de entrada no ponto de operação")
                st.markdown(f"qᵢ₀ = {ss.ta_qi0:.4f} m³/s")
                ta_qi0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_qi0"
                )
                if ta_qi0 != ss.ta_qi0:
                    ss.ta_qi0 = ta_qi0
                    st.rerun()

            # Ti0
            with st.container(border=True):
                st.markdown("Temperatura de entrada no ponto de operação")
                st.markdown(f"Tᵢ₀ = {ss.ta_Ti0:.2f} K")
                ta_Ti0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_Ti0"
                )
                if ta_Ti0 != ss.ta_Ti0:
                    ss.ta_Ti0 = ta_Ti0
                    st.rerun()

        with col4:
            # =========================
            # JANELA DE TEMPO / DEGRAU
            # =========================

            # t0
            with st.container(border=True):
                st.markdown("Instante inicial da aplicação do degrau")
                st.markdown(f"t₀ = {ss.ta_t0:.2f} s")
                ta_t0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_t0"
                )
                if ta_t0 != ss.ta_t0:
                    ss.ta_t0 = ta_t0
                    st.rerun()

            # tf
            with st.container(border=True):
                st.markdown("Instante final da janela de análise")
                st.markdown(f"t_f = {ss.ta_tf:.2f} s")
                ta_tf = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_tf"
                )
                if ta_tf != ss.ta_tf:
                    ss.ta_tf = ta_tf
                    st.rerun()

            # t_step
            with st.container(border=True):
                st.markdown("Instante de aplicação do degrau")
                st.markdown(f"t_step = {ss.ta_t_step:.2f} s")
                ta_t_step = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_t_step"
                )
                if ta_t_step != ss.ta_t_step:
                    ss.ta_t_step = ta_t_step
                    st.rerun()

            # =========================
            # PERTURBAÇÕES
            # =========================

            # dqi
            with st.container(border=True):
                st.markdown("Perturbação na vazão de entrada")
                st.markdown(f"Δqᵢ = {ss.ta_dqi:.4f} m³/s")
                ta_dqi = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="ta_dqi"
                )
                if ta_dqi != ss.ta_dqi:
                    ss.ta_dqi = ta_dqi
                    st.rerun()

            # dTi
            with st.container(border=True):
                st.markdown("Perturbação na temperatura de entrada")
                st.markdown(f"ΔTᵢ = {ss.ta_dTi:.2f} K")
                ta_dTi = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="ta_dTi"
                )
                if ta_dTi != ss.ta_dTi:
                    ss.ta_dTi = ta_dTi
                    st.rerun()

            # dqc
            with st.container(border=True):
                st.markdown("Perturbação na vazão mássica de vapor")
                st.markdown(f"Δq_c = {ss.ta_dqc:.6f} kg/s")
                ta_dqc = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="ta_dqc"
                )
                if ta_dqc != ss.ta_dqc:
                    ss.ta_dqc = ta_dqc
                    st.rerun()

            # V0
            with st.container(border=True):
                st.markdown("Volume do fluido no ponto de operação")
                st.markdown(f"V₀ = {ss.ta_V0:.6f} m³")
                ta_V0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_V0"
                )
                if ta_V0 != ss.ta_V0:
                    ss.ta_V0 = ta_V0
                    st.rerun()

            # T0
            with st.container(border=True):
                st.markdown("Temperatura do fluido no ponto de operação")
                st.markdown(f"T₀ = {ss.ta_T0:.2f} K")
                ta_T0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    min_value=0.0,
                    key="ta_T0"
                )
                if ta_T0 != ss.ta_T0:
                    ss.ta_T0 = ta_T0
                    st.rerun()

            # qc0
            with st.container(border=True):
                st.markdown("Vazão mássica de vapor no ponto de operação")
                st.markdown(f"q_c0 = {ss.ta_qc0:.6f} kg/s")
                ta_qc0 = st.number_input(
                    "",
                    label_visibility="hidden",
                    key="ta_qc0"
                )
                if ta_qc0 != ss.ta_qc0:
                    ss.ta_qc0 = ta_qc0
                    st.rerun()

    # Linha divisória
    space_line()

    # Textos explicativos
    st.markdown("""
        O sistema de tanque com aquecimento modelado representa um processo dinâmico termo-hidráulico no qual a evolução do volume e da temperatura do fluido armazenado ocorre de forma acoplada. 
        Sua dinâmica resulta da interação entre os fenômenos de escoamento do líquido e de troca de energia térmica, que atuam simultaneamente ao longo do tempo.

        No aspecto hidráulico, o volume de fluido no interior do tanque varia de acordo com o balanço entre a vazão de entrada e a vazão de saída. 
        A vazão de saída é governada pelo escoamento gravitacional, descrito pela Lei de Torricelli, sendo proporcional à raiz quadrada do nível do líquido no tanque. 
        Esse comportamento introduz uma não linearidade natural no sistema, pois pequenas variações no volume alteram o nível e, consequentemente, a taxa de descarga. 
        O resultado é uma dinâmica de primeira ordem para o volume, na qual o tanque tende a atingir um regime estacionário quando a vazão de saída se iguala à vazão de entrada.

        No aspecto térmico, a temperatura do fluido varia em função do balanço de energia no tanque. 
        A energia armazenada depende diretamente da massa de fluido presente, que é proporcional ao volume, e da temperatura do líquido. 
        O fluido que entra no tanque transporta energia associada à sua temperatura de entrada, enquanto o fluido que sai remove energia à temperatura do próprio tanque, assumindo mistura perfeita. 
        Além disso, o sistema recebe calor por meio da condensação de vapor na camisa, cuja contribuição térmica é proporcional à vazão mássica de vapor condensado e ao calor latente de condensação.

        A dinâmica térmica é fortemente influenciada pelo volume do tanque, que atua como uma capacidade térmica efetiva. 
        Volumes maiores tornam a variação de temperatura mais lenta, pois a mesma quantidade de energia é distribuída em uma massa maior de fluido. 
        Por outro lado, variações na vazão de entrada ou na temperatura de entrada alteram diretamente a taxa com que a temperatura do tanque é puxada em direção a novos valores de equilíbrio.

        O acoplamento entre as dinâmicas hidráulica e térmica ocorre de forma assimétrica. 
        O volume influencia a evolução da temperatura ao determinar a capacidade térmica do sistema, mas a temperatura não afeta diretamente o balanço de massa. 
        Esse acoplamento unidirecional é típico de processos reais, nos quais fenômenos hidráulicos e térmicos coexistem, mas com diferentes níveis de influência mútua.

        Em torno de um ponto de operação estacionário, o sistema pode ser linearizado, resultando em duas equações diferenciais lineares de primeira ordem. 
        A dinâmica do volume permanece independente da temperatura, enquanto a dinâmica da temperatura depende tanto das perturbações térmicas quanto das variações de vazão e volume. 
        Essa forma linearizada permite analisar o comportamento do sistema para pequenas perturbações, facilitando a comparação entre o modelo não linear real e sua aproximação linear.

        A resposta dinâmica do tanque com aquecimento apresenta comportamento típico de processos industriais reais, com respostas suaves, não oscilatórias e convergência assintótica para o regime permanente. 
        Após a aplicação de degraus nas entradas, o volume e a temperatura evoluem com escalas de tempo distintas, refletindo a separação natural entre fenômenos hidráulicos e térmicos. 
        Esse comportamento ilustra de forma clara como balanços de massa e energia se combinam para descrever a dinâmica de sistemas de processo contínuos.
    """)

    space_line()

    # Sugestões
    st.markdown("""
        Ao alterar os parâmetros do tanque com aquecimento, os gráficos de V(t) e T(t), tanto no modelo não linear quanto no linearizado, respondem de forma coerente com a física do sistema,
        mostrando como cada termo dos balanços de massa e energia influencia diretamente a dinâmica termo-hidráulica:

        Área da seção transversal (A): ao aumentar A, para um mesmo volume V o nível h = V/A diminui, o que reduz a vazão de saída associada ao termo de Torricelli. Isso torna a dinâmica do volume mais lenta,
        pois o tanque “escoa menos” para o mesmo V. Além disso, como V0 depende de A, mudanças em A alteram o ponto de operação e podem deslocar o equilíbrio de volume e o comportamento observado.
        Valores menores de A aumentam o nível para o mesmo volume, elevam a vazão de saída e tornam V(t) mais “rápido” para retornar ao regime.

        Constante de Torricelli (k): aumentar k intensifica a vazão de saída para um mesmo nível, fazendo o tanque drenar mais rapidamente. Na prática, isso reduz o volume em regime permanente para uma mesma vazão de entrada,
        e também reduz a constante de tempo hidráulica, deixando a curva V(t) mais ágil. Valores menores de k “seguram” o escoamento, elevam o volume de equilíbrio e tornam a resposta do volume mais lenta.

        Densidade do fluido (ρ): a densidade aparece principalmente no balanço de energia, pois altera a capacidade térmica total do sistema (massa armazenada). Aumentar ρ faz o fluido armazenar mais energia por unidade de volume,
        o que tende a deixar T(t) mais lenta para responder a perturbações térmicas, enquanto reduzir ρ torna a temperatura mais sensível e mais rápida para variar, para o mesmo V.

        Calor específico (cₚ): o efeito de cₚ é semelhante ao de ρ: valores maiores aumentam a capacidade térmica efetiva do tanque, amortecendo variações de temperatura, então T(t) muda mais devagar.
        Valores menores reduzem a “inércia térmica”, fazendo a temperatura responder mais rapidamente às mudanças em qᵢ, Tᵢ e q_c.

        Calor latente de condensação (λ_c): aumentar λ_c amplifica o efeito de aquecimento associado ao vapor para uma mesma vazão de condensação q_c, elevando a contribuição de calor e tornando T(t) mais sensível a Δq_c.
        Valores menores de λ_c reduzem a potência térmica injetada pela condensação, deixando o aquecimento menos efetivo.

        Vazão de entrada no ponto de operação (qᵢ₀): aumentar qᵢ₀ eleva o volume de equilíbrio, pois o tanque precisa de um nível maior para que a vazão de saída iguale a entrada.
        Em termos térmicos, qᵢ₀ também atua como um “puxão” mais forte da temperatura em direção à temperatura de entrada, acelerando o retorno ao regime em T(t). Reduzir qᵢ₀ tende a diminuir V0 e enfraquecer esse efeito de renovação térmica.

        Temperatura de entrada no ponto de operação (Tᵢ₀): ao elevar Tᵢ₀, a alimentação passa a carregar mais energia para dentro do tanque, aumentando a tendência de elevação de T(t).
        Se Tᵢ₀ for menor, o fluido de entrada atua como um resfriamento, puxando a temperatura para baixo. Esse parâmetro afeta principalmente o valor de equilíbrio térmico e o sentido do transitório.

        Volume no ponto de operação (V₀): aumentar V₀, mantendo o restante, aumenta a massa armazenada e faz a temperatura variar mais lentamente, pois o termo de capacidade térmica fica maior.
        Além disso, V₀ também influencia as constantes de tempo do sistema linearizado. Valores menores de V₀ deixam T(t) mais “nervosa”, com variações mais rápidas, e deixam a comparação não linear vs linear mais sensível a perturbações maiores.

        Temperatura no ponto de operação (T₀): alterar T₀ muda o ponto de equilíbrio ao redor do qual você observa as perturbações. Em geral, T₀ atua como referência do regime permanente,
        então mudar T₀ desloca o patamar final esperado e altera os termos de linearização que dependem de diferenças como (Tᵢ₀ - T₀).

        Vazão de condensação no ponto de operação (q_c0): q_c0 define a “carga térmica base” fornecida pelo vapor. Tornar q_c0 mais positivo aumenta o aquecimento constante e eleva o patamar de temperatura de equilíbrio,
        enquanto valores menores reduzem o aporte de calor. Esse parâmetro é o equivalente térmico de um “aquecedor” ligado mais forte ou mais fraco.

        Perturbação em vazão (Δqᵢ): um degrau positivo em Δqᵢ tende a aumentar V(t) até um novo equilíbrio hidráulico. Na temperatura, Δqᵢ pode tanto aquecer quanto resfriar, dependendo de Tᵢ em relação a T.
        Se Tᵢ for maior que T, aumentar a vazão de entrada acelera o aquecimento; se Tᵢ for menor, acelera o resfriamento. Em geral, degraus grandes tornam a diferença entre o não linear e o linearizado mais visível.

        Perturbação em temperatura de entrada (ΔTᵢ): aumentar ΔTᵢ injeta energia extra via alimentação, elevando T(t) sem alterar diretamente o balanço de volume.
        É uma forma “limpa” de observar a dinâmica térmica, pois V(t) praticamente mantém o comportamento hidráulico enquanto T(t) reage de forma clara.

        Perturbação em condensação (Δq_c): aumentar Δq_c aumenta diretamente o calor fornecido pelo vapor, elevando T(t) sem afetar V(t).
        Esse degrau mostra bem o papel do termo de aquecimento por condensação, e também evidencia quando a aproximação linear é válida: quanto maior o degrau, maior a chance de o não linear se afastar do linearizado.

        Tempo inicial, final e instante do degrau (t₀, t_f, t_step): esses parâmetros não mudam a física do sistema, apenas controlam quando a perturbação acontece e quanto da trajetória você enxerga nos gráficos.
        Um t_f maior permite observar o retorno completo ao regime permanente, enquanto t_step desloca o momento em que as curvas mudam de patamar.

        Por fim, a comparação entre os modelos não linear e linearizado tende a ser excelente para pequenas perturbações, já que a linearização aproxima bem a dinâmica em torno do equilíbrio.
        À medida que os degraus aumentam ou o ponto de operação fica incoerente com os demais parâmetros, a diferença entre as curvas cresce, o que é esperado, pois o termo de Torricelli e a divisão por V tornam o sistema naturalmente não linear.
    """)