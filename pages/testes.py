# Databricks notebook source
# MAGIC %md
# MAGIC # TANQUE AQUECIDO

# COMMAND ----------

"""
Tanque com aquecimento — simulação não-linear e linearizada
Equações (conformes a imagem fornecida):

1) Balanço de massa:
   dV/dt = qi - k * sqrt(h),   com h = V / A

2) Balanço de energia:
   rho * V * cp * dT/dt = rho * qi * cp * (Ti - T) + rho_c * qc * lambda_c

Estados:
   x = [V, T]

Entradas (variáveis de entrada do sistema):
   u = [qi, Ti, qc]   (vazão de entrada, temp entrada, taxa de condensação de vapor)

Objetivo:
  - calcular ponto de operação (x*, u*)
  - calcular A = df/dx |_{x*,u*}, B = df/du |_{x*,u*} por diferenças finitas
  - simular não-linear com step em qc
  - simular linearizado com o mesmo step e comparar T(t)
"""

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# ------------------------------
# 1) Parâmetros físicos (ajuste aqui)
# ------------------------------
rho = 1000.0         # kg/m^3
cp = 4184.0          # J/(kg K)
rho_c = 1000.0       # densidade do condensado ~ água (kg/m^3)
lambda_c = 2.26e6    # J/kg (calor latente da água) - ajuste se quiser
A = 1.0              # m^2 área da seção do tanque
k = 0.05             # coeficiente (m^3/(s * sqrt(m))) -> qout = k * sqrt(h)

# entrada nominal (ponto de operação)
qi_nom = 0.002       # m^3/s (vazão líquida de entrada)
Ti_nom = 293.0       # K (temperatura da entrada)
qc_nom = 0.0005      # kg/s (taxa de condensado / condensação que entrega calor)

# Monte vetores de entrada nominal
u_nom = np.array([qi_nom, Ti_nom, qc_nom])

# ------------------------------
# 2) Definir o modelo não-linear f(x,u)
#    x = [V, T]; u = [qi, Ti, qc]
# ------------------------------
def f_nl(t, x, u):
    V, T = x
    qi, Ti, qc = u

    # proteção: V não negativo
    V = max(V, 1e-12)

    # geometria
    h = V / A

    # balanço de massa
    dVdt = qi - k * np.sqrt(max(h, 0.0))

    # balanço de energia:
    # rho * V * cp * dT/dt = rho * qi * cp * (Ti - T) + rho_c * qc * lambda_c
    dTdt = (rho * qi * cp * (Ti - T) + rho_c * qc * lambda_c) / (rho * V * cp)

    return np.array([dVdt, dTdt])

# ------------------------------
# 3) Calcular ponto de operação (x*, u*)
#    - a partir de u_nom
#    - dV/dt = 0 --> qi = k * sqrt(h*)  => h* = (qi/k)^2
#    - V* = A * h*
#    - T* = Ti + (rho_c*qc*lambda_c) / (rho*qi*cp)
# ------------------------------
def compute_steady_state(u):
    qi, Ti, qc = u
    # cuidado se qi == 0 (não há solução física com V*>0 se qc>0 porque V* -> infinito)
    if qi <= 0:
        raise ValueError("qi deve ser > 0 para existir estado estacionário com V*>0")
    h_star = (qi / k) ** 2          # h* = (qi/k)^2
    V_star = A * h_star
    T_star = Ti + (rho_c * qc * lambda_c) / (rho * qi * cp)
    return np.array([V_star, T_star])

x_star = compute_steady_state(u_nom)
print("Ponto de operação (x*): V* = {:.6f} m^3, T* = {:.4f} K".format(x_star[0], x_star[1]))

# ------------------------------
# 4) Linearização numérica por diferenças finitas
#    A = df/dx, B = df/du  (avaliadas em (x*, u_nom))
# ------------------------------
def numerical_jacobians(f, x0, u0, eps=1e-6):
    n = len(x0)
    m = len(u0)
    f0 = f(0, x0, u0)

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    # variação nos estados
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps * max(1.0, abs(x0[i]))
        f1 = f(0, x0 + dx, u0)
        A[:, i] = (f1 - f0) / dx[i]

    # variação nas entradas
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps * max(1.0, abs(u0[j]))
        f1 = f(0, x0, u0 + du)
        B[:, j] = (f1 - f0) / du[j]

    return A, B

A_mat, B_mat = numerical_jacobians(f_nl, x_star, u_nom)
np.set_printoptions(precision=6, suppress=True)
print("\nMatriz A (df/dx) no ponto de operação:\n", A_mat)
print("\nMatriz B (df/du) no ponto de operação:\n", B_mat)

# ------------------------------
# 5) Montar e simular:
#    - Simulação não-linear: integra f_nl com u(t) (degrau em qc)
#    - Simulação linear: integra dx/dt = A (x-x*) + B (u(t)-u*)
# ------------------------------

# Tempo de simulação
t0, tf = 0.0, 21.0         # 1 hora em segundos (ajuste se quiser)
t_eval = np.linspace(t0, tf, 10000)

# Defina uma perturbação: degrau em qc (incremento delta_qc aplicado em t>=0)
delta_qc = 0.0005   # kg/s (por exemplo, dobrar qc)
def u_time(t):
    # aqui mantemos qi e Ti constantes, e aplicamos degrau em qc
    qi = qi_nom
    Ti = Ti_nom
    qc = qc_nom + delta_qc   # passo permanente
    return np.array([qi, Ti, qc])

# 5a) Simular sistema não-linear
def nl_rhs(t, x):
    u = u_time(t)
    return f_nl(t, x, u)

x0 = x_star.copy()   # iniciar no ponto de operação antes do degrau (instantâneo)
sol_nl = solve_ivp(nl_rhs, (t0, tf), x0, t_eval=t_eval, method='RK45', atol=1e-8, rtol=1e-6)

# 5b) Simular sistema linearizado (estado deslocamento dx = x - x*)
# dx/dt = A * dx + B * du  ; du = u(t) - u_nom
def linear_rhs(t, dx):
    u = u_time(t)
    du = u - u_nom
    return A_mat.dot(dx) + B_mat.dot(du)

dx0 = np.zeros_like(x_star)  # começa no equilíbrio (antes do passo)
sol_lin = solve_ivp(linear_rhs, (t0, tf), dx0, t_eval=t_eval, method='RK45', atol=1e-10, rtol=1e-8)

# recuperar x_lin = x_star + dx
x_lin = (x_star.reshape(-1,1) + sol_lin.y)

# ------------------------------
# 6) Plot: comparar Temperatura T(t) do modelo não-linear e linearizado
# ------------------------------
T_nl = sol_nl.y[1, :]
T_lin = x_lin[1, :]

fig = go.Figure()
fig.add_trace(go.Scatter(x=sol_nl.t, y=T_nl, mode='lines', name='T(t) - não-linear'))
fig.add_trace(go.Scatter(x=sol_lin.t, y=T_lin, mode='lines', name='T(t) - linearizado', line=dict(dash='dash')))

fig.update_layout(
    title="Comparação: Temperatura T(t) — não-linear vs linearizado\n(degrau em qc = +{:.4f} kg/s)".format(delta_qc),
    xaxis_title="Tempo (s)",
    yaxis_title="Temperatura (K)",
    legend=dict(x=0.01, y=0.99)
)

# Mostra também volume para diagnóstico (opcional)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sol_nl.t, y=sol_nl.y[0,:], mode='lines', name='V(t) - não-linear'))
fig2.add_trace(go.Scatter(x=sol_lin.t, y=x_lin[0,:], mode='lines', name='V(t) - linearizado', line=dict(dash='dash')))
fig2.update_layout(title="Volume V(t) — comparação", xaxis_title="Tempo (s)", yaxis_title="Volume (m^3)")

# Impressões informativas
print("\nPerturbação aplicada: delta_qc = {:.6f} kg/s".format(delta_qc))
print("Estado inicial x* = ", x_star)
print("Entrada nominal u* = ", u_nom)

# Exibir figuras
fig.show()
fig2.show()

# ------------------------------
# 7) Observações sobre interpretação:
# ------------------------------
# - O código assume que o degrau em qc é aplicado instantaneamente no t=0.
# - O modelo linearizado é válido localmente: quanto maior for delta_qc,
#   maior a diferença entre respostas linear e não-linear.
# - Se quiser testar outro tipo de perturbação (ex: passo em qi ou Ti),
#   altere u_time(t) adequadamente e reexecute.
# - Você pode ajustar eps na função numerical_jacobians para melhorar precisão
#   numérica das derivadas parciais.


# COMMAND ----------

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# =========================
# 1) PARÂMETROS DO MODELO
# =========================
A = 1.0          # m^2  (área seção transversal do tanque)
k = 0.15         # (m^3/s)/sqrt(m)  (Torricelli ajustado p/ sua unidade de V e A)
rho = 1000.0     # kg/m^3
cp = 4180.0      # J/(kg.K)

# Vapor/condensação:
lambda_c = 2.26e6  # J/kg (calor latente)
rho_c = 1000.0     # kg/m^3 (densidade do condensado, se usar q_c volumétrico)

# Escolha 1:
QC_IS_MASS = True  # True: q_c é vazão mássica (kg/s). False: q_c é volumétrica (m^3/s)

# =========================
# 2) PONTO DE OPERAÇÃO (equilíbrio)
# =========================
qi0 = 0.10      # m^3/s
Ti0 = 300.0     # K
V0  = A * ( (qi0 / k) ** 2 )  # de 0 = qi0 - k*sqrt(V0/A) => V0 = A*(qi0/k)^2

# Para garantir equilíbrio térmico, escolha qc0 que zere dT/dt em T0:
T0 = 310.0      # K (você escolhe o T0 de operação)
# Equilíbrio térmico: 0 = qi0*(Ti0 - T0)/V0 + Qterm/V0
# Qterm = (qc0*lambda_c)/(rho*cp) se qc0 mássico; ou (rho_c*qc0*lambda_c)/(rho*cp) se volumétrico
if QC_IS_MASS:
    qc0 = - rho*cp * qi0 * (Ti0 - T0) / lambda_c     # kg/s
else:
    qc0 = - rho*cp * qi0 * (Ti0 - T0) / (rho_c*lambda_c)  # m^3/s

# =========================
# 3) PERTURBAÇÕES (degrau) nas entradas
# =========================
t0, tf = 0.0, 500.0
t_step = 0.0

dqi = 0.02      # m^3/s
dTi = 5.0       # K
dqc = 0.0       # (kg/s) ou (m^3/s), conforme QC_IS_MASS

def inputs(t):
    """Retorna qi, Ti, qc (com degrau em t_step)."""
    if t < t_step:
        return qi0, Ti0, qc0
    return qi0 + dqi, Ti0 + dTi, qc0 + dqc

# =========================
# 4) NÃO LINEAR: EDOs originais
# =========================
def f_nl(t, x):
    V, T = x
    qi, Ti, qc = inputs(t)

    # mecânica (Torricelli)
    qout = k * np.sqrt(max(V, 0.0) / A)
    dVdt = qi - qout

    # térmica
    if QC_IS_MASS:
        Qterm = (qc * lambda_c) / (rho * cp)   # equivalente em (m^3/s)*K? -> entra como termo de aquecimento/ V
    else:
        Qterm = (rho_c * qc * lambda_c) / (rho * cp)

    # dT/dt = qi*(Ti - T)/V + Qterm/V
    dTdt = 0.0
    if V > 1e-9:
        dTdt = (qi * (Ti - T) + Qterm) / V

    return [dVdt, dTdt]

# =========================
# 5) LINEARIZADO (em torno do equilíbrio)
#    Estados: ΔV, ΔT ; Entradas: Δqi, ΔTi, Δqc
# =========================
# A11 = -k/(2*sqrt(A)*sqrt(V0))
A11 = -k / (2.0 * np.sqrt(A) * np.sqrt(V0))
A12 = 0.0

# Para f2 = qi(Ti-T)/V + Qterm/V
# A22 = -qi0/V0
A22 = -qi0 / V0

# A21 no equilíbrio zera (pois f2(V0,T0,u0)=0), mas deixo calculável se quiser:
# A21 = -( qi0*(Ti0-T0) + Qterm0 ) / V0^2 ; no equilíbrio isso dá 0
if QC_IS_MASS:
    Qterm0 = (qc0 * lambda_c) / (rho * cp)
else:
    Qterm0 = (rho_c * qc0 * lambda_c) / (rho * cp)
A21 = -(qi0 * (Ti0 - T0) + Qterm0) / (V0**2)  # deve dar ~0

# Matriz B (avaliada no ponto de operação)
B11 = 1.0
B12 = 0.0
B13 = 0.0

B21 = (Ti0 - T0) / V0
B22 = qi0 / V0
if QC_IS_MASS:
    B23 = lambda_c / (rho * cp * V0)
else:
    B23 = (rho_c * lambda_c) / (rho * cp * V0)

def du(t):
    """Δu(t) = [Δqi, ΔTi, Δqc] com degrau."""
    if t < t_step:
        return np.array([0.0, 0.0, 0.0])
    return np.array([dqi, dTi, dqc])

def f_lin(t, dx):
    dV, dT = dx
    dqi_t, dTi_t, dqc_t = du(t)

    ddVdt = A11 * dV + B11 * dqi_t
    ddTdt = A21 * dV + A22 * dT + B21 * dqi_t + B22 * dTi_t + B23 * dqc_t
    return [ddVdt, ddTdt]

# =========================
# 6) SIMULAÇÕES
# =========================
t_eval = np.linspace(t0, tf, 2000)

# Não linear: inicia no equilíbrio
sol_nl = solve_ivp(f_nl, (t0, tf), [V0, T0], t_eval=t_eval, rtol=1e-7, atol=1e-9)

# Linear: inicia em Δx = 0
sol_lin = solve_ivp(f_lin, (t0, tf), [0.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-12)

V_nl, T_nl = sol_nl.y
dV_lin, dT_lin = sol_lin.y
V_lin = V0 + dV_lin
T_lin = T0 + dT_lin

t = sol_nl.t

# =========================
# 7) PLOTLY: 2 FIGURAS
# =========================
# Gráfico mecânico (V)
fig_V = go.Figure()
fig_V.add_trace(go.Scatter(x=t, y=V_nl, mode="lines", name="Não linear: V(t)"))
fig_V.add_trace(go.Scatter(x=t, y=V_lin, mode="lines", name="Linearizado: V(t)"))
fig_V.add_vline(x=t_step, line_dash="dash")
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
fig_T.add_vline(x=t_step, line_dash="dash")
fig_T.update_layout(
    title="EDO térmica: Temperatura T(t), não linear vs linearizado",
    xaxis_title="Tempo (s)",
    yaxis_title="Temperatura (K)",
    legend_title="Modelo"
)

# Exibir (em notebook/Streamlit você usa st.plotly_chart)
fig_V.show()
fig_T.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # EDO artigo

# COMMAND ----------

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# Parâmetros (ajuste aqui)
# =========================
J   = 0.01      # inércia [kg*m^2]
b   = 0.1       # atrito viscoso [N*m*s]
k1  = 0.05      # constante de torque [N*m/A]
L   = 0.5       # indutância [H]
E   = 12.0      # tensão aplicada [V]
R0  = 2.0       # resistência em T0 [ohm]
alpha = 0.004   # coef. temp. resistência [1/K]
T0  = 293.15    # temperatura de referência [K] (20°C)
k2  = 0.05      # constante FEM (back-emf) [V*s/rad]
Ct  = 10.0      # capacitância térmica [J/K]
hA  = 1.5       # h*A [W/K]
T_inf = 293.15  # ambiente [K]

# =========================
# Sistema de EDOs
# y = [w, i, T]
# =========================
def motor_termo_odes(t, y):
    w, i, T = y

    R_T = R0 * (1.0 + alpha * (T - T0))          # R(T)
    dw_dt = (k1 * i - b * w) / J                  # J*w_dot = k1*i - b*w
    di_dt = (E - R_T * i - k2 * w) / L            # L*i_dot = E - R(T)*i - k2*w
    dT_dt = (i**2 * R_T - hA * (T - T_inf)) / Ct  # Ct*T_dot = i^2*R(T) - hA(T-Tinf)

    return [dw_dt, di_dt, dT_dt]

# =========================
# Condições iniciais e tempo
# =========================
w0 = 0.0          # rad/s
i0 = 0.0          # A
T_init = T_inf    # K

t0, tf = 0.0, 30.0
t_eval = np.linspace(t0, tf, 2000)

# =========================
# Resolver
# =========================
sol = solve_ivp(
    motor_termo_odes,
    (t0, tf),
    [w0, i0, T_init],
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

t = sol.t
w = sol.y[0]
i = sol.y[1]
T = sol.y[2]

# =========================
# Gráficos (3 figuras separadas)
# =========================
plt.figure()
plt.plot(t, w)
plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade angular ω [rad/s]")
plt.title("EDO mecânica: ω(t)")
plt.grid(True)

plt.figure()
plt.plot(t, i)
plt.xlabel("Tempo [s]")
plt.ylabel("Corrente i [A]")
plt.title("EDO elétrica: i(t)")
plt.grid(True)

plt.figure()
plt.plot(t, T)
plt.xlabel("Tempo [s]")
plt.ylabel("Temperatura T [K]")
plt.title("EDO térmica: T(t)")
plt.grid(True)

plt.show()


# COMMAND ----------

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# Parâmetros (ajuste aqui)
# =========================
J   = 0.01      # inércia [kg*m^2]
b   = 0.1       # atrito viscoso [N*m*s]
k1  = 0.05      # constante de torque [N*m/A]
L   = 0.5       # indutância [H]
E   = 12.0      # tensão aplicada [V] (valor nominal, agora o E(t) vai sobrescrever isso no di_dt)
R0  = 2.0       # resistência em T0 [ohm]
alpha = 0.004   # coef. temp. resistência [1/K]
T0  = 293.15    # temperatura de referência [K] (20°C)
k2  = 0.05      # constante FEM (back-emf) [V*s/rad]
Ct  = 10.0      # capacitância térmica [J/K]
hA  = 1.5       # h*A [W/K]
T_inf = 293.15  # ambiente [K]

# =========================
# Tempo
# =========================
t0, tf = 0.0, 30.0
t_eval = np.linspace(t0, tf, 2000)

# =========================
# Entrada aleatória E(t)
# =========================
E_min, E_max = 10.0, 14.0   # intervalo de tensão [V]
dt_E = 0.2                  # a cada dt_E s sorteia um novo valor
seed = 42                   # troque pra ter outra sequência

t_E = np.arange(t0, tf + dt_E, dt_E)
rng = np.random.default_rng(seed)
E_vals = rng.uniform(E_min, E_max, size=len(t_E))

def E_of_t(t):
    # sinal em degraus: pega o último valor sorteado antes de t
    idx = np.searchsorted(t_E, t, side="right") - 1
    idx = np.clip(idx, 0, len(E_vals) - 1)
    return E_vals[idx]

# =========================
# Sistema de EDOs
# y = [w, i, T]
# =========================
def motor_termo_odes(t, y):
    w, i, T = y

    R_T = R0 * (1.0 + alpha * (T - T0))           # R(T)
    dw_dt = (k1 * i - b * w) / J                   # J*w_dot = k1*i - b*w

    E_t = E_of_t(t)                                # <<< agora E varia no tempo
    di_dt = (E_t - R_T * i - k2 * w) / L           # L*i_dot = E(t) - R(T)*i - k2*w

    dT_dt = (i**2 * R_T - hA * (T - T_inf)) / Ct   # Ct*T_dot = i^2*R(T) - hA(T-Tinf)

    return [dw_dt, di_dt, dT_dt]

# =========================
# Condições iniciais
# =========================
w0 = 0.0
i0 = 0.0
T_init = T_inf

# =========================
# Resolver
# =========================
sol = solve_ivp(
    motor_termo_odes,
    (t0, tf),
    [w0, i0, T_init],
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

# =========================
# Gráficos (3 figuras separadas)
# =========================
plt.figure()
plt.plot(t, w)
plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade angular ω [rad/s]")
plt.title("EDO mecânica: ω(t)")
plt.grid(True)

plt.figure()
plt.plot(t, i)
plt.xlabel("Tempo [s]")
plt.ylabel("Corrente i [A]")
plt.title("EDO elétrica: i(t)")
plt.grid(True)

plt.figure()
plt.plot(t, T)
plt.xlabel("Tempo [s]")
plt.ylabel("Temperatura T [K]")
plt.title("EDO térmica: T(t)")
plt.grid(True)

# (Opcional) ver a entrada aleatória
plt.figure()
plt.step(t, E_plot, where="post")
plt.xlabel("Tempo [s]")
plt.ylabel("Tensão E(t) [V]")
plt.title("Entrada aleatória: E(t)")
plt.grid(True)

plt.show()


# COMMAND ----------

import numpy as np
import plotly.graph_objects as go

# =========================
# Parâmetros (ajuste aqui)
# =========================
k = 0.15        # (m^3/s)/sqrt(m)  (constante ajustada)
A = 1.0         # m^2
V0 = 1.0        # m^3  (volume de operação)

# Entrada: degrau em qi
q_step = 0.05   # m^3/s (amplitude de Δqi)

# Tempo de simulação
t_max = 50.0    # s
n = 1200
t = np.linspace(0, t_max, n)

# =========================
# G11(s) = 1/(s + a)
# a = k/(2*sqrt(A)*sqrt(V0))
# Resposta ao degrau: ΔV(t) = (q_step/a)*(1 - exp(-a t))
# =========================
a = k / (2.0 * np.sqrt(A) * np.sqrt(V0))

# Evita divisão por zero caso a seja 0
if a == 0:
    dV = q_step * t  # integrador puro: 1/s
else:
    dV = (q_step / a) * (1.0 - np.exp(-a * t))

# =========================
# Plot (Plotly)
# =========================
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=dV,
        mode="lines",
        name="ΔV(t) (degrau em Δqi)"
    )
)

fig.update_layout(
    title="Resposta ao degrau: G11(s) = ΔV(s)/Δqi(s)",
    xaxis_title="Tempo (s)",
    yaxis_title="ΔV(t) (m³)",
    hovermode="x unified"
)

fig.show()

# COMMAND ----------

from PIL import Image

# Abrir imagem
img = Image.open("images/motor_bomba_bcs.png")

# Tamanho original
width, height = img.size
print(f"Tamanho original: {width}x{height}")

# Fator de escala
scale = 2  # dobra o tamanho

# Novo tamanho
new_size = (width * scale, height * scale)

# Redimensionar com boa qualidade
img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

# Salvar nova imagem
img_resized.save("images/motor_bomba_bcs.png")

print(f"Novo tamanho: {new_size[0]}x{new_size[1]}")


# COMMAND ----------


