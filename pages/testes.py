# Databricks notebook source
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objs as go

# -----------------------------
# Parâmetros do motor
# -----------------------------
R = 3       # Resistência
K1 = 2      # Constante de torque
K2 = 6      # Constante de FEM
J = 2       # Inércia
B = 2       # Amortecimento
E0 = 10     # Tensão de entrada

# -----------------------------
# EDO do motor DC
# -----------------------------
def motor_dc_no_L(t, x):
    w, theta = x
    i = (E0 - K2 * w) / R
    dw_dt = (K1 * i - B * w) / J
    dtheta_dt = w
    return [dw_dt, dtheta_dt]

# -----------------------------
# Simulação
# -----------------------------
t0, tf = 0, 5
t_eval = np.linspace(t0, tf, 2000)
x0 = [0, 0]

sol = solve_ivp(motor_dc_no_L, (t0, tf), x0, t_eval=t_eval)

w = sol.y[0]
theta = sol.y[1]
t = sol.t

# -----------------------------
# Gráfico 1: θ(t)
# -----------------------------
fig_theta = go.Figure()
fig_theta.add_trace(go.Scatter(x=t, y=theta, mode='lines', name='θ(t)'))
fig_theta.update_layout(
    title='Posição Angular θ(t)',
    xaxis_title='Tempo (s)',
    yaxis_title='θ(t) [rad]',
    template='plotly_dark'
)

fig_theta.show()

# -----------------------------
# Gráfico 2: ω(t)
# -----------------------------
fig_w = go.Figure()
fig_w.add_trace(go.Scatter(x=t, y=w, mode='lines', name='ω(t)', line=dict(width=3)))
fig_w.update_layout(
    title='Velocidade Angular ω(t)',
    xaxis_title='Tempo (s)',
    yaxis_title='ω(t) [rad/s]',
    template='plotly_dark'
)

fig_w.show()

# COMMAND ----------

import numpy as np
import plotly.express as px

# Parâmetros do motor (exemplo)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Condições iniciais
theta0 = 0.5      # posição inicial [rad]
omega0 = 2.0      # velocidade inicial [rad/s]

# Tempo total
t_final = 2.0

# Constantes auxiliares
a = (k1 * k2) / R + b
tau = J / a
omega_ss = (k1 * E0 / R) / a

# Vetor de tempo
t = np.linspace(0, t_final, 1000)

# Soluções analíticas com θ0 e ω0
omega = omega_ss + (omega0 - omega_ss) * np.exp(-t / tau)

theta = (
    theta0
    + omega_ss * t
    + (omega0 - omega_ss) * tau * (np.exp(-t / tau) - 1)
)

# Gráfico em Plotly
fig = px.line(
    x=t,
    y=theta,
    labels={"x": "Tempo (s)", "y": "Posição angular θ(t) [rad]"},
    title="Posição angular do motor DC ao longo do tempo (com θ₀ e ω₀)"
)

fig.show()
import numpy as np
import plotly.express as px

# Parâmetros do motor (exemplo, ajuste para o seu caso)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Constante de tempo e velocidade em regime permanente
a = (k1 * k2) / R + b       # coeficiente da velocidade
tau = J / a                 # constante de tempo mecânica
omega_ss = (k1 * E0 / R) / a

# Vetor de tempo
t = np.linspace(0, t_final, 1000)

# Solução analítica para theta(t) com condição inicial theta(0) = 0
theta = omega_ss * (t - tau * (1 - np.exp(-t / tau)))

# Gráfico com plotly
fig = px.line(x=t, y=theta,
              labels={"x": "Tempo (s)", "y": "Posição angular θ(t) [rad]"},
              title="Posição angular do motor DC ao longo do tempo")
fig.show()


# COMMAND ----------


# Parâmetros do motor (exemplo, ajuste para o seu caso)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Constante de tempo e velocidade em regime permanente
a = (k1 * k2) / R + b       # coeficiente da velocidade no termo da EDO
tau = J / a                 # constante de tempo mecânica
omega_ss = (k1 * E0 / R) / a  # velocidade em regime permanente

# Vetor de tempo
t = np.linspace(0, t_final, 1000)

# Solução analítica para ω(t) com condição inicial ω(0) = 0
omega = omega_ss * (1 - np.exp(-t / tau))

# Gráfico com Plotly
fig = px.line(
    x=t,
    y=omega,
    labels={"x": "Tempo (s)", "y": "Velocidade angular ω(t) [rad/s]"},
    title="Velocidade angular do motor DC ao longo do tempo"
)

fig.show()

# COMMAND ----------

import numpy as np
import plotly.express as px

# Parâmetros do motor (exemplo, mesmo padrão)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Termos auxiliares
a = (k1 * k2) / R + b
tau = J / a
omega_ss = (k1 * E0 / R) / a

# Vetor de tempo
t_final = 2.0
t = np.linspace(0, t_final, 1000)

# Aceleração angular α(t)
alpha = (omega_ss / tau) * np.exp(-t / tau)

# Gráfico com Plotly
fig = px.line(
    x=t,
    y=alpha,
    labels={"x": "Tempo (s)", "y": "Aceleração angular α(t) [rad/s²]"},
    title="Aceleração angular do motor DC ao longo do tempo"
)

fig.show()


# COMMAND ----------

import numpy as np
import plotly.express as px

# Parâmetros do motor (exemplo)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Termos auxiliares
a = (k1 * k2) / R + b
tau = J / a
omega_ss = (k1 * E0 / R) / a

# Vetor de tempo
t_final = 2.0
t = np.linspace(0, t_final, 1000)

# Velocidade angular
omega = omega_ss * (1 - np.exp(-t / tau))

# Corrente elétrica
i = (E0 - k2 * omega) / R

# Gráfico com Plotly
fig = px.line(
    x=t,
    y=i,
    labels={"x": "Tempo (s)", "y": "Corrente i(t) [A]"},
    title="Corrente elétrica do motor DC ao longo do tempo"
)

fig.show()


# COMMAND ----------

import numpy as np
import plotly.express as px

# Parâmetros do motor (exemplo)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Termos auxiliares
a = (k1 * k2) / R + b
tau = J / a
omega_ss = (k1 * E0 / R) / a

# Vetor de tempo
t_final = 2.0
t = np.linspace(0, t_final, 1000)

# Velocidade angular
omega = omega_ss * (1 - np.exp(-t / tau))

# Corrente elétrica
i = (E0 - k2 * omega) / R

# Torque gerado
Tg = k1 * i

# Torque resistivo
Tf = b * omega

# Torque resultante
Tr = Tg - Tf

# Gráfico com Plotly
fig = px.line(
    x=t,
    y=Tr,
    labels={"x": "Tempo (s)", "y": "Torque resultante T_r(t) [N·m]"},
    title="Torque resultante do motor DC ao longo do tempo"
)

fig.show()


# COMMAND ----------

import numpy as np
import plotly.express as px

# Parâmetros do motor (exemplo)
J = 0.01      # inércia [kg.m²]
k1 = 0.1      # constante de torque
k2 = 0.1      # constante de f.c.e.m.
R = 1.0       # resistência [ohm]
b = 0.01      # atrito viscoso
E0 = 24.0     # degrau de tensão [V]

# Condições iniciais
theta0 = 0.0      # posição inicial [rad]
omega0 = 0.0      # velocidade inicial [rad/s]

# Tempo total
t_final = 2.0

# Constantes auxiliares
a = (k1 * k2) / R + b
tau = J / a
omega_ss = (k1 * E0 / R) / a

# Vetor de tempo
t = np.linspace(0, t_final, 1000)

# Soluções analíticas com θ0 e ω0
omega = omega_ss + (omega0 - omega_ss) * np.exp(-t / tau)

theta = (
    theta0
    + omega_ss * t
    + (omega0 - omega_ss) * tau * (np.exp(-t / tau) - 1)
)

# Gráfico em Plotly
fig = px.line(
    x=t,
    y=theta,
    labels={"x": "Tempo (s)", "y": "Posição angular θ(t) [rad]"},
    title="Posição angular do motor DC ao longo do tempo (com θ₀ e ω₀)"
)

fig.show()


# COMMAND ----------

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -----------------------------
# Parâmetros do motor (exemplo)
# -----------------------------
J  = 0.01   # inércia [kg.m²]
k1 = 0.1    # constante de torque
k2 = 0.1    # constante de f.c.e.m.
R  = 1.0    # resistência [ohm]
b  = 0.01   # atrito viscoso [N.m.s/rad]

# -----------------------------
# Tensão variável E(t)
# -----------------------------
def E_t(t):
    if t < 1.0:
        return 0.0
    elif t < 3.0:
        return 24.0
    else:
        return 24.0 + 6.0 * np.sin(2 * np.pi * t)

# -----------------------------
# Sistema de 1ª ordem
# x[0] = θ(t)        (posição angular)
# x[1] = dθ/dt(t)    (velocidade angular)
# -----------------------------
def motor_dc(t, x):
    theta  = x[0]
    omega  = x[1]          # dθ/dt
    E      = E_t(t)

    # Equação vinda da EDO original
    dtheta_dt = omega
    domega_dt = (k1 / (J * R)) * E - ((k1 * k2) / (R * J) + b / J) * omega

    return [dtheta_dt, domega_dt]

# -----------------------------
# Condições iniciais e integração
# -----------------------------
t0 = 0.0
tf = 5.0
x0 = [0.0, 0.0]   # θ(0) = 0 rad, ω(0) = 0 rad/s

t_eval = np.linspace(t0, tf, 1000)

sol = solve_ivp(
    motor_dc,
    (t0, tf),
    x0,
    t_eval=t_eval
)

t     = sol.t
theta = sol.y[0]
omega = sol.y[1]

# -----------------------------
# Plot da posição angular θ(t)
# -----------------------------
plt.figure()
plt.plot(t, theta)
plt.xlabel("Tempo [s]")
plt.ylabel("Posição angular θ(t) [rad]")
plt.title("Resposta da posição angular para uma tensão E(t) variável")
plt.grid(True)
plt.show()


# COMMAND ----------


