# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do circuito RC
E = 5      # tensão da fonte (V)
R = 1000   # resistência (ohms)
C = 100e-6 # capacitância (F)
tau = R * C

# Vetor de tempo
t = np.linspace(0, 5*tau, 200)

# Equações
i = (E / R) * np.exp(-t / tau)              # Corrente
Vc = E * (1 - np.exp(-t / tau))             # Tensão no capacitor

# Plot 1: Corrente
plt.figure(figsize=(10,4))
plt.plot(t, i, color='red', label='i(t) = (ε/R)·e^(-t/RC)')
plt.axhline(E/R, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("Tempo (s)")
plt.ylabel("Corrente (A)")
plt.title("Corrente no circuito RC durante a carga do capacitor")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Tensão no capacitor
plt.figure(figsize=(10,4))
plt.plot(t, Vc, color='green', label='Vc(t) = ε·(1 - e^(-t/RC))')
plt.axhline(E, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("Tempo (s)")
plt.ylabel("Tensão no capacitor (V)")
plt.title("Tensão no capacitor durante a carga")
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------

import plotly.graph_objects as go

# --- Plot 1: Corrente ---
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
    y=E/R,
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

fig_i.show()


# --- Plot 2: Tensão no capacitor ---
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
    y=E,
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

fig_vc.show()


# COMMAND ----------

import numpy as np
import plotly.graph_objects as go

# parâmetros base
L = 1000e-3   # H
C = 10000e-6   # F
E = 100     # V
t = np.linspace(0, 4, 1000)

# três casos
R_sub = 5  # R² < 4L/C
R_crit = 20  # R² = 4L/C
R_super = 40 # R² > 4L/C

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
q_sub = rlc_response(R_sub, L, C, E, t)
q_crit = rlc_response(R_crit, L, C, E, t)
q_super = rlc_response(R_super, L, C, E, t)

# gráfico
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

fig.show()


# COMMAND ----------

import numpy as np
import plotly.graph_objects as go

# parâmetros coerentes
L = 1         # H
C = 100e-6    # F
E = 100       # V
t = np.linspace(0, 5, 2000)  # 5 s de simulação

# resistências para cada regime
R_sub  = 5
R_crit = 200
R_super = 500

def Vc_response(R, L, C, E, t):
    alpha = R / (2 * L)
    omega_0 = 1 / np.sqrt(L * C)
    disc = alpha**2 - omega_0**2

    if disc < 0:  # subamortecido
        omega_d = np.sqrt(omega_0**2 - alpha**2)
        Vc = E * (1 - np.exp(-alpha*t) * (np.cos(omega_d*t) + (alpha/omega_d)*np.sin(omega_d*t)))
    elif np.isclose(disc, 0):  # crítico
        Vc = E * (1 - np.exp(-alpha*t) * (1 + alpha*t))
    else:  # superamortecido
        r1 = -alpha + np.sqrt(disc)
        r2 = -alpha - np.sqrt(disc)
        A = E * r2 / (r2 - r1)
        B = -E * r1 / (r2 - r1)
        Vc = E * (1 - A*np.exp(r1*t) - B*np.exp(r2*t))
    return Vc

# respostas
Vc_sub = Vc_response(R_sub, L, C, E, t)
Vc_crit = Vc_response(R_crit, L, C, E, t)
Vc_super = Vc_response(R_super, L, C, E, t)

# gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Vc_sub, mode='lines', name='Subamortecido', line=dict(color='green')))
fig.add_trace(go.Scatter(x=t, y=Vc_crit, mode='lines', name='Criticamente Amortecido', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=t, y=Vc_super, mode='lines', name='Superamortecido', line=dict(color='red')))

fig.update_layout(
    title='Tensão no Capacitor Vc(t) em Circuitos RLC',
    xaxis_title='Tempo (s)',
    yaxis_title='Tensão no Capacitor (V)',
    template='plotly_white'
)

fig.show()


# COMMAND ----------


