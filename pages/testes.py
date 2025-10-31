# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
m = 1.0     # massa (kg)
k = 20.0    # constante da mola (N/m)
c = 2.0     # coeficiente de amortecimento (N·s/m)

# Cálculo de parâmetros derivados
wn = np.sqrt(k/m)               # frequência natural não amortecida (rad/s)
zeta = c / (2*np.sqrt(k*m))     # fator de amortecimento (adimensional)
wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else 0   # frequência amortecida (rad/s)

# Tempo
t = np.linspace(0, 10, 1000)    # tempo (s)

# Condições iniciais
z0 = 1.0       # posição inicial (m)
zdot0 = 0.0    # velocidade inicial (m/s)

# Solução (subamortecido)
if zeta < 1:
    A = z0                      # amplitude inicial (m)
    B = (zdot0 + zeta*wn*z0) / wd   # constante dependente das condições iniciais (m)
    z = np.exp(-zeta*wn*t) * (A*np.cos(wd*t) + B*np.sin(wd*t))   # posição (m)

# Criticamente amortecido
elif np.isclose(zeta, 1):
    A1 = z0                     # constante (m)
    A2 = zdot0 + wn*z0          # constante (m/s)
    z = (A1 + A2*t) * np.exp(-wn*t)   # posição (m)

# Superamortecido
else:
    s1 = -wn*(zeta - np.sqrt(zeta**2 - 1))   # raiz característica (1/s)
    s2 = -wn*(zeta + np.sqrt(zeta**2 - 1))   # raiz característica (1/s)
    A1 = (zdot0 - s2*z0)/(s1 - s2)           # constante (m)
    A2 = z0 - A1                             # constante (m)
    z = A1*np.exp(s1*t) + A2*np.exp(s2*t)    # posição (m)

# Plot
plt.figure(figsize=(8,4))
plt.plot(t, z, label=f'ζ={zeta:.2f}')
plt.title('Resposta temporal z(t) - Sistema massa-mola-amortecedor')
plt.xlabel('Tempo (s)')          # eixo x: tempo (s)
plt.ylabel('Posição z(t) (m)')   # eixo y: posição (m)
plt.grid(True)
plt.legend()
plt.show()

# COMMAND ----------


