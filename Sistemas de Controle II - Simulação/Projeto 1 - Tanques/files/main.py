import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# Função para calcular as derivadas das alturas dos tanques
def derivadas_alturas(t, L1_dot, L2_dot, Kp, Vp, g, Aout1, Aout2, At1, At2):
    # Limita as derivadas das alturas para evitar que ultrapassem 30 cm
    max_height = 30  # Altura máxima de 30 cm (0.3 m)
    L1_dot = np.clip(L1_dot, -np.sqrt(2 * max_height * g), np.sqrt(2 * max_height * g))
    L2_dot = np.clip(L2_dot, -np.sqrt(2 * max_height * g), np.sqrt(2 * max_height * g))

    # Calcula as derivadas das alturas com as limitações
    L1_dot = (Kp * Vp - Aout1 * np.sqrt(2 * g * L1_dot)) / At1
    L2_dot = (Aout1 * np.sqrt(2 * g * L1_dot) - Aout2 * np.sqrt(2 * g * L2_dot)) / At2
    return L1_dot, L2_dot

# Parâmetros
Kp = 3.3  # Constante da bomba
Vp = 2.7  # Tensão da bomba
g = 981  # Aceleração da gravidade (cm/s^2)
Dout1 = 0.47625
Dout2 = 0.47625
Aout1 = (np.pi * (Dout1)**2) / 4  # Área de saída do tanque 1 (m^2)
Aout2 = (np.pi * (Dout2)**2) / 4  # Área de saída do tanque 2 (m^2)
At1 = np.pi * (4.445/2)**2 / 4  # Área interna do tanque 1 (m^2)
At2 = np.pi * (4.445/2)**2 / 4  # Área interna do tanque 2 (m^2)

# Função de atualização do sistema de controle do tanque 1
def tanque1_update(t, x, u, params):
    L1_dot, _ = derivadas_alturas(t, x[0], x[1], Kp, Vp, g, Aout1, Aout2, At1, At2)
    return [L1_dot, 0.0]

# Função de saída do sistema de controle do tanque 1
def tanque1_output(t, x, u, params):
    return x[0]

# Função de atualização do sistema de controle do tanque 2
def tanque2_update(t, x, u, params):
    _, L2_dot = derivadas_alturas(t, x[0], x[1], Kp, Vp, g, Aout1, Aout2, At1, At2)
    return [0.0, L2_dot]

# Função de saída do sistema de controle do tanque 2
def tanque2_output(t, x, u, params):
    return x[1]

# Parâmetros do sistema dos tanques
tanque1_params = {'A1': At1, 'Cd1': Dout1}
tanque2_params = {'A2': At2, 'Cd2': Dout2}

# Define o sistema de controle do primeiro tanque como um sistema de entrada/saída não linear
tanque1 = ctl.NonlinearIOSystem(
    tanque1_update, tanque1_output, states=2, name='tanque1',
    inputs=('u'), outputs=('h1'), params=tanque1_params)

# Define o sistema de controle do segundo tanque como um sistema de entrada/saída não linear
tanque2 = ctl.NonlinearIOSystem(
    tanque2_update, tanque2_output, states=2, name='tanque2',
    inputs=('u'), outputs=('h2'), params=tanque2_params)

# Simulação do sistema de controle dos tanques
t = np.linspace(0, 30, 1000)
u = np.ones(len(t)) * 1  # Entrada degrau

# Simulação do primeiro tanque
tout1, yout1 = ctl.input_output_response(tanque1, t, u)

# Simulação do segundo tanque
tout2, yout2 = ctl.input_output_response(tanque2, t, u)

# Plotagem dos resultados
plt.plot(tout1, yout1, label='Tanque 1')
plt.plot(tout2, yout2, label='Tanque 2')
plt.xlabel('Tempo (s)')
plt.ylabel('Altura (m)')
plt.title('Resposta ao degrau do sistema de controle dos tanques')
plt.legend()
plt.grid(True)
plt.show()
