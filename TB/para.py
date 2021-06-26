"""
Parameter definitions for the model
"""
import numpy as np

# time-tick: 10 min
k = 600 / 6

# Diffusion calc constant
Dk = 1

# Chemokine diffusion coefficient /0.1 min
diffC = 0.64 * Dk / 4

# Chemokine degradation coefficient /0.1 min
decayC = 0.001 * Dk

# Prob. of bacteria being killed within M_R
p_k = 2.36 * 0.01  # 8.51

# Prob. T cell kills a macrophage
pT_k = 3.61 * 0.01  # 6.31

# Percentage of B_I being destroyed when killed
P_kill = 50 * 0.01

# Capacity of extracellular bacteria
K_BE = 200

# Intracellular growth rate of bacteria /min
alpha_BI = np.linspace(0.0002, 0.0006, 5) / (10.0) * Dk  # 0.00021
# contain: 0.0003

# Extracellular growth rate of bacteria /min
alpha_BE = 0.00015 / (10.0) * Dk

# No. of intracellular bacteria when -> chronically infected MP
N_c = 10

# No. of bacteria causes a MP to burst
K_BI = 20

# No. of T needed to activate MP
N_tact = 4

# Prob. of T recruitment # 32
T_recr = np.linspace(10, 40, 5) * 0.01
# contain: 32.5

# Prob. of T movement
T_move = np.linspace(0.5, 2, 5) * 0.01  # 4.97
# contain: 0.825

# Prob. of a T to activate a MP
T_actm = np.linspace(3, 10, 5) * 0.01
# contain: 10

# T lifespan day / 10 min
T_ls = 3 * 24 * 6

# T decay day
T_decay = 10

# Prob. of MP recruitment
M_recr = np.linspace(2, 5, 5) * 0.01  # 2.11
# contain: 4.5

# Resting MP lifespan day / 10 min
M_rls = 100 * 24 * 6

# Activated MP lifespan day / 10 min
M_als = 10 * 24 * 6

# No. of burstings to become necrotic
N_necr = 8

# T cell speed mium/min
T_sp = 10

# Resting MP speed mium/min
M_rsp = 1

# Activated MP speed mium/min
M_asp = 0.025

# Infected MP speed mium/min
M_isp = 0.0007

# Initial No. of resting MP
M_init = 150
# contain: 200

# No. of bacteria killed by MP
N_RK = 2

# No. of bacteria killed by MP
N_phag = 10

# Initial No. of extracellular bacteria
BE_init = 16

# Chemokine released by MP
c_I = 5000

# Time when T cells begin to arrive / 6 s
t_T = 10 * 24 * 600

# Running time
t_total = round(50 * 24 * 600 / k)
