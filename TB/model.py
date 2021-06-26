"""
TB Model
"""

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from TB.agents import (
    Env,
    T,
    RestMP,
    InfectMP,
    ChronInfectMP,
    ActivatedMP,
    Source,
    Necrosis,
)
from TB.schedule import RandomActivationByBreed


class TB(Model):

    height = 100
    width = 100

    verbose = True  # Print-monitoring

    description = "TB model"

    def __init__(
        self,
        height=100,
        width=100,
        # time-tick: 10 min
        k=600 / 6,
        # Chemokine diffusion coefficient /0.1 min
        diffC=0.64 / 4,
        # Chemokine degradation coefficient /0.1 min
        decayC=0.001,
        # Prob. of bacteria being killed within M_R
        p_k=2.36 * 0.01,  # 8.51
        # Prob. T cell kills a macrophage
        pT_k=3.61 * 0.01,  # 6.31
        # Percentage of B_I being destroyed when killed
        P_kill=50 * 0.01,
        # Capacity of extracellular bacteria
        K_BE=200,
        # Intracellular growth rate of bacteria /min
        alpha_BI=0.0003 / (10.0),
        # 0.0002~0.0006
        # contain: 0.0003
        # Extracellular growth rate of bacteria /min
        alpha_BE=0.00015 / (10.0),
        # No. of intracellular bacteria when -> chronically infected MP
        N_c=10,
        # No. of bacteria causes a MP to burst
        K_BI=20,
        # Prob. of T recruitment # 32
        T_recr=32.5 * 0.01,
        # 10~40
        # contain: 32.5
        # Prob. of T movement
        T_move=0.825 * 0.01,
        # 0.5~2
        # contain: 0.825
        # Prob. of a T to activate a MP
        T_actm=3 * 0.01,
        # 3~10
        # contain: 3
        # T lifespan day / 10 min
        T_ls=3 * 24 * 6,
        # Prob. of MP recruitment
        M_recr=2.75 * 0.01,
        # 2~5
        # contain: 2.75
        # Resting MP lifespan day / 10 min
        M_rls=100 * 24 * 6,
        # Activated MP lifespan day / 10 min
        M_als=10 * 24 * 6,
        # No. of burstings to become necrotic
        N_necr=8,
        # Initial No. of resting MP
        M_init=150,
        # No. of bacteria killed by MP
        N_RK=2,
        # No. of bacteria killed by MP
        N_phag=10,
        # Initial No. of extracellular bacteria
        BE_init=16,
        # Chemokine released by MP
        c_I=5000,
        # Time when T cells begin to arrive / 6 s
        t_T=10 * 24 * 600,
        # Running time: 100 d
        t_total=round(100 * 24 * 600),
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.k = k
        self.diffC = diffC
        self.decayC = decayC
        self.p_k = p_k
        self.pT_k = pT_k
        self.P_kill = P_kill
        self.K_BE = K_BE
        self.alpha_BI = alpha_BI
        self.alpha_BE = alpha_BE
        self.N_c = N_c
        self.K_BI = K_BI
        self.T_recr = T_recr
        self.T_move = T_move
        self.T_actm = T_actm
        self.T_ls = T_ls
        self.M_recr = M_recr
        self.M_rls = M_rls
        self.M_als = M_als
        self.N_necr = N_necr
        self.M_init = M_init
        self.N_RK = N_RK
        self.N_phag = N_phag
        self.BE_init = BE_init
        self.c_I = c_I
        self.t_T = t_T
        self.t_total = t_total

        # Create Environment & basic settings
        self.env = Env(self.next_id(), self)
        self.schedule = RandomActivationByBreed(self)
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.datacollector = DataCollector(
            {
                "RestMP": lambda m: m.schedule.get_breed_count(RestMP),
                "InfectMP": lambda m: m.schedule.get_breed_count(InfectMP),
                "ChonInfectMP": lambda m: m.schedule.get_breed_count(ChronInfectMP),
                "ActivatedMP": lambda m: m.schedule.get_breed_count(ActivatedMP),
                "T": lambda m: m.schedule.get_breed_count(T),
                "Necrosis": lambda m: m.schedule.get_breed_count(Necrosis),
            }
        )

        # Create Extracellular Bacteria
        for (x, y) in [(49, 49), (49, 50), (50, 49), (50, 50)]:
            self.env.BE[x, y] = self.BE_init / 4
        self.schedule.add(self.env)

        # Create resting macrophage
        for i in range(self.M_init):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            MP = RestMP(self.next_id(), (x, y), self, True)
            self.grid.place_agent(MP, (x, y))
            self.schedule.add(MP)
        MP_to_be_infected = RestMP(self.next_id(), (49, 49), self, True)
        self.grid.place_agent(MP_to_be_infected, (49, 49))
        self.schedule.add(MP_to_be_infected)

        # Create blood vessel (Source)
        for pos in [(25, 25), (25, 75), (75, 25), (75, 75)]:
            src = Source(self.next_id(), pos, self)
            self.grid.place_agent(src, pos)
            self.schedule.add(src)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()

        # collect data for testing
        self.datacollector.collect(self)
        if self.verbose:
            print(
                [
                    self.schedule.time,
                    np.sum(self.env.BE),
                    self.schedule.get_breed_count(RestMP),
                    self.schedule.get_breed_count(InfectMP),
                    self.schedule.get_breed_count(ChronInfectMP),
                    self.schedule.get_breed_count(ActivatedMP),
                    self.schedule.get_breed_count(T),
                    self.schedule.get_breed_count(Necrosis),
                ]
            )

    def run_model(self):

        for i in range(self.t_total):
            self.step()
            if (
                np.sum(self.env.BE) == 0.0
                and self.schedule.get_breed_count(InfectMP) == 0
                and self.schedule.get_breed_count(ChronInfectMP) == 0
            ):
                break
