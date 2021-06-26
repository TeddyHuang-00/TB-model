import numpy as np
from typing import Any
import random
import math
from mesa import Agent
from mesa.space import accept_tuple_argument
from numpy.lib.twodim_base import triu_indices_from


def accept_tuple_argument(wrapped_function):
    """Decorator to allow grid methods that take a list of (x, y) coord tuples
    to also handle a single position, by automatically wrapping tuple in
    single-item list rather than forcing user to do it.

    """

    def wrapper(*args: Any):
        if isinstance(args[1], tuple) and len(args[1]) == 2:
            return wrapped_function(args[0], [args[1]])
        else:
            return wrapped_function(*args)

    return wrapper


def sigmond(x):
    # in random walk
    return 1 / (1 + np.exp(-x))


class DirectedRandomWalker(Agent):
    """
    Class implementing random walker methods in a generalized manner.

    Not indended to be used on its own, but to inherit its methods to multiple
    other agents.

    """

    grid = None
    x = None
    y = None
    moore = True

    def __init__(self, unique_id, pos, model, moore=True):
        """
        grid: The MultiGrid object in which the agent lives.
        x: The agent's current x coordinate
        y: The agent's current y coordinate
        moore: If True, may move in all 8 directions.
                Otherwise, only up, down, left, right.
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.moore = moore
        self.time = 1

    def directed_random_move(self) -> tuple:
        """
        Random walk of cells towards chemokine gradient

        Return: tuple of (x, y) as next-move direction
        """
        # Choose the most possible cell to move
        next_moves = self.model.grid.get_neighborhood(self.pos, moore=True)
        pos_cnt = len(next_moves)
        c = np.zeros(pos_cnt)
        for i in range(pos_cnt):
            if self.model.env.C[next_moves[i][0], next_moves[i][1]] >= 1.0:
                c[i] += self.model.env.C[next_moves[i][0], next_moves[i][1]]
        c_mean = np.mean(c)
        if c_mean > 0:
            c_relative = (c - c_mean) / c_mean
            c_sigmond = sigmond(c_relative * 15)
            p = c_sigmond / np.sum(c_sigmond)
        else:
            p = np.ones(pos_cnt) / pos_cnt
        pre_index = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        index = np.random.choice(pre_index[:pos_cnt], p=p)

        return next_moves[index]

    @accept_tuple_argument
    def get_cell_list_contents(self, cell_list, classType=object):
        """
        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.
            classType: typical agent class

        Returns:
            A list of the typical class of the contents of the cells identified in cell_list

        """
        return [
            obj
            for obj in list(self.model.grid.iter_cell_list_contents(cell_list))
            if isinstance(obj, classType)
        ]

    def timeAdd(self):
        self.time += 1

    def timeRestore(self):
        self.time = 1


class Env(Agent):
    """
    Environment consists of Chemokine and Extracellular Bacteria
    Chemokine is released by macrophages to attract macrophages and T cells
    """

    def __init__(self, unique_id, model):
        """
        C: concentration of chemokine
        BE: No. of extracellular bacteria
        death_cnt: No. of chronically infected macrophage die in the grid
        Necrosis: the status of the grid, cells cannot excess necrotic places
        """
        super().__init__(unique_id, model)
        self.height = self.model.height
        self.width = self.model.width
        self.C = np.zeros((self.height, self.width))
        self.C1 = self.C.copy()
        self.BE = np.zeros((self.height, self.width))
        self.death_cnt = np.zeros((self.height, self.width))
        self.necrosis = np.zeros((self.height, self.width), dtype=bool)
        self.start_time = 1
        self.time = 1
        self.oneStep = 1

    def step(self):
        for cnt in range(round(self.model.k)):
            # Decay and Diffusion of chemokine
            self.C = (1 - self.model.decayC) * self.C
            self.Diffusion()

            # Replication of bacteria
            self.BE = self.BE + self.model.alpha_BE * self.BE * (
                1 - (self.BE / self.model.K_BE)
            )

            # Necrosis
            self.necrosis = self.death_cnt >= self.model.N_necr

    def Diffusion(self):
        # Diffusion function
        self.C1[1:-1, 1:-1] = self.C[1:-1, 1:-1] + self.model.diffC * (
            (self.C[2:, 1:-1] - 2 * self.C[1:-1, 1:-1] + self.C[:-2, 1:-1])
            + (self.C[1:-1, 2:] - 2 * self.C[1:-1, 1:-1] + self.C[1:-1, :-2])
        )

        self.C = self.C1.copy()

    def timeAdd(self):
        self.time += 1

    def timeRestore(self):
        self.time = 1


class Necrosis(Agent):
    """
    Necrosis forms when a number of chronically infected macrophages burst or are killed by T cells
    """

    def __init__(self, unique_id, pos, model):
        super().__init__(unique_id, model)
        self.pos = pos


class T(DirectedRandomWalker):
    """
    T cells
    """

    def __init__(self, unique_id, pos, model, moore):
        """
        pos: position of the T cell
        age: age of the T cell, ranging from 0 to T cell lifespan
        """
        super().__init__(unique_id, pos, model, moore=moore)
        self.pos = pos
        self.age = random.randint(0, self.model.T_ls)

        self.start_time = 1
        self.time = 1
        self.oneStep = 100 / self.model.k

    def step(self):
        """
        Random walk towards chemokine gradient
        Cannot be occupied by another T cell
        When there is one macrophage, T moves with the prob. T_move
        """
        next_move = self.directed_random_move()
        if (
            len(self.get_cell_list_contents([next_move], T)) == 0
            and self.model.env.necrosis[next_move] == False
        ):
            if (len(self.get_cell_list_contents([next_move], MP)) == 0) or (
                random.random() < self.model.T_move
            ):
                self.model.grid.move_agent(self, next_move)

        # Aging
        self.age += 1
        if self.age >= self.model.T_ls:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)


class MP(DirectedRandomWalker):
    """
    General macrophage
    """

    def __init__(self, unique_id, pos, model, moore):
        """
        pos: position of the macrophage
        age: age of the macrophage, ranging from 0 to macrophage lifespan
        BI: internal bacteria of the macrophage
        """
        super().__init__(unique_id, pos, model, moore=moore)
        self.pos = pos
        self.age = 0
        self.BI = 0

        self.start_time = 1
        self.time = 1
        self.oneStep = 100 / self.model.k

    def ChemokineSecretion(self):
        self.model.env.C[self.pos[0], self.pos[1]] += self.model.c_I

    def MPWalk(self):
        """
        Random walk towards chemokine gradient
        Cannot be occupied by another macrophage
        """
        next_move = self.directed_random_move()
        if (
            len(self.get_cell_list_contents([next_move], MP)) == 0
            and self.model.env.necrosis[next_move] == False
        ):
            self.model.grid.move_agent(self, next_move)


class RestMP(MP):
    """
    Rested Macrophage
    """

    def __init__(self, unique_id, pos, model, moore):
        """
        pos: position of the macrophage
        age: age of the macrophage, ranging from 0 to rested macrophage lifespan
        BI: internal bacteria of the macrophage
        exist: whether the macrophage has died
        """
        super().__init__(unique_id, pos, model, moore=moore)
        self.pos = pos
        self.age = random.randint(0, self.model.M_rls)
        self.exist = True

        self.start_time = 1
        self.time = 1
        self.walk_cnt = 0
        self.oneStep = 100 / self.model.k

    def step(self):
        # walk for every 10 ticks
        if self.walk_cnt == 10:
            self.MPWalk()
            self.walk_cnt = 0
        else:
            self.walk_cnt += 1

        # Kill extracellular bacteria or being infected
        if self.model.env.BE[self.pos] <= self.model.N_RK:
            self.model.env.BE[self.pos] = 0
        else:
            if random.random() < self.model.p_k:
                self.model.env.BE[self.pos] -= self.model.N_RK
            else:
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)
                self.exist = False
                self.model.env.BE[self.pos] -= self.model.N_RK
                inMP = InfectMP(
                    self.model.next_id(),
                    self.pos,
                    self.model,
                    self.moore,
                    self.age,
                    self.model.N_RK,
                )
                self.model.grid.place_agent(inMP, self.pos)
                self.model.schedule.add(inMP)

        # Aging & Die
        if self.exist:
            self.age += 1
            if self.age >= self.model.M_rls:
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)


class InfectMP(MP):
    """
    Infected macrophage
    """

    def __init__(self, unique_id, pos, model, moore, age, B_I):
        """
        pos: position of the macrophage
        age: age of the macrophage, same to when it is infected
        BI: internal bacteria of the macrophage
        exist: whether the macrophage has died
        """
        super().__init__(unique_id, pos, model, moore=moore)
        self.B_I = B_I
        self.age = age
        self.exist = True

        self.start_time = 1
        self.time = 1
        self.walk_cnt = 0
        self.oneStep = 100 / self.model.k

    def step(self):
        # Random_walk
        if self.walk_cnt == 100:
            self.MPWalk()
            self.walk_cnt = 0
        else:
            self.walk_cnt += 1

        # Chemokine secretion & Growth of bacteria inside
        self.ChemokineSecretion()
        self.B_I = pow(1 + self.model.alpha_BI, self.model.k) * self.B_I

        # Become chronically infected
        if self.B_I > self.model.N_c:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
            self.exist = False
            chinMP = ChronInfectMP(
                self.model.next_id(), self.pos, self.model, self.moore, self.B_I
            )
            self.model.grid.place_agent(chinMP, self.pos)
            self.model.schedule.add(chinMP)

        # Become activated by T cells
        if self.exist:
            ngh = self.model.grid.get_neighbors(
                self.pos, moore=False, include_center=True
            )
            ngh_T = [obj for obj in ngh if isinstance(obj, T)]
            if random.random() < len(ngh_T) * self.model.T_actm:
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)
                self.exist = False
                actMP = ActivatedMP(
                    self.model.next_id(), self.pos, self.model, self.moore
                )
                self.model.grid.place_agent(actMP, self.pos)
                self.model.schedule.add(actMP)

        # Aging & Die, then release intracellular bacteria
        if self.exist:
            self.age += 1
            if self.age >= self.model.M_rls:
                x, y = self.pos
                self.model.env.BE[x - 1 : x + 2, y - 1 : y + 2] += self.B_I / 9
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)
                self.exist = False


class ChronInfectMP(MP):
    """
    Chronically infected macrophage
    """

    def __init__(self, unique_id, pos, model, moore, B_I):
        """
        pos: position of the macrophage
        age: age of the macrophage, ranging from 0 to chronically infected macrophage lifespan
        BI: internal bacteria of the macrophage
        exist: whether the macrophage has died
        """
        super().__init__(unique_id, pos, model, moore=moore)
        self.B_I = B_I
        self.age = random.randint(0, self.model.M_rls)
        self.exist = True

        self.start_time = 1
        self.time = 1
        self.walk_cnt = 0
        self.oneStep = 100 / self.model.k

    def step(self):
        if self.walk_cnt == 100:
            self.MPWalk()
            self.walk_cnt = 0
        else:
            self.walk_cnt += 1

        for i in range(round(self.model.k)):
            self.B_I = self.B_I + self.model.alpha_BI * self.B_I * (
                1 - self.B_I / (self.model.K_BI + 30)
            )

        x, y = self.pos

        # T cell killing
        cell_T = self.get_cell_list_contents([self.pos], T)
        if len(cell_T) > 0 and random.random() < self.model.pT_k:
            self.model.env.BE[x - 1 : x + 2, y - 1 : y + 2] += 0.5 * self.B_I / 9
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
            self.exist = False
            self.model.env.death_cnt[x, y] += 1.0

        # Aging & Bursting
        if self.exist:
            self.age += 1
            if self.age >= self.model.M_rls or self.B_I > self.model.K_BI:
                self.model.env.BE[x - 1 : x + 2, y - 1 : y + 2] += self.B_I / 9
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)
                self.exist = False
                self.model.env.death_cnt[x, y] += 1.0

        if (
            self.model.env.death_cnt[x, y] >= self.model.N_necr
            and self.model.env.necrosis[x, y] == False
        ):
            self.model.env.necrosis[x, y] = True
            necrosis = Necrosis(self.model.next_id(), self.pos, self.model)
            self.model.grid.place_agent(necrosis, self.pos)


class ActivatedMP(MP):
    """
    Activated macrophage
    """

    def __init__(self, unique_id, pos, model, moore):
        """
        pos: position of the macrophage
        age: age of the macrophage, ranging from 0 to activated macrophage lifespan
        """
        super().__init__(unique_id, pos, model, moore=moore)
        self.age = random.randint(0, self.model.M_als)

        self.start_time = 1
        self.time = 1
        self.walk_cnt = 0
        self.oneStep = 100 / self.model.k

    def step(self):
        # Random walk
        if self.walk_cnt == 10:
            self.MPWalk()
            self.walk_cnt = 0
        else:
            self.walk_cnt += 1

        # Chemokine secretion
        self.ChemokineSecretion()

        # Kill extracellular bacteria
        if self.model.env.BE[self.pos] > self.model.N_phag:
            self.model.env.BE[self.pos] -= self.model.N_phag
        else:
            self.model.env.BE[self.pos] = 0

        # Aging
        self.age += 1
        if self.age >= self.model.M_als:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)


class Source(Agent):
    """
    Blood vessel
    """

    def __init__(self, unique_id, pos, model):
        """
        pos: position of the vessel
        """
        super().__init__(unique_id, model)
        self.pos = pos
        self.time = 1
        self.start_time = 1
        self.oneStep = 100 / self.model.k

    def step(self):
        # Recruitment of macrophages and T cells based on prob.
        this_cell = self.model.grid.get_cell_list_contents([self.pos])
        place_MP = False
        place_T = False
        if random.random() < self.model.M_recr:
            place_MP = True
        # Recruit T cells after self.model.t_T ticks
        if (random.random() < self.model.T_recr) and (
            self.model.schedule.time > self.model.t_T / self.model.k
        ):
            place_T = True
        if place_T or place_MP:
            for obj in this_cell:
                if isinstance(obj, MP):
                    place_MP = False
                if isinstance(obj, T):
                    place_T = False
                if self.model.env.C[self.pos[0], self.pos[1]] < 1.0:
                    place_T = place_MP = False

        if place_MP:
            newMP = RestMP(self.model.next_id(), self.pos, self.model, moore=True)
            self.model.grid.place_agent(newMP, newMP.pos)
            self.model.schedule.add(newMP)
        if place_T:
            newT = T(self.model.next_id(), self.pos, self.model, moore=True)
            self.model.grid.place_agent(newT, newT.pos)
            self.model.schedule.add(newT)

    def timeAdd(self):
        self.time += 1

    def timeRestore(self):
        self.time = 1
