import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from TB.agents import ActivatedMP, ChronInfectMP, Env, InfectMP, RestMP, Source, T
from TB.model import TB


class RMP(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Rest MP: " + str(model.schedule.get_breed_count(RestMP))


class IMP(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Infected MP: " + str(model.schedule.get_breed_count(InfectMP))


class CIMP(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Chronicly Infected MP: " + str(
            model.schedule.get_breed_count(ChronInfectMP)
        )


class AMP(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Activated MP: " + str(model.schedule.get_breed_count(ActivatedMP))


class T_stat(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "T: " + str(model.schedule.get_breed_count(T))


class BE(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Bacteria number: " + str(np.sum(model.env.BE))


def TB_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is RestMP:
        portrayal["Color"] = ["#84e184", "#adebad", "#d6f5d6"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is ActivatedMP:
        portrayal["Color"] = ["#0000FF", "#0000FF", "#0000FF"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 2
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is InfectMP:
        portrayal["Color"] = ["#FFA500", "#FFA500", "#FFA500"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 3
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is ChronInfectMP:
        portrayal["Color"] = ["#FF0000", "#FFF000", "#FF0000"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 3
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is T:
        portrayal["Color"] = ["#FFC0CB", "#FFC0CB", "#FFC0CB"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 3
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is Source:
        portrayal["Color"] = ["#000000", "#000000", "#000000"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    """
    elif type(agent) is Chemokine:
        if (agent.C >= 1.0):
            portrayal["Color"] = ["#000000", "#000000", "#000000"]
            portrayal["Shape"] = "rect"
            portrayal["Filled"] = 'true'
            portrayal["Layer"] = 2
            portrayal["w"] = 1
            portrayal["h"] = 1
    """

    return portrayal


canvas_element = CanvasGrid(TB_portrayal, 100, 100, 500, 500)
"""
model_params = {
    "RestMP": UserSettableParameter("checkbox", "restMP", True
    ),
}
"""
RMP_counting = RMP()
IMP_counting = IMP()
CIMP_counting = CIMP()
AMP_counting = AMP()
T_counting = T_stat()

line_chart = ChartModule(
    [
        {"Label": "RestMP", "Color": "Black"},
        {"Label": "InfectMP", "Color": "Blue"},
        {"Label": "ChonInfectMP", "Color": "Red"},
        {"Label": "ActivatedMP", "Color": "Green"},
        {"Label": "T", "Color": "Purple"},
        {"Label": "BE", "Color": "Orange"},
    ]
)
server = ModularServer(
    TB,
    [
        canvas_element,
        RMP_counting,
        IMP_counting,
        CIMP_counting,
        AMP_counting,
        T_counting,
        line_chart,
    ],
    "TB",
)
server.port = 8521
