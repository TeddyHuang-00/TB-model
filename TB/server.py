import numpy as np
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

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


class Necro(TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Necrosis: " + str(model.schedule.get_breed_count(Necrosis))


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
        portrayal["Layer"] = 1
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is InfectMP:
        portrayal["Color"] = ["#FFA500", "#FFA500", "#FFA500"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is ChronInfectMP:
        portrayal["Color"] = ["#FF0000", "#FFF000", "#FF0000"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is T:
        portrayal["Color"] = ["#FFC0CB", "#FFC0CB", "#FFC0CB"]
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 2
        portrayal["w"] = 1
        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 0
        portrayal["w"] = 1
        portrayal["h"] = 1

    elif type(agent) is Necrosis:
        portrayal["Color"] = ["#CC7722", "#CC7722", "#CC7722"]
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 3
        portrayal["w"] = 1
        portrayal["h"] = 1


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
Necrosis_counting = Necro()

RMP_line_chart = ChartModule([{"Label": "RestMP", "Color": "Black"}])
IMP_line_chart = ChartModule([{"Label": "InfectMP", "Color": "Blue"}])
CIMP_line_chart = ChartModule([{"Label": "ChonInfectMP", "Color": "Red"}])
AMP_line_chart = ChartModule([{"Label": "ActivatedMP", "Color": "Green"}])
T_line_chart = ChartModule([{"Label": "T", "Color": "Purple"}])
Necrosis_line_chart = ChartModule([{"Label": "Necrosis", "Color": "Cyan"}])
BE_line_chart = ChartModule([{"Label": "BE", "Color": "Orange"}])

canvas_element = CanvasGrid(TB_portrayal, 100, 100, 500, 500)
server = ModularServer(
    TB,
    [
        canvas_element,
        RMP_counting,
        IMP_counting,
        CIMP_counting,
        AMP_counting,
        T_counting,
        Necrosis_counting,
        RMP_line_chart,
        IMP_line_chart,
        CIMP_line_chart,
        AMP_line_chart,
        T_line_chart,
        Necrosis_line_chart,
        BE_line_chart,
    ],
    "TB",
)
server.port = 8521
