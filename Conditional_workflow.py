from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal

class quad(TypedDict):
    a: int
    b: int
    c: int

    eq: str
    dis: float
    res: str

def show(state: quad):

    eq = f"{state["a"]}x2{state["b"]}x{state["c"]}"

    return {"eq": eq}

def disc(state: quad):

    ans = state["b"]**2 - 4* state["a"]* state["c"]

    return {"dis": ans}

def real(state: quad):

    first = (-state["b"] + state["dis"]**0.5) / (2*state["a"])
    second = (-state["b"] - state["dis"]**0.5) / (2*state["a"])

    result = f"The results are {first} and {second}"

    return {"res": result}

def noreal(state: quad):

    result = "No real roots"

    return {"res": result}

def repeat(state: quad):

    rist = -(state["b"])/(2 * state["a"])

    return {"res": rist}

def check(state: quad) -> Literal["real", "noreal", "repeat"]:
    if state["dis"] > 0:
        return "real"
    elif state["dis"] == 0:
        return "repeat"
    else:
        return "noreal"


graph = StateGraph(quad)

graph.add_node("show", show)
graph.add_node("disc", disc)
graph.add_node("real", real)
graph.add_node("noreal", noreal)
graph.add_node("repeat", repeat)

graph.add_edge(START, "show")
graph.add_edge("show", "disc")
graph.add_conditional_edges("disc", check)
graph.add_edge("real", END)
graph.add_edge("noreal", END)
graph.add_edge("repeat", END)

workflow = graph.compile()

initial = {
    "a": 4, 
    "b": -5,
    "c": -4
}

result = workflow.invoke(initial)
print(result)