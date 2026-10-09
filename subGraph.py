from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

class substate(TypedDict):
    input: str
    output: str

def trans(state: substate):
    que = state["input"]

    prompt = f"""
            Translate this input {que} into hindi language, do not change or add anything, just 
            tanslate as it is.
        """
    resp = llm.invoke(prompt).content
    return {"output": resp}

subgraph = StateGraph(substate)

subgraph.add_node("trans", trans)
subgraph.add_edge(START, "trans")
subgraph.add_edge("trans", END)

subwork = subgraph.compile()

class parent(TypedDict):
    prasn: str
    uttar: str
    translate: str

def ans(state: parent):
    topic = state["prasn"]
    prompt = f"write 5 line intro about the topic {topic}"
    resp = llm.invoke(prompt).content
    return {"uttar": resp}

def translate(state: parent):
    topic = state["uttar"]
    resp = subwork.invoke({"input": topic})
    return {"translate": resp["output"]}

graph = StateGraph(parent)

graph.add_node("ans", ans)
graph.add_node("translate", translate)
graph.add_edge(START, "ans")
graph.add_edge("ans", "translate")
graph.add_edge("translate", END)

workflow = graph.compile()

initial = {"prasn" : "UFC"}

resp = workflow.invoke(initial)
print(resp)