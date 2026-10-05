from langgraph.graph import StateGraph, END, START
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)

class pickstate(TypedDict):
    name : str
    line: str


graph = StateGraph(pickstate)

def generate(state: pickstate):
    name  = state["name"]
    prompt = f"""
    You are a creative and charming pickup-line generator.

    The user will give you a name.

    Your task is to generate ONE clever, natural, and playful pickup line based on the name.

    Try to use one or more of these approaches:
    1. The meaning of her name, if the meaning is reasonably known.
    2. A rhyme or wordplay based on how her name sounds.
    3. A clever association with the name.
    4. A compliment that naturally incorporates the name.
    5. If the name has multiple possible meanings, choose the most common or recognizable one.

    Rules:
    - Keep the pickup line short: 1-2 sentences maximum.
    - Make it charming and playful, not creepy or overly sexual.
    - It should sound like something a real person could actually say.
    - Do NOT simply say "Your name is beautiful."
    - Make the girl's name an important part of the joke or compliment.
    - If you are unsure about the meaning of the name, rely on pronunciation, rhyme, or wordplay instead of inventing a meaning.
    - Do not explain your reasoning.
    - Return ONLY the pickup line.

    Girl's name: {name}
    """

    resp = llm.invoke(prompt).content

    return {"line" : resp}


graph.add_node("generate", generate)

graph.add_edge(START, "generate")
graph.add_edge("generate", END)

check = InMemorySaver()

workflow = graph.compile(checkpointer=check)
initial = {
    "name" : "narendra"
}

first = {"configurable": {"thread_id" : "1"}}
response = workflow.invoke(initial, config=first)

print(response)

print(workflow.get_state(first))
history = list(workflow.get_state_history(first))

for state in history:
    print(state)

# Agar kai sari thread id ho to aise karo, jyada theek rahega
# elvis_history = list(
#     workflow.get_state_history(
#         {"configurable": {"thread_id": "elvis"}}
#     )
# )
