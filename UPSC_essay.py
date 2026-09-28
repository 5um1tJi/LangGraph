from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)

class eval(BaseModel):
    feedback: str = Field(description="Feedback of the essay")
    score : int = Field(description="Score out of 10", ge=0, le=10)

newllm = llm.with_structured_output(eval, method="json_mode")
essay = """
Chess is a game that strips away luck entirely, leaving only strategy, foresight, and raw mental discipline.
Played across a modest sixty-four-square board, it has survived for centuries as one of humanity's greatest intellectual tests.
Every match starts with absolute symmetry, giving both players the exact same pieces and opportunities.
The complexity begins immediately with the first move, opening up millions of possible mathematical variations.
Pawns, knights, bishops, rooks, queens, and kings each move with distinct rules that demand careful coordination.
A player must think several steps ahead, anticipating counter-moves while protecting their own pieces from sudden threats.
Mistakes are unforgiving, often turning a winning position into a painful defeat because of one careless oversight.
Beyond individual matches, chess teaches valuable real-world lessons about patience, risk management, and accountability.
There are no teammates to blame when a plan fails, making every victory or loss entirely your own responsibility.
The rise of online platforms has transformed chess from a quiet parlor hobby into a massive global internet phenomenon.
Millions of people now play daily on mobile apps, watching grandmasters stream high-stakes games live from anywhere.
Engine analysis and computer databases have completely revolutionized how modern competitors study opening theory and tactics.
Yet, despite powerful silicon processors, human intuition and psychological pressure still dictate the outcome over the board.
Elite tournaments routinely capture international attention, highlighting the intense pressure faced by professional grandmasters.
Young prodigies continue to push boundaries, lowering the age records for grandmaster titles and reshaping the competitive landscape.
Classic rivalries spark fierce debates among fans, driving massive engagement across digital communities and social platforms.
Learning the game requires memorizing basic patterns, but mastering it demands a lifelong commitment to deep study.
Children who learn chess often develop sharper problem-solving skills, better concentration, and enhanced spatial reasoning.
At its core, chess is a universal language bridging cultures, ages, and backgrounds through pure intellectual competition.
Whether played in a quiet park or on a global championship stage, the royal game remains an enduring masterpiece of human ingenuity.
"""
class upscState(TypedDict):
    essay: str
    langfeed: str
    anafeed: str
    qualfeed: str
    final_feed: str
    indi_score: Annotated[list[float], operator.add]
    avg_score : float


def eval_lang(state: upscState):
    prompt = f"Analyse this essay and provide a language quality feedback and a score out of 10. Respond in valid json format. \n {state['essay']}"
    output = newllm.invoke(prompt)
    return {"langfeed": output.feedback, "indi_score" : [output.score]}

def eval_ana(state: upscState):
    prompt = f"Analyse this essay and provide an analysis feedback and a score out of 10. Respond in valid json format. \n {state['essay']}"
    output = newllm.invoke(prompt)
    return {"anafeed": output.feedback, "indi_score" : [output.score]}

def eval_thoug(state: upscState):
    prompt = f"Analyse the thought process of this essay and provide a feedback and a score out of 10. Respond in valid json format. \n {state['essay']}"
    output = newllm.invoke(prompt)
    return {"qualfeed": output.feedback, "indi_score" : [output.score]}
def final_eval(state: upscState):
    prompt = f"Based on the following feedback create a summarized feedback. language_feedback - {state['langfeed']} \n analysis_feedback - {state['anafeed']} \n quality_feedback - {state['qualfeed']}"
    output = llm.invoke(prompt).content
    return {"final_feed": output}

graph = StateGraph(upscState)

graph.add_node("eval_lang", eval_lang)
graph.add_node("eval_ana", eval_ana)
graph.add_node("eval_thoug", eval_thoug)
graph.add_node("final_eval", final_eval)

graph.add_edge(START,"eval_lang")
graph.add_edge(START, "eval_ana")
graph.add_edge(START, "eval_thoug")
graph.add_edge("eval_lang", "final_eval")
graph.add_edge("eval_thoug", "final_eval")
graph.add_edge("eval_ana", "final_eval")
graph.add_edge("final_eval", END)

workflow = graph.compile()

initial = {
    "essay": essay
}
workflow.invoke(initial)

