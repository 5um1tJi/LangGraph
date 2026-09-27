from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)

class eval(BaseModel):
    feedback: str = Field(description="Feedback of the essay")
    score : int = Field(description="Score out of 10", ge=0, le=10)

newllm = llm.with_structured_output(eval)
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

prompt = f"analyse this essay and provide a quality summary of the essay and a score out of 10 /n {essay}"

class upscState(TypedDict):
    essay: str
    langfeed: str
    anafeed: str
    qualfeed: str
    indi_score: str



ans = newllm.invoke(prompt)
print(ans.score)
print(ans.feedback)