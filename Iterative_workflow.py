from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)

generator = ChatGroq(
    model="openai/gpt-oss-20b",
)

evaluator = ChatGroq(
    model="openai/gpt-oss-20b",
)

optim = ChatGroq(
    model = "openai/gpt-oss-20b",
)

class twstate(TypedDict):
    topic : str
    comeback : str
    eval : Literal["good", "not_good_enough"]
    feedback : str
    iter : int
    maxIter: int



def generate(state: twstate):

    # prompt
    messages = [
        SystemMessage(
            content="""
    You are a world-class comeback writer and verbal-wit expert.
    Given an insult or disrespectful statement, create a comeback that
    completely flips the situation.
    Your goal is NOT simply to insult harder.
    Your goal is to make the original speaker's statement collapse
    under its own logic.
    Use:
    - Wordplay
    - Irony
    - Sarcasm
    - Double meanings
    - Misdirection
    - Unexpected comparisons
    - Self-own reversals
    - Precise observations
    - Short punchlines

    A legendary comeback should feel:
    CALM + CLEVER + UNEXPECTED + EFFORTLESS.

    Never sound defensive.
    Never explain the joke.
    Never use generic "at least I'm..." responses.

    Before answering, silently generate multiple possible comebacks,
    compare them, and choose the one with the strongest punchline.

    Return ONLY the final comeback.
    """
        ),

        HumanMessage(
            content=f"""
    INSULT:
    "{state['topic']}"

    Create the ultimate comeback.

    Requirements:
    - Maximum 2 sentences.
    - Prefer 1 sentence.
    - Maximum 280 characters.
    - The punchline should hit at the end.
    - Flip their own words or logic against them if possible.
    - Make it witty rather than merely abusive.
    - No explanation.
    - No quotation marks.
    - No hashtags.
    - No emojis unless they genuinely improve the punchline.
    - This is attempt {state['iter'] + 1}.

    Return ONLY the comeback.
    """
        )
    ]

    response = generator(messages).content

    return {"comeback" : response}




graph = StateGraph(twstate)

graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)
graph.add_node("optimise", optimise)
