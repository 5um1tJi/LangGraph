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

    response = generator.invoke(messages).content

    return {"comeback" : response}

def evaluate(state: twstate):
    evaluator_messages = [
    SystemMessage(
        content="""
        You are a strict evaluator for comeback replies.

        Your job is to evaluate whether a generated comeback is strong enough
        to be considered a genuinely clever and devastating response to the
        original insult.

        Evaluate the comeback based on:

        1. Cleverness — Is it genuinely witty rather than generic?
        2. Punch — Does it hit hard and feel satisfying?
        3. Relevance — Does it directly respond to the original insult?
        4. Originality — Does it avoid common, boring comebacks?
        5. Wordplay — Does it cleverly use the other person's words, logic,
        assumptions, or situation against them?
        6. Confidence — Does it sound effortless rather than defensive?
        7. Brevity — Is it short and punchy?
        8. Memorability — Would someone remember the comeback afterward?

        Return exactly ONE of these two outputs:

        good
        not_good_enough

        Return "good" ONLY if the comeback is genuinely strong, clever,
        and punchy enough to be used as a great comeback.

        Return "not_good_enough" if it is generic, weak, predictable,
        too defensive, irrelevant, unnecessarily long, or fails to deliver
        a strong punchline.

        IMPORTANT:
        - Do not provide feedback.
        - Do not explain your decision.
        - Do not give a score.
        - Do not output anything except exactly:
        good
        OR
        not_good_enough
        """
            ),

            HumanMessage(
                content=f"""
        Original insult:
        "{state['topic']}"

        Generated comeback:
        "{state['comeback']}"

        Evaluate the comeback now.

        Return ONLY:
        good
        or
        not_good_enough
        """
        )
    ]

    response = evaluator.invoke(evaluator_messages).content

    return {"eval": response}

def optimise(state: twstate):
    # optimizer prompt
    optimize_messages = [
    SystemMessage(
        content="""
    You are an expert comeback optimizer.

    Your job is to improve a previously generated comeback that was
    evaluated as "not_good_enough".

    Do NOT simply rewrite it with different words.
    Identify what made the previous comeback weak and silently fix it.

    Improve it by:
    - Making the punchline sharper.
    - Making the response more clever and unexpected.
    - Using the original insult more effectively.
    - Turning the other person's logic or words against them.
    - Adding better wordplay or irony when appropriate.
    - Removing unnecessary words.
    - Making it sound effortless and confident.
    - Making the ending hit harder.
    - Avoiding generic or predictable insults.
    - Keeping the comeback natural and something a real person could say.

    The new comeback must be BETTER than the previous attempt.

    IMPORTANT:
    - Do not explain what you changed.
    - Do not mention the evaluation.
    - Do not give multiple options.
    - Do not apologize.
    - Do not use question-answer format.
    - Return ONLY the improved comeback.
    """
        ),

        HumanMessage(
            content=f"""
    Original insult:
    "{state['topic']}"

    Previous comeback:
    "{state['comeback']}"

    Previous evaluation:
    "{state['eval']}"

    This comeback was judged not_good_enough.

    Create a significantly stronger version.

    Requirements:
    - Maximum 280 characters.
    - Prefer one sentence.
    - Keep it short and punchy.
    - The punchline should land at the end.
    - Do not merely make it more aggressive.
    - Make it more clever.
    - Use the original insult as ammunition whenever possible.
    - Do not repeat the exact previous comeback.
    - Return ONLY the improved comeback.

    This is optimization iteration #{state['iter'] + 1}.
    """
        )
    ]

    response = optim.invoke(optimize_messages).content
    iter = state["iter"]+1

    return {"comeback" : response, "iter" : iter}

def route(state: twstate):
    if state["eval"] == "good" or state["iter"] >= state["maxIter"] :
        return "good"
    else:
        return "not_good_enough"

graph = StateGraph(twstate)

graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)
graph.add_node("optimise", optimise)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", route, {"good": END, "not_good_enough": "optimise"})
graph.add_edge("optimise","evaluate")

workflow = graph.compile()

intial = {
    "topic" : "you look stupid",
    "iter": 1,
    "maxIter" : 5
}

result = workflow.invoke(intial)
print(result)
