import os, requests, asyncio
from IPython.display import display, Markdown
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, SQLiteSession
from pydantic import BaseModel
from typing_extensions import TypedDict
from datetime import datetime
from agents import RunResult
from agents import handoff, RunContextWrapper, CodeInterpreterTool, input_guardrail, GuardrailFunctionOutput, TResponseInputItem
from agents.extensions import handoff_filters
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

print("API keys loaded")
print(f"OpenAI: {openai_api_key[:5]}***")
print(f"Tavily: {tavily_api_key[:5]}***")

def print_markdown(txt: str):
    display(Markdown(txt))

main_model = "gpt-4.1-mini"
small_model = "gpt-4.1-nano"

class TavilyParams(TypedDict):
    query: str
    max_results: int

class PoliticalTopicOutput(BaseModel):
    is_political: bool
    reasoning: str

politics_guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about political topics, politicians, elections, government policy, or anything related to politics. If so, set is_political to true and explain why in reasoning.",
    output_type=PoliticalTopicOutput
)

@input_guardrail
async def politics_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(politics_guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_political
    )

@function_tool
def tavily_search(params: TavilyParams) -> str:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key,
        "query": params["query"],
        "max_results": params.get("max_results", 3),
    }
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    if resp.status_code != 200:
        return f"Tavily error {resp.status_code}"
    items = resp.json().get("results", [])
    return "\n".join([f"- {i['title']}: {i['content']}" for i in items]) or "No hits"

class SearchPlanItem(BaseModel):
    reason: str
    query: str

class SearchPlan(BaseModel):
    searches: list[SearchPlanItem]

date = datetime.now().strftime("%Y-%m-%d")

planner_agent = Agent(
    name="Planner",
    instructions=f"""Current date: {date}
You are a research planner agent.
Break down the user's request into 3 distinct web searches with clear reasons and queries.
Output must match the SearchPlan schema.""",
    model=main_model,
    output_type=SearchPlan,
    input_guardrails=[politics_guardrail]
)

q1 = "Why is Trump meeting with Putin this week?"
run1 = await Runner.run(starting_agent=planner_agent, input=q1)
print_markdown(f"{run1.final_output}")

class Summary(BaseModel):
    summary: str

search_agent = Agent(
    name="Searcher",
    instructions="""Use Tavily search to find recent and relevant information.
Summarize results in no more than 200 words.""",
    tools=[tavily_search],
    model=main_model,
    output_type=Summary
)

fundamentals_agent = Agent(
    name="FundamentalsAnalyst",
    instructions="""Analyze company fundamentals including revenue, growth, and margins.""",
    output_type=Summary,
    model=main_model,
    tools=[tavily_search]
)

class FinalReport(BaseModel):
    short_summary: str
    markdown_report: str
    follow_up_questions: list[str]

async def extract_summary(run_result: RunResult) -> str:
    return run_result.final_output.summary

writer_agent = Agent(
    name="Writer",
    instructions="""Write a detailed markdown investment report with an executive summary and follow-up questions.""",
    model=main_model,
    output_type=FinalReport,
    tools=[
        fundamentals_agent.as_tool("fundamentals", "Get fundamentals analysis", custom_output_extractor=extract_summary),
        search_agent.as_tool("search", "Get search results", custom_output_extractor=extract_summary),
    ],
)

class PlannerToWriterInput(BaseModel):
    original_query: str
    search_plan: SearchPlan

def on_planner_to_writer(ctx: RunContextWrapper[None], input_data: PlannerToWriterInput):
    print_markdown("Transfer: Planner → Writer")

handoff_to_writer = handoff(
    agent=writer_agent,
    input_type=PlannerToWriterInput,
    on_handoff=on_planner_to_writer,
    tool_name_override="transfer_to_writer",
    tool_description_override="Transfer to Writer with original query and search plan"
)

planner_with_handoff = planner_agent.clone(
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
{planner_agent.instructions}
When you have produced the SearchPlan, call the handoff tool with the required JSON input.""",
    handoffs=[handoff_to_writer]
)

async def run_handoffs_demo(user_query: str):
    print_markdown(user_query)
    run_res = await Runner.run(planner_with_handoff, user_query)
    report = run_res.final_output
    print_markdown(report.short_summary)
    print_markdown(report.markdown_report)
    return run_res

handoff_result = await run_handoffs_demo(
    "Can you let me know about the stock holdings of people who are in congress"
)

handoff_result = await run_handoffs_demo(
    "Do a deep dive on the latest news and developments in the AAPL stock. Also I need the fundamentals analysis of the company."
)
