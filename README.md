# Multi-Agent System with Guardrails

Advanced research pipeline featuring specialized agents with input guardrails, agent handoffs, and structured output for investment research.

## Features

**Input Guardrails**
- Political topic detection and blocking
- Custom guardrail agent validates inputs
- Prevents processing of restricted content

**Specialized Agent Team**
- **Planner Agent:** Breaks queries into search strategies
- **Search Agent:** Executes Tavily web searches
- **Fundamentals Analyst:** Analyzes company financials
- **Writer Agent:** Generates markdown reports

**Agent Handoffs**
- Automatic transfer between agents
- Structured data passing (Pydantic models)
- Context preservation across handoffs

**Structured Output**
- Type-safe responses using Pydantic
- Search plans, summaries, and final reports
- Follow-up question generation

## Tech Stack

- **Framework:** OpenAI Agents SDK
- **Models:** GPT-4.1-mini (main), GPT-4.1-nano (guardrail)
- **Tools:** Tavily Search API
- **Output:** Pydantic BaseModel schemas
- **Async:** Full async/await support

## Workflow

```
User Query → Politics Guardrail → Planner → Writer → [Search + Fundamentals] → Final Report
```

## Setup

```bash
pip install openai python-dotenv requests pydantic agents
```

Create `.env`:
```
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

Run:
```bash
python main.PY
```

## Example Output

Generates comprehensive markdown investment reports with:
- Executive summary
- Detailed analysis
- Follow-up questions
- Source citations

Demonstrates advanced agent orchestration patterns including guardrails, handoffs, and multi-agent collaboration for complex research tasks.
