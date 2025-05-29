from google.adk.agents import Agent
from google.adk.tools import google_search

web_search_agent = Agent(
    name="web_search_agent",
    model="gemini-2.0-flash-exp",
    description="Performs Google Search for factual grounding.",
    instruction="Use google_search to fetch factual information.",
    tools=[google_search]  # only google_search built‑in tool here
)
