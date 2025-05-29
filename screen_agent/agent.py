from google.adk.agents import Agent
from .utils.screen_capture import capture_screen

# General screen analysis agent
screen_agent = Agent(
    name="screen_agent",
    model="gemini-2.0-flash-exp",
    description="Analyzes screen images.",
    instruction="I will automatically capture and analyze your screen. Ask me questions about what you see.",
    tools=[capture_screen]   # ← only your custom screen‐capture tool here
)

# Specialized game strategy agent
game_agent = Agent(
    name="game_agent",
    model="gemini-2.0-flash-exp",
    description="Game strategy helper.",
    instruction="Help with game strategies and moves.",
    tools=[]                  # no extra tools
)

# Specialized puzzle-solving agent
puzzle_agent = Agent(
    name="puzzle_agent",
    model="gemini-2.0-flash-exp",
    description="Puzzle solver.",
    instruction="Solve puzzles like Sudoku or chess.",
    tools=[]                  # no extra tools
)