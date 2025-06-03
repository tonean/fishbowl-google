from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from .utils.screen_capture import capture_screen
from google_search_agent.agent import web_search_agent
from .web_interaction_agent import web_interaction_agent

# Coding Assistant Agent
coding_agent = Agent(
    name="coding_agent",
    model="gemini-2.0-flash-exp",
    description="Expert coding assistant with code analysis, debugging, and best practices knowledge.",
    instruction="""I am a specialized coding assistant with expertise in:
- Code analysis and review
- Debugging and problem-solving
- Best practices and design patterns
- Performance optimization
- Security considerations
- Documentation and testing

I can help with:
1. Code review and improvement suggestions
2. Debugging and error resolution
3. Architecture and design decisions
4. Performance optimization
5. Security best practices
6. Documentation and testing strategies""",
    tools=[capture_screen]  # Can analyze code on screen
)

# Design Assistant Agent
design_agent = Agent(
    name="design_agent",
    model="gemini-2.0-flash-exp",
    description="UI/UX design expert with visual analysis capabilities.",
    instruction="""I am a specialized design assistant with expertise in:
- UI/UX design principles
- Visual design analysis
- Accessibility standards
- Design system implementation
- User experience optimization
- Design-to-code translation

I can help with:
1. Design review and feedback
2. UI/UX best practices
3. Accessibility compliance
4. Design system implementation
5. Visual consistency
6. User experience optimization""",
    tools=[capture_screen]  # Can analyze design screenshots
)

# Data Analysis Agent
data_analysis_agent = Agent(
    name="data_analysis_agent",
    model="gemini-2.0-flash-exp",
    description="Data analysis expert with statistical and analytical capabilities.",
    instruction="""I am a specialized data analysis assistant with expertise in:
- Statistical analysis
- Data visualization
- Data cleaning and preprocessing
- Pattern recognition
- Trend analysis
- Predictive modeling

I can help with:
1. Data analysis and interpretation
2. Statistical testing
3. Data visualization recommendations
4. Pattern and trend identification
5. Data quality assessment
6. Analytical methodology""",
    tools=[capture_screen, AgentTool(agent=web_search_agent)]  # Can analyze data visualizations and search for reference
)

# Code Review Agent
code_review_agent = Agent(
    name="code_review_agent",
    model="gemini-2.0-flash-exp",
    description="Specialized code review agent with focus on quality and best practices.",
    instruction="""I am a specialized code review assistant with expertise in:
- Code quality assessment
- Security vulnerability detection
- Performance optimization
- Best practices enforcement
- Documentation review
- Testing coverage analysis

I can help with:
1. Code quality review
2. Security vulnerability assessment
3. Performance optimization suggestions
4. Best practices compliance
5. Documentation quality
6. Test coverage analysis""",
    tools=[capture_screen]  # Can analyze code on screen
)

# Data Science Agent
data_science_agent = Agent(
    name="data_science_agent",
    model="gemini-2.0-flash-exp",
    description="Data science expert with machine learning and statistical analysis capabilities.",
    instruction="""I am a specialized data science assistant with expertise in:
- Machine learning
- Statistical analysis
- Feature engineering
- Model evaluation
- Data preprocessing
- Predictive modeling

I can help with:
1. Machine learning model selection
2. Statistical analysis
3. Feature engineering
4. Model evaluation and validation
5. Data preprocessing
6. Predictive modeling""",
    tools=[capture_screen, AgentTool(agent=web_search_agent)]  # Can analyze visualizations and search for reference
)

# Export all agents
__all__ = [
    'coding_agent',
    'design_agent',
    'data_analysis_agent',
    'code_review_agent',
    'data_science_agent',
    'web_interaction_agent'
] 