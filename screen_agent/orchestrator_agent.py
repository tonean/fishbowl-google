# app/screen_agent/orchestrator_agent.py
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Content, Part
from screen_agent.vision_tool import vision_via_files_api
from screen_agent.memory_system import EnhancedMemorySystem
from google_search_agent.agent import web_search_agent
from screen_agent.agent import game_agent, puzzle_agent
from screen_agent.specialized_agents import (
    coding_agent,
    design_agent,
    code_review_agent,
    web_interaction_agent 
)
from screen_agent.enhanced_data_analysis_agent import (
    enhanced_data_analysis_agent,
    financial_data_agent,
    business_intelligence_agent,
    scientific_data_agent
)
from typing import Dict, List, Any, Optional, Set
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Represents a result from a tool execution"""
    tool_name: str
    result: str
    timestamp: str
    confidence: float
    metadata: Dict[str, Any]

class ToolCoordinator:
    """Coordinates tool usage and maintains shared knowledge"""
    
    def __init__(self):
        self.tool_results: Dict[str, List[ToolResult]] = {}  # tool_name -> results
        self.shared_knowledge: Dict[str, Any] = {}  # key -> value
        self.last_visual_context: Optional[Dict[str, Any]] = None  # Track the last visual context
        self.visual_context_expiry: Optional[datetime] = None  # When the visual context expires
        self.tool_capabilities: Dict[str, Set[str]] = {
            'vision_via_files_api': {'screen_access', 'visual_analysis'},
            'enhanced_data_analysis_agent': {'data_analysis', 'statistical_analysis', 'visualization_advice', 'chart_analysis', 'ml_analysis'},
            'financial_data_agent': {'financial_analysis', 'market_analysis'},
            'business_intelligence_agent': {'business_analysis', 'kpi_analysis'},
            'scientific_data_agent': {'scientific_analysis', 'research_analysis'},
            'web_search_agent': {'web_search', 'fact_checking'},
            'screen_agent': {'screen_access', 'visual_analysis'},
            'design_agent': {'visual_analysis', 'design_analysis'},
            'coding_agent': {'code_analysis', 'visual_analysis'},
            'code_review_agent': {'code_analysis', 'visual_analysis'},
            'web_interaction_agent': {'web_interaction', 'translation', 'document_interaction', 'screen_access', 'visual_analysis'}  # Add this line

        }
        
        # Track which agents have visual capabilities
        self.visual_capable_agents = {
            'vision_via_files_api',
            'screen_agent',
            'enhanced_data_analysis_agent',
            'design_agent',
            'coding_agent',
            'code_review_agent',
            'web_interaction_agent'

        }
    
    def record_tool_result(self, tool_name: str, result: str, confidence: float, metadata: Dict[str, Any] = None):
        """Record a tool's execution result"""
        if tool_name not in self.tool_results:
            self.tool_results[tool_name] = []
            
        tool_result = ToolResult(
            tool_name=tool_name,
            result=result,
            timestamp=datetime.utcnow().isoformat(),
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.tool_results[tool_name].append(tool_result)
        
        # Extract and store shared knowledge
        self._extract_shared_knowledge(tool_result)
        
        # Update visual context if this is a visual analysis
        if tool_name in self.visual_capable_agents:
            self._update_visual_context(tool_result)
    
    def _update_visual_context(self, tool_result: ToolResult):
        """Update the current visual context"""
        now = datetime.utcnow()
        
        # Extract visual information
        visual_info = {
            'content': tool_result.result,
            'timestamp': tool_result.timestamp,
            'confidence': tool_result.confidence,
            'source_tool': tool_result.tool_name,
            'metadata': tool_result.metadata
        }
        
        # If there's an analysis section, extract it
        if 'Analysis:' in tool_result.result:
            analysis = tool_result.result.split('Analysis:')[1].strip()
            visual_info['analysis'] = analysis
        
        # Update the visual context
        self.last_visual_context = visual_info
        # Set expiry to 5 minutes from now
        self.visual_context_expiry = now + timedelta(minutes=5)
    
    def get_relevant_knowledge(self, tool_name: str) -> Dict[str, Any]:
        """Get knowledge relevant to a specific tool"""
        relevant_knowledge = {}
        
        # CRITICAL FIX: For data analysis agents, ALWAYS provide current visual context
        if tool_name in ['enhanced_data_analysis_agent', 'financial_data_agent', 'business_intelligence_agent', 'scientific_data_agent']:
            # Ensure we have fresh visual data
            now = datetime.utcnow()
            if (not self.last_visual_context or 
                not self.visual_context_expiry or 
                now >= self.visual_context_expiry):
                try:
                    fresh_visual = vision_via_files_api()
                    if fresh_visual and "cannot access" not in fresh_visual.lower():
                        self._update_visual_context(ToolResult(
                            tool_name='vision_via_files_api',
                            result=fresh_visual,
                            timestamp=now.isoformat(),
                            confidence=0.9,
                            metadata={'auto_refresh': True}
                        ))
                except Exception as e:
                    logger.error(f"Error refreshing visual context: {e}")
            
            # Always provide visual context if available
            if self.last_visual_context:
                relevant_knowledge['current_screen_data'] = self.last_visual_context
                # Also provide a formatted version for easy agent consumption
                content = self.last_visual_context.get('content', '')
                if 'Analysis:' in content:
                    analysis = content.split('Analysis:')[1].strip()
                    relevant_knowledge['formatted_visual_analysis'] = analysis
        # Get capabilities of the requesting tool
        tool_capabilities = self.tool_capabilities.get(tool_name, set())
        
        # Always include valid visual context for visual-capable agents
        now = datetime.utcnow()
        if self.last_visual_context and self.visual_context_expiry and now < self.visual_context_expiry:
            if tool_name in self.visual_capable_agents or any(cap in tool_capabilities for cap in ['visual_analysis', 'data_analysis']):
                relevant_knowledge['current_visual_context'] = self.last_visual_context
        
        # Share visual information with any tool that might need it
        if any(cap in tool_capabilities for cap in ['visual_analysis', 'data_analysis', 'statistical_analysis', 'visualization_advice', 'chart_analysis']):
            if 'last_visual_analysis' in self.shared_knowledge:
                relevant_knowledge['visual_analysis'] = self.shared_knowledge['last_visual_analysis']
            if 'last_visual_data' in self.shared_knowledge:
                relevant_knowledge['visual_data'] = self.shared_knowledge['last_visual_data']
        
        return relevant_knowledge
    def _extract_text_for_translation(query: str, visual_context: str) -> str:
        """Extract text that needs translation from query and visual context"""
        
        # Look for quoted text in the query
        import re
        quoted_text = re.findall(r'"([^"]+)"', query)
        if quoted_text:
            return quoted_text[0]
        
        # Look for "this" references and try to identify from visual context
        if 'this' in query.lower():
            # Try to extract visible text from visual context
            # This is a simplified approach - in practice, you'd need more sophisticated text extraction
            lines = visual_context.split('\n')
            potential_text = []
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('Analysis:') and not line.startswith('I can see'):
                    potential_text.append(line)
            
            if potential_text:
                return potential_text[0]  # Return first substantial text line
        
        return "No specific text identified for translation"
    
    def should_use_tool(self, tool_name: str, query: str) -> bool:
        """Determine if a tool should be used based on previous results and query"""
        # Check if we have recent visual analysis that could be useful
        if any(word in query.lower() for word in ['see', 'look', 'show', 'display', 'graph', 'chart', 'image', 'picture']):
            # If we have valid visual context, use it instead of requesting a new capture
            now = datetime.utcnow()
            if self.last_visual_context and self.visual_context_expiry and now < self.visual_context_expiry:
                return False  # Don't use the tool, use existing context
            
            if 'last_visual_analysis' in self.shared_knowledge:
                # If this tool has visual capabilities, it can use the shared knowledge
                if tool_name in self.visual_capable_agents:
                    return True
                # If this tool needs visual analysis, it can use the shared knowledge
                if any(cap in self.tool_capabilities.get(tool_name, set()) for cap in ['visual_analysis', 'data_analysis']):
                    return True
        
        return True
    
    def get_visual_capable_agent(self, query: str) -> Optional[str]:
        """Get the most appropriate visual-capable agent for a query"""
        if not self.visual_capable_agents:
            return None
            
        # Check query for domain-specific keywords
        query_lower = query.lower()
        
        # Financial analysis
        if any(word in query_lower for word in ['financial', 'market', 'stock', 'trading', 'investment']):
            return 'financial_data_agent'
            
        # Business analysis
        if any(word in query_lower for word in ['business', 'sales', 'revenue', 'marketing']):
            return 'business_intelligence_agent'
            
        # Scientific analysis
        if any(word in query_lower for word in ['scientific', 'research', 'experiment', 'data']):
            return 'scientific_data_agent'
            
        # Design analysis
        if any(word in query_lower for word in ['design', 'ui', 'ux', 'interface']):
            return 'design_agent'
            
        # Code analysis
        if any(word in query_lower for word in ['code', 'programming', 'debug', 'review']):
            return 'code_review_agent'
            
        # Default to enhanced data analysis for general visual analysis
        return 'enhanced_data_analysis_agent'

# Initialize enhanced memory system and tool coordinator
memory_system = EnhancedMemorySystem()
tool_coordinator = ToolCoordinator()

def get_smart_context_response(query: str, conversation_id: str, 
                              context_type: str = 'adaptive') -> str:
    """
    Get an intelligent context-aware response that adapts to the query type and user preferences
    
    Args:
        query: The user's query
        conversation_id: ID of the current conversation
        context_type: Type of context to provide ('adaptive', 'detailed', 'focused', 'minimal')
    """
    try:
        # Get user preferences to tailor the response
        user_prefs = memory_system.get_user_preferences(conversation_id)
        
        # Determine optimal context based on query analysis
        if context_type == 'adaptive':
            context_type = _determine_optimal_context_type(query, user_prefs)
        
        # Extract key entities and topics from the query
        query_entities = memory_system.entity_extractor.extract_entities(query)
        query_words = query.lower().split()
        
        context_parts = []
        
        # 1. Get relevant conversation history
        if context_type in ['detailed', 'adaptive']:
            history = memory_system.get_conversation_context(
                conversation_id, 
                max_messages=8 if context_type == 'detailed' else 5,
                include_entities=True,
                include_topics=True
            )
            context_parts.append(f"Conversation Context:\n{history}")
        elif context_type == 'focused':
            # Only get recent messages that are highly relevant
            recent_messages = memory_system.get_conversation_history(conversation_id, limit=3)
            if recent_messages:
                focused_context = []
                for msg in recent_messages:
                    if any(entity.name.lower() in query.lower() for entity in msg.entities):
                        focused_context.append(f"{msg.role}: {msg.content}")
                if focused_context:
                    context_parts.append("Relevant Recent Context:\n" + "\n".join(focused_context))
        
        # 2. Get related entities and their relationships
        related_entities = []
        for entity in query_entities:
            related_concepts = memory_system.get_related_concepts(entity.name, max_depth=2, limit=5)
            if related_concepts:
                related_entities.extend([concept for concept, weight in related_concepts])
        
        # Also check for entities mentioned in the query text
        for word in query_words:
            if len(word) > 3:  # Skip short words
                related_concepts = memory_system.get_related_concepts(word, max_depth=1, limit=3)
                related_entities.extend([concept for concept, weight in related_concepts])
        
        if related_entities and context_type != 'minimal':
            unique_entities = list(set(related_entities))[:10]  # Limit to top 10
            context_parts.append(f"Related Knowledge: {', '.join(unique_entities)}")
        
        # 3. Search for relevant historical messages
        relevant_messages = []
        
        # Search by entities found in query
        for entity in query_entities:
            messages = memory_system.search_by_entity(entity.name, limit=3)
            relevant_messages.extend(messages)
        
        # Search by topics/keywords
        for word in query_words:
            if len(word) > 4:  # Only search for substantial words
                messages = memory_system.search_by_topic(word, limit=2)
                relevant_messages.extend(messages)
        
        # Filter and deduplicate relevant messages
        if relevant_messages and context_type in ['detailed', 'adaptive']:
            unique_messages = []
            seen_ids = set()
            
            # Sort by importance and recency
            relevant_messages.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            
            for msg in relevant_messages[:5]:  # Limit to top 5 most relevant
                if msg.id not in seen_ids:
                    unique_messages.append(f"Historical Context: {msg.content[:200]}...")
                    seen_ids.add(msg.id)
            
            if unique_messages:
                context_parts.append("\n".join(unique_messages))
        
        # 4. Add user preferences and behavioral insights
        if user_prefs and context_type != 'minimal':
            pref_context = []
            
            if 'communication_style' in user_prefs:
                pref_context.append(f"User prefers {user_prefs['communication_style']} responses")
            
            if 'favorite_topics' in user_prefs:
                topics = user_prefs['favorite_topics'][:3]  # Top 3 topics
                pref_context.append(f"User frequently discusses: {', '.join(topics)}")
            
            if pref_context:
                context_parts.append(f"User Profile: {'; '.join(pref_context)}")
        
        # 5. Add temporal context if relevant
        if any(word in query.lower() for word in ['yesterday', 'today', 'last', 'recent', 'earlier']):
            from datetime import datetime, timedelta
            today = datetime.utcnow().isoformat()[:10]
            yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()[:10]
            
            recent_messages = memory_system.search_by_timeframe(yesterday, today + "T23:59:59")
            if recent_messages:
                context_parts.append(f"Recent Activity: Found {len(recent_messages)} recent messages")
        
        # 6. Build the final context-aware prompt
        if not context_parts:
            # Minimal context fallback
            basic_history = memory_system.get_conversation_context(conversation_id, max_messages=2)
            context_prompt = f"""Current query: {query}
Recent context: {basic_history}

Please provide a helpful response."""
        else:
            context_sections = "\n\n".join(context_parts)
            context_prompt = f"""{context_sections}

Current Query: {query}

Instructions: Provide a response that:
1. Takes into account the conversation history and related knowledge
2. Matches the user's preferred communication style
3. References relevant past discussions when appropriate
4. Builds upon previously established context and relationships

Response:"""
        
        return context_prompt
        
    except Exception as e:
        logger.error(f"Error generating smart context: {e}")
        # Fallback to basic context
        basic_context = memory_system.get_conversation_context(conversation_id, max_messages=3)
        return f"Context: {basic_context}\n\nQuery: {query}\n\nPlease respond helpfully."

def _determine_optimal_context_type(query: str, user_prefs: Dict[str, Any]) -> str:
    """Determine the optimal context type based on query analysis and user preferences"""
    
    query_lower = query.lower()
    
    # Check for context indicators in the query
    detailed_indicators = ['explain', 'detailed', 'comprehensive', 'full context', 'everything about']
    focused_indicators = ['specific', 'just', 'only', 'briefly', 'quick']
    minimal_indicators = ['simple', 'short', 'yes/no', 'basic']
    
    # Analyze query complexity
    is_complex = len(query.split()) > 10 or '?' in query or any(word in query_lower for word in ['how', 'why', 'what', 'when', 'where'])
    
    # Check user preferences
    user_style = user_prefs.get('communication_style', 'balanced')
    
    # Decision logic
    if any(indicator in query_lower for indicator in minimal_indicators):
        return 'minimal'
    elif any(indicator in query_lower for indicator in detailed_indicators):
        return 'detailed'
    elif any(indicator in query_lower for indicator in focused_indicators):
        return 'focused'
    elif user_style == 'detailed' and is_complex:
        return 'detailed'
    elif user_style == 'concise' and not is_complex:
        return 'focused'
    else:
        return 'adaptive'  # Balanced approach

def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze the user's query to understand intent and requirements"""
    
    query_lower = query.lower()
    
    intent_analysis = {
        'type': 'general',
        'requires_search': False,
        'requires_vision': False,
        'requires_memory': False,
        'complexity': 'medium',
        'temporal_reference': False,
        'entity_focused': False,
        'domain': 'general',
        'route_to_agent': None,
        'tool_sequence': []  # Track sequence of tools to use
    }
    
    web_interaction_keywords = ['translate', 'paste', 'copy', 'google docs', 'docs', 'document', 'spreadsheet', 'interact', 'click', 'type', 'fill', 'form']
    translation_keywords = ['translate', 'translation', 'put in google docs', 'paste in docs', 'google docs']

    if any(keyword in query_lower for keyword in translation_keywords):
        intent_analysis['domain'] = 'web_interaction'
        intent_analysis['route_to_agent'] = 'web_interaction_agent'
        intent_analysis['requires_vision'] = True
        intent_analysis['tool_sequence'] = ['vision_via_files_api', 'web_interaction_agent']

    # Detect query type
    if any(word in query_lower for word in ['search', 'find', 'look up', 'google']):
        intent_analysis['type'] = 'search'
        intent_analysis['requires_search'] = True
        intent_analysis['route_to_agent'] = 'web_search_agent'
    elif any(word in query_lower for word in ['screen', 'image', 'picture', 'see', 'show']):
        intent_analysis['type'] = 'vision'
        intent_analysis['requires_vision'] = True
        # First use vision_via_files_api to get screen content
        intent_analysis['tool_sequence'].append('vision_via_files_api')
        # Then route to appropriate agent based on content
        if any(word in query_lower for word in ['graph', 'chart', 'visualize', 'plot', 'data']):
            intent_analysis['route_to_agent'] = 'enhanced_data_analysis_agent'
    elif any(word in query_lower for word in ['remember', 'recall', 'mentioned', 'said before', 'earlier']):
        intent_analysis['type'] = 'memory'
        intent_analysis['requires_memory'] = True
    elif any(word in query_lower for word in ['game', 'puzzle', 'play', 'solve']):
        intent_analysis['type'] = 'interactive'
        if 'game' in query_lower:
            intent_analysis['route_to_agent'] = 'game_agent'
        elif 'puzzle' in query_lower:
            intent_analysis['route_to_agent'] = 'puzzle_agent'
    elif any(keyword in query_lower for keyword in web_interaction_keywords):
        intent_analysis['domain'] = 'web_interaction'
        intent_analysis['route_to_agent'] = 'web_interaction_agent'
        intent_analysis['requires_vision'] = True
        intent_analysis['tool_sequence'].append('vision_via_files_api')
        
    # Detect domain-specific intent
    coding_keywords = ['code', 'programming', 'debug', 'function', 'class', 'algorithm', 'api', 'database', 'python', 'javascript', 'java', 'c++', 'bug', 'syntax']
    design_keywords = ['design', 'ui', 'ux', 'interface', 'layout', 'style', 'css', 'accessibility', 'mockup', 'wireframe', 'prototype']
    data_keywords = ['data', 'analysis', 'statistics', 'visualization', 'chart', 'graph', 'dashboard', 'metrics', 'dataset', 'csv']
    ml_keywords = ['machine learning', 'ml', 'model', 'training', 'prediction', 'ai', 'neural network', 'deep learning', 'classification', 'regression']
    financial_keywords = ['financial', 'finance', 'stock', 'investment', 'portfolio', 'trading', 'market', 'revenue', 'profit', 'roi']
    business_keywords = ['business', 'kpi', 'performance', 'analytics', 'report', 'sales', 'customer', 'marketing', 'strategy']
    scientific_keywords = ['scientific', 'research', 'experiment', 'hypothesis', 'analysis', 'lab', 'study', 'publication']
    
    if any(keyword in query_lower for keyword in coding_keywords):
        intent_analysis['domain'] = 'coding'
        if any(word in query_lower for word in ['review', 'check', 'audit', 'quality']):
            intent_analysis['route_to_agent'] = 'code_review_agent'
        else:
            intent_analysis['route_to_agent'] = 'coding_agent'
    elif any(keyword in query_lower for keyword in design_keywords):
        intent_analysis['domain'] = 'design'
        intent_analysis['route_to_agent'] = 'design_agent'
    elif any(keyword in query_lower for keyword in financial_keywords):
        intent_analysis['domain'] = 'financial'
        intent_analysis['route_to_agent'] = 'financial_data_agent'
    elif any(keyword in query_lower for keyword in business_keywords):
        intent_analysis['domain'] = 'business'
        intent_analysis['route_to_agent'] = 'business_intelligence_agent'
    elif any(keyword in query_lower for keyword in scientific_keywords):
        intent_analysis['domain'] = 'scientific'
        intent_analysis['route_to_agent'] = 'scientific_data_agent'
    elif any(keyword in query_lower for keyword in ml_keywords):
        intent_analysis['domain'] = 'ml'
        intent_analysis['route_to_agent'] = 'enhanced_data_analysis_agent'
    elif any(keyword in query_lower for keyword in data_keywords):
        intent_analysis['domain'] = 'data'
        intent_analysis['route_to_agent'] = 'enhanced_data_analysis_agent'
    
    # Detect complexity
    if len(query.split()) > 15 or query.count('?') > 1:
        intent_analysis['complexity'] = 'high'
    elif len(query.split()) < 5:
        intent_analysis['complexity'] = 'low'
    
    # Detect temporal references
    temporal_words = ['today', 'yesterday', 'last week', 'recently', 'earlier', 'before', 'when']
    if any(word in query_lower for word in temporal_words):
        intent_analysis['temporal_reference'] = True
    
    # Detect if query is entity-focused
    # Simple heuristic: proper nouns or quoted terms
    import re
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
    quoted_terms = re.findall(r'"([^"]*)"', query)
    
    if proper_nouns or quoted_terms:
        intent_analysis['entity_focused'] = True
    
    return intent_analysis

def get_contextual_system_prompt(conversation_id: str, query_intent: Dict[str, Any]) -> str:
    """Generate a contextual system prompt based on conversation state and query intent"""
    
    # Get conversation statistics
    stats = memory_system.get_statistics()
    user_prefs = memory_system.get_user_preferences(conversation_id)
    
    # Base system prompt
    system_prompt = """I am an intelligent assistant with enhanced memory and context awareness. 
I remember our conversations, track important entities and topics, and adapt my responses to your preferences."""
    
    # Add routing information if query should be routed to specialized agent
    if query_intent.get('route_to_agent'):
        agent_name = query_intent['route_to_agent']
        system_prompt += f"\n\nINTERNAL: Use {agent_name} silently for specialized assistance - do not mention this to the user."
    
    # Add capability information based on query intent
    if query_intent['requires_search']:
        system_prompt += "\nI can search the web for current information and real-time data."
    
    if query_intent['requires_vision']:
        system_prompt += "\nI can analyze images and screenshots to help with visual tasks."
    
    if query_intent['requires_memory']:
        system_prompt += f"\nI maintain detailed memory of our conversations with {stats['total_messages']} messages and {stats['total_entities']} tracked entities."
    
    # Add user preference context
    if user_prefs.get('communication_style'):
        style = user_prefs['communication_style']
        system_prompt += f"\nI'll provide {style} responses based on your preferred communication style."
    
    if user_prefs.get('favorite_topics'):
        topics = ', '.join(user_prefs['favorite_topics'][:3])
        system_prompt += f"\nI know you frequently discuss: {topics}."
    
    # Add complexity guidance
    if query_intent['complexity'] == 'high':
        system_prompt += "\nI'll provide comprehensive, detailed responses for complex queries."
    elif query_intent['complexity'] == 'low':
        system_prompt += "\nI'll keep responses focused and concise for simple queries."
    
    return system_prompt


def get_enhanced_context_aware_response(query: str, conversation_id: str) -> str:
    """Main function to get enhanced context-aware response with feedback loop and query routing"""
    try:
        # Analyze the query to understand intent and requirements
        query_intent = analyze_query_intent(query)
        
        # Calculate confidence score based on query analysis
        confidence_score = _calculate_query_confidence(query, query_intent)
        
        # Check if clarification is needed
        if memory_system.should_request_clarification("orchestrator", "query_analysis", confidence_score):
            return f"I'm not entirely confident about understanding your request (confidence: {confidence_score:.2f}). Could you please clarify what you're looking for?"
        
        # Generate smart contextual response based on intent and conversation state
        context_response = get_smart_context_response(
            query, 
            conversation_id, 
            context_type='adaptive'
        )
        
        # Add system-level context for the orchestrator
        system_context = get_contextual_system_prompt(conversation_id, query_intent)
        
        # Process tool sequence if specified
        tool_responses = []
        if query_intent.get('tool_sequence'):
            for tool_name in query_intent['tool_sequence']:
                if tool_coordinator.should_use_tool(tool_name, query):
                    # Get relevant knowledge for this tool
                    tool_knowledge = tool_coordinator.get_relevant_knowledge(tool_name)
                    
                    # Execute tool and record result
                    if tool_name == 'vision_via_files_api':
                        result = vision_via_files_api()
                        tool_coordinator.record_tool_result(
                            tool_name=tool_name,
                            result=result,
                            confidence=confidence_score,
                            metadata={'query': query}
                        )
                        tool_responses.append(result)
        
        # Add agent routing instruction if applicable
        routing_instruction = ""
        visual_data_for_agent = ""

        if query_intent.get('route_to_agent'):
            agent_name = query_intent['route_to_agent']
            
            # CRITICAL FIX: Always ensure visual data is available for data analysis agents
            if agent_name in ['enhanced_data_analysis_agent', 'financial_data_agent', 'business_intelligence_agent', 'scientific_data_agent']:
                # First check if we have recent valid visual context
                now = datetime.utcnow()
                if (not tool_coordinator.last_visual_context or 
                    not tool_coordinator.visual_context_expiry or 
                    now >= tool_coordinator.visual_context_expiry):
                    
                    # Get fresh visual data
                    try:
                        current_visual = vision_via_files_api()
                        if current_visual and "cannot access" not in current_visual.lower():
                            tool_coordinator.record_tool_result(
                                tool_name='vision_via_files_api',
                                result=current_visual,
                                confidence=0.9,
                                metadata={'query': query, 'for_agent': agent_name}
                            )
                    except Exception as e:
                        logger.error(f"Error getting visual data: {e}")
                
                # Now get the visual context for the agent
                if tool_coordinator.last_visual_context:
                    visual_content = tool_coordinator.last_visual_context.get('content', '')
                    # Extract the analysis section specifically
                    if 'Analysis:' in visual_content:
                        analysis_section = visual_content.split('Analysis:')[1].strip()
                        visual_data_for_agent = f"\n\nVISUAL CONTEXT FOR ANALYSIS:\n{analysis_section}\n"
                    else:
                        visual_data_for_agent = f"\n\nVISUAL CONTEXT FOR ANALYSIS:\n{visual_content}\n"
            
            # CRITICAL FIX: Updated routing instruction with proper context passing
            agent_knowledge = tool_coordinator.get_relevant_knowledge(agent_name)

            routing_instruction = f"""\n\nIMPORTANT AGENT ROUTING:
        Use the {agent_name} tool with this complete context. You have access to the current screen data.
        {visual_data_for_agent}
        
        Request format: "Based on the visual data showing [description], analyze [specific request]"
        
        Provide the specialized analysis directly as your own response."""    
            if agent_knowledge:
                routing_instruction += f"\n\nAdditional knowledge available:\n"
                for key, value in agent_knowledge.items():
                    if key != 'current_visual_context':  # Don't duplicate visual context
                        routing_instruction += f"- {key}: {value.get('content', '')[:200]}...\n"
        
        # Record the action
        action_id = memory_system.record_agent_action(
            agent_name="orchestrator",
            action_type="query_analysis",
            input_data={"query": query, "intent": query_intent},
            output_data={
                "context": context_response,
                "system_prompt": system_context,
                "tool_responses": tool_responses,
                "routing": query_intent.get('route_to_agent')
            },
            confidence=confidence_score
        )
        
        # Combine everything into a comprehensive prompt
        enhanced_prompt = f"""{system_context}

{context_response}



Query Analysis: {json.dumps(query_intent, indent=2)}
Confidence Score: {confidence_score:.2f}
{routing_instruction}

Tool Responses:
{chr(10).join(tool_responses) if tool_responses else 'No tool responses yet'}


CRITICAL INSTRUCTIONS:
1. Use specialized agents silently - NEVER mention routing to the user
2. When using agents, include all available visual context in your request
3. Present the agent's analysis as your own direct response
4. If an agent provides analysis, deliver it immediately without asking the user to repeat their question
5. Be seamless and transparent - the user should not know about internal agent routing."""

        return enhanced_prompt
        
    except Exception as e:
        logger.error(f"Error in enhanced context generation: {e}")
        # Record failed action
        memory_system.record_agent_action(
            agent_name="orchestrator",
            action_type="query_analysis",
            input_data={"query": query},
            output_data={"error": str(e)},
            confidence=0.0,
            success=False
        )
        # Fallback to basic context
        return get_smart_context_response(query, conversation_id, context_type='minimal')

def _calculate_query_confidence(query: str, query_intent: Dict[str, Any]) -> float:
    """Calculate confidence score for query understanding"""
    confidence_factors = []
    
    # 1. Query complexity factor
    words = query.split()
    if len(words) < 3:
        confidence_factors.append(0.6)  # Very short queries are ambiguous
    elif len(words) > 20:
        confidence_factors.append(0.8)  # Long queries usually provide more context
    else:
        confidence_factors.append(0.9)  # Medium length is ideal
    
    # 2. Intent clarity factor
    if query_intent['type'] == 'general':
        confidence_factors.append(0.7)  # General queries are less clear
    else:
        confidence_factors.append(0.9)  # Specific intent is clearer
    
    # 3. Entity presence factor
    if query_intent['entity_focused']:
        confidence_factors.append(0.9)  # Entity-focused queries are clearer
    else:
        confidence_factors.append(0.7)
    
    # 4. Temporal reference factor
    if query_intent['temporal_reference']:
        confidence_factors.append(0.8)  # Temporal references add clarity
    else:
        confidence_factors.append(0.9)
    
    # 5. Question mark factor
    if '?' in query:
        confidence_factors.append(0.9)  # Questions are usually clear
    else:
        confidence_factors.append(0.7)
    
    # 6. Domain specificity factor
    if query_intent['domain'] != 'general':
        confidence_factors.append(0.9)  # Domain-specific queries are clearer
    else:
        confidence_factors.append(0.7)
    
    # 7. Agent routing factor
    if query_intent.get('route_to_agent'):
        confidence_factors.append(0.95)  # Clear routing increases confidence
    else:
        confidence_factors.append(0.8)
    
    # Calculate final confidence score
    return sum(confidence_factors) / len(confidence_factors)

# Update the orchestrator agent with new specialized agents
orchestrator = Agent(
    name="enhanced_unified_assistant",
    model="gemini-2.0-flash-exp",
    description="Advanced context-aware assistant with intelligent memory, entity tracking, adaptive responses, and intelligent query routing to specialized agents.",
    instruction="""I am an advanced AI assistant with sophisticated memory and context awareness capabilities, including intelligent query routing to specialized agents.

Key Features:
- Enhanced Memory: I maintain detailed conversation history with entity extraction, topic modeling, and relationship tracking
- Smart Context: I provide adaptive context based on query type, user preferences, and conversation patterns  
- Intelligent Search: I can search for and reference relevant information from our conversation history
- Personalization: I learn and adapt to user communication preferences and interests
- Multi-modal: I can analyze screens, search the web, help with games, and solve puzzles
- Domain Expertise: I have specialized agents for coding, design, data analysis, visualization, and more
- Query Routing: I automatically route queries to the most appropriate specialized agent based on domain analysis
- Feedback Loop: I learn from user feedback to improve my responses over time
- Confidence Scoring: I know when to ask for clarification based on my confidence level

CRITICAL VISUAL HANDLING RULES:
1. NEVER allow "I cannot see" or "Please provide the graph" responses when routing to visual-capable agents
2. ALWAYS automatically include current visual context in agent requests
3. When routing to data analysis agents, format the request as: "Based on the visual data showing [description], analyze [specific request]"
4. If visual context is not immediately available, get it using vision_via_files_api before routing
5. Present all agent responses seamlessly as my own analysis


CRITICAL EXECUTION RULES:
1. When I identify that a query needs specialized analysis, I MUST immediately call the appropriate agent tool
2. I NEVER just mention routing - I execute it immediately  
3. For visual analysis queries, I automatically call vision_via_files_api first, then the appropriate analysis agent
4. I present the specialized agent's response as my own analysis
5. I NEVER tell the user about internal routing processes

AUTOMATIC TOOL USAGE PATTERNS:

For data analysis queries mentioning graphs, charts, or visual data:
1. IMMEDIATELY call vision_via_files_api to get current screen data
2. IMMEDIATELY call enhanced_data_analysis_agent with the visual context
3. Present the analysis directly

For financial analysis queries:
1. Call vision_via_files_api if visual data is mentioned
2. IMMEDIATELY call financial_data_agent with full context
3. Present the financial analysis directly

For business metrics queries:
1. Call vision_via_files_api if visual data is mentioned  
2. IMMEDIATELY call business_intelligence_agent with full context
3. Present the business analysis directly

For scientific data queries:
1. Call vision_via_files_api if visual data is mentioned
2. IMMEDIATELY call scientific_data_agent with full context
3. Present the research analysis directly

EXAMPLE CORRECT BEHAVIOR:
User: "Analyze this financial graph"
My actions:
1. Call vision_via_files_api → get visual data
2. Call financial_data_agent with: "Based on the visual data showing [specific graph details], provide comprehensive financial analysis of [specific elements]"
3. Present the agent's response as my own analysis

EXAMPLE INCORRECT BEHAVIOR (NEVER DO THIS):
User: "Analyze this financial graph"  
My response: "I'll route this to the financial_data_agent..." ❌ WRONG!

The user should never see routing instructions - they should only see the final analysis.

When routing to specialized agents for visual analysis:
- Include the complete visual context in the agent request
- Never ask the user to provide data that should be automatically available
- Format agent requests to reference the visual data explicitly
- Ensure agents understand they have access to the visual information

Example routing for graph analysis:
Instead of: "Analyze the graph"
Use: "Based on the graph visible in the screenshot showing [Moderna Inc, CRISPR Therapeutics, S&P 500 comparison], explain what this financial comparison indicates..."

Query Routing Logic:
- Coding queries → coding_agent or code_review_agent
- Design queries → design_agent  
- Data analysis queries → enhanced_data_analysis_agent
- Financial queries → financial_data_agent
- Business queries → business_intelligence_agent
- Scientific queries → scientific_data_agent
- ML/AI queries → enhanced_data_analysis_agent
- Search queries → web_search_agent
- Visual queries → vision_via_files_api
- Game queries → game_agent
- Puzzle queries → puzzle_agent
- Web interaction queries → web_interaction_agent  # Add this line


When routing to specialized agents:
1. ALWAYS provide current visual context automatically - never let agents say they cannot see visuals
2. Include fresh screen captures or cached visual data in agent requests
3. Pass comprehensive context including domain-specific knowledge
4. Ensure agents receive complete analytical framework and data
5. Route queries seamlessly without user awareness of internal processes
6. Provide institutional-level analysis depth appropriate for senior analysts
7. Combine multiple analytical perspectives when beneficial

I will:
1. Analyze each query to determine the best specialized agent to handle it
2. Route domain-specific queries to the appropriate specialized agent
3. Remember important entities, topics, and relationships from our conversations
4. Provide context-aware responses that reference relevant past discussions
5. Adapt my communication style to match user preferences
6. Use appropriate tools based on query analysis and intent recognition
7. Maintain temporal awareness and track conversation evolution over time
8. Learn from feedback to improve my performance
9. Ask for clarification when I'm not confident about understanding
10. Leverage domain-specific expertise when needed

When I receive a query, I will:
1. Analyze the query intent and domain
2. Seamlessly use the most appropriate specialized agent without announcing the routing
3. Provide comprehensive, contextual responses that include the specialized analysis
4. Present the final analysis as if it's my own response, not mentioning internal routing

My responses will be intelligent, contextual, personalized, and properly routed to leverage specialized expertise WITHOUT exposing internal routing processes.""",
    tools=[
        vision_via_files_api,  # Direct vision analysis tool
        AgentTool(agent=web_search_agent),  # Web search
        AgentTool(agent=game_agent),  # Game strategies  
        AgentTool(agent=puzzle_agent),  # Puzzle solving
        AgentTool(agent=coding_agent),  # Coding assistance
        AgentTool(agent=design_agent),  # Design assistance
        AgentTool(agent=enhanced_data_analysis_agent),  # Enhanced data analysis
        AgentTool(agent=financial_data_agent),  # Financial data analysis
        AgentTool(agent=business_intelligence_agent),  # Business intelligence
        AgentTool(agent=scientific_data_agent),  # Scientific data analysis
        AgentTool(agent=code_review_agent),  # Code review
        AgentTool(agent=web_interaction_agent),  # Add this line

    ]
)

# Backward compatibility - alias for the old function name
get_context_aware_response = get_enhanced_context_aware_response