# app/screen_agent/enhanced_orchestrator_agent.py
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Content, Part
from screen_agent.vision_tool import vision_via_files_api
from screen_agent.memory_system import EnhancedMemorySystem
from google_search_agent.agent import web_search_agent
from screen_agent.agent import game_agent, puzzle_agent
from typing import Dict, List, Any, Optional
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Initialize enhanced memory system
memory_system = EnhancedMemorySystem()

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
        'entity_focused': False
    }
    
    # Detect query type
    if any(word in query_lower for word in ['search', 'find', 'look up', 'google']):
        intent_analysis['type'] = 'search'
        intent_analysis['requires_search'] = True
    elif any(word in query_lower for word in ['screen', 'image', 'picture', 'see', 'show']):
        intent_analysis['type'] = 'vision'
        intent_analysis['requires_vision'] = True
    elif any(word in query_lower for word in ['remember', 'recall', 'mentioned', 'said before', 'earlier']):
        intent_analysis['type'] = 'memory'
        intent_analysis['requires_memory'] = True
    elif any(word in query_lower for word in ['game', 'puzzle', 'play', 'solve']):
        intent_analysis['type'] = 'interactive'
    
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
    """
    Main function to get enhanced context-aware response with feedback loop
    """
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
        
        # Record the action
        action_id = memory_system.record_agent_action(
            agent_name="orchestrator",
            action_type="query_analysis",
            input_data={"query": query, "intent": query_intent},
            output_data={"context": context_response, "system_prompt": system_context},
            confidence=confidence_score
        )
        
        # Combine everything into a comprehensive prompt
        enhanced_prompt = f"""{system_context}

{context_response}

Query Analysis: {json.dumps(query_intent, indent=2)}
Confidence Score: {confidence_score:.2f}

Please provide a response that leverages all available context and capabilities."""

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
    
    # Calculate final confidence score
    return sum(confidence_factors) / len(confidence_factors)

# Add feedback handling to the orchestrator agent
orchestrator = Agent(
    name="enhanced_unified_assistant",
    model="gemini-2.0-flash-exp",
    description="Advanced context-aware assistant with intelligent memory, entity tracking, and adaptive responses.",
    instruction="""I am an advanced AI assistant with sophisticated memory and context awareness capabilities.

Key Features:
- Enhanced Memory: I maintain detailed conversation history with entity extraction, topic modeling, and relationship tracking
- Smart Context: I provide adaptive context based on query type, user preferences, and conversation patterns  
- Intelligent Search: I can search for and reference relevant information from our conversation history
- Personalization: I learn and adapt to user communication preferences and interests
- Multi-modal: I can analyze screens, search the web, help with games, and solve puzzles
- Feedback Loop: I learn from user feedback to improve my responses over time
- Confidence Scoring: I know when to ask for clarification based on my confidence level

I will:
1. Remember important entities, topics, and relationships from our conversations
2. Provide context-aware responses that reference relevant past discussions
3. Adapt my communication style to match user preferences
4. Use appropriate tools based on query analysis and intent recognition
5. Maintain temporal awareness and track conversation evolution over time
6. Learn from feedback to improve my performance
7. Ask for clarification when I'm not confident about understanding

My responses will be intelligent, contextual, and personalized while being helpful and accurate.""",
    tools=[
        vision_via_files_api,  # Direct vision analysis tool
        AgentTool(agent=web_search_agent),  # Web search
        AgentTool(agent=game_agent),  # Game strategies  
        AgentTool(agent=puzzle_agent),  # Puzzle solving
    ]
)

# Backward compatibility - alias for the old function name
get_context_aware_response = get_enhanced_context_aware_response