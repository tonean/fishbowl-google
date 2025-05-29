import os
import sys
import time
import asyncio
import traceback
import signal
import atexit
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# ── 1) Load environment variables ──
try:
    load_dotenv(find_dotenv())
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Failed to load environment variables: {str(e)}")
    sys.exit(1)

API_KEY = os.getenv("GOOGLE_API_KEY")
USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
if not API_KEY:
    logger.error("Missing GOOGLE_API_KEY in environment!")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = USE_VERTEX
logger.info("API configuration set successfully")

# ── 2) ADK & GenAI imports ──
try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
    logger.info("ADK & GenAI imports successful")
except Exception as e:
    logger.error(f"Failed to import ADK & GenAI modules: {str(e)}")
    sys.exit(1)

try:
    from screen_agent.orchestrator_agent import (
        orchestrator, 
        memory_system, 
        get_enhanced_context_aware_response
    )
    logger.info("Orchestrator agent imported successfully")
except Exception as e:
    logger.error(f"Failed to import orchestrator agent: {str(e)}")
    sys.exit(1)

# ── 3) Runner setup ──
try:
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="enhanced_unified_assistant_app",
        agent=orchestrator,
        session_service=session_service
    )
    logger.info("Enhanced runner setup complete")
except Exception as e:
    logger.error(f"Failed to setup runner: {str(e)}")
    sys.exit(1)

USER_ID = "enhanced_screen_user"
SESSION_ID = None
CONVERSATION_ID = None

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.query_count = 0
        self.total_response_time = 0.0
        self.start_time = time.time()
        
    def record_query(self, response_time: float):
        self.query_count += 1
        self.total_response_time += response_time
        
    def get_stats(self) -> dict:
        uptime = time.time() - self.start_time
        avg_response_time = self.total_response_time / self.query_count if self.query_count > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_queries': self.query_count,
            'average_response_time': avg_response_time,
            'memory_stats': memory_system.get_statistics()
        }

perf_monitor = PerformanceMonitor()

async def initialize_session():
    """Initialize session with enhanced error handling and logging"""
    global SESSION_ID, CONVERSATION_ID
    try:
        logger.info("Starting enhanced session initialization")
        session = await session_service.create_session(
            app_name=runner.app_name,
            user_id=USER_ID
        )
        SESSION_ID = session.id
        
        # Create a new conversation in the enhanced memory system
        CONVERSATION_ID = memory_system.create_conversation(
            title=f"Enhanced Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Add a system message to initialize the conversation
        memory_system.add_message(
            CONVERSATION_ID,
            "system", 
            "Enhanced AI Assistant session initialized with advanced memory and context awareness.",
            metadata={
                "session_id": SESSION_ID,
                "initialization_time": datetime.utcnow().isoformat(),
                "features": ["enhanced_memory", "entity_extraction", "topic_modeling", "user_preferences"]
            }
        )
        
        print("INIT: Enhanced AI Assistant initialized successfully!")
        print(f"INIT: Advanced features active: Memory System, Entity Tracking, Context Awareness")
        logger.info(f"Session initialized - ID: {SESSION_ID}, Conversation: {CONVERSATION_ID}")
        
        # Log initial memory statistics
        stats = memory_system.get_statistics()
        logger.info(f"Memory system stats: {stats}")
        
    except Exception as e:
        logger.error(f"Failed to initialize session: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

async def handle_user_feedback(action_id: str, feedback: str, feedback_type: str) -> None:
    """Handle user feedback for an agent action"""
    try:
        # Calculate confidence adjustment based on feedback type
        if feedback_type == "positive":
            confidence_adjustment = 0.1
        elif feedback_type == "negative":
            confidence_adjustment = -0.1
        else:  # neutral
            confidence_adjustment = 0.0
        
        # Add feedback to memory system
        memory_system.add_action_feedback(
            action_id=action_id,
            user_feedback=feedback,
            feedback_type=feedback_type,
            confidence_adjustment=confidence_adjustment,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "source": "user_feedback"
            }
        )
        
        logger.info(f"Recorded user feedback for action {action_id}: {feedback_type}")
        
    except Exception as e:
        logger.error(f"Error handling user feedback: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

async def call_enhanced_agent(query: str) -> str:
    """Enhanced agent call with improved error handling and performance monitoring"""
    if not SESSION_ID or not CONVERSATION_ID:
        raise RuntimeError("Session or conversation not initialized")
        
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {query[:100]}...")
        
        # Add user message to enhanced memory system (this will extract entities, topics, etc.)
        user_message = memory_system.add_message(CONVERSATION_ID, "user", query)
        logger.info(f"User message processed - Entities: {len(user_message.entities)}, Topics: {user_message.topics}")
        
        # Get enhanced context-aware prompt
        context_prompt = get_enhanced_context_aware_response(query, CONVERSATION_ID)
        logger.debug(f"Context prompt generated: {len(context_prompt)} characters")
        
        # Check if clarification is needed
        if context_prompt.startswith("I'm not entirely confident"):
            return context_prompt
        
        # Create content for the agent
        content = Content(role="user", parts=[Part(text=context_prompt)])
        
        # Run the agent
        events = runner.run(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        )
        
        response_text = ""
        events_list = await asyncio.to_thread(lambda: list(events))
        
        for ev in events_list:
            if ev.is_final_response():
                response_text = ev.content.parts[0].text
                break
                
        if response_text:
            # Add assistant response to enhanced memory system
            assistant_message = memory_system.add_message(
                CONVERSATION_ID, 
                "assistant", 
                response_text,
                metadata={
                    "response_time": time.time() - start_time,
                    "context_length": len(context_prompt),
                    "query_type": "enhanced"
                }
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            perf_monitor.record_query(response_time)
            
            logger.info(f"Response generated - Time: {response_time:.2f}s, Entities: {len(assistant_message.entities)}")
            
            return response_text
        else:
            logger.warning("No response generated from agent")
            return "I apologize, but I wasn't able to generate a response. Please try rephrasing your query."
        
    except Exception as e:
        logger.error(f"Error in enhanced agent call: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Add error to memory system for learning
        try:
            memory_system.add_message(
                CONVERSATION_ID,
                "system",
                f"Error occurred while processing query: {str(e)}",
                metadata={"error": True, "query": query}
            )
        except:
            pass  # Don't let memory errors cascade
            
        raise

async def process_enhanced_input():
    """Enhanced input processing with command handling"""
    print("INIT: Enhanced AI Assistant ready for interaction!")
    print("INIT: Type 'help' for available commands or just ask questions naturally")
    sys.stdout.flush()
    
    for line in sys.stdin:
        query = line.strip()
        if not query:
            continue
            
        # Handle special commands
        if query.lower() == 'help':
            print_help()
            continue
        elif query.lower() == 'stats':
            print_stats()
            continue
        elif query.lower() == 'memory':
            print_memory_stats()
            continue
        elif query.lower().startswith('search_entity:'):
            entity_name = query[14:].strip()
            search_entity(entity_name)
            continue
        elif query.lower().startswith('search_topic:'):
            topic = query[13:].strip()
            search_topic(topic)
            continue
        elif query.lower() == 'cleanup':
            cleanup_memory()
            continue
        elif query.lower() in ['quit', 'exit', 'bye']:
            print("SYSTEM: Goodbye! Session data has been saved.")
            break
            
        try:
            response = await call_enhanced_agent(query)
            print(response)
            sys.stdout.flush()
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            sys.stdout.flush()

def print_help():
    """Print available commands"""
    help_text = """
ENHANCED AI ASSISTANT - Available Commands:
==========================================
help                    - Show this help message
stats                   - Show performance statistics  
memory                  - Show memory system statistics
search_entity: <name>   - Search for messages mentioning an entity
search_topic: <topic>   - Search for messages about a topic
cleanup                 - Clean up old memory data
quit/exit/bye          - Exit the assistant

You can also ask questions naturally - the assistant will use context and memory automatically!
Examples:
- "What did we discuss about Python earlier?"
- "Remember what I said about my project?"
- "Show me information related to machine learning"
"""
    print(help_text)

def print_stats():
    """Print performance statistics"""
    stats = perf_monitor.get_stats()
    print(f"""
PERFORMANCE STATISTICS:
======================
Uptime: {stats['uptime_seconds']:.2f} seconds
Total Queries: {stats['total_queries']}
Average Response Time: {stats['average_response_time']:.2f} seconds

MEMORY SYSTEM:
{stats['memory_stats']}
""")

def print_memory_stats():
    """Print detailed memory statistics"""
    if CONVERSATION_ID:
        stats = memory_system.get_statistics()
        user_prefs = memory_system.get_user_preferences(CONVERSATION_ID)
        
        print(f"""
MEMORY SYSTEM DETAILS:
=====================
Conversations: {stats['conversations']}
Total Messages: {stats['total_messages']}
Tracked Entities: {stats['total_entities']}
Knowledge Graph Nodes: {stats['knowledge_graph_nodes']}
Knowledge Graph Edges: {stats['knowledge_graph_edges']}
Storage Size: {stats['storage_size_mb']:.2f} MB

USER PREFERENCES:
{user_prefs}
""")

def search_entity(entity_name: str):
    """Search for messages mentioning a specific entity"""
    try:
        messages = memory_system.search_by_entity(entity_name, limit=5)
        if messages:
            print(f"\nFound {len(messages)} messages mentioning '{entity_name}':")
            for i, msg in enumerate(messages, 1):
                print(f"{i}. [{msg.timestamp[:19]}] {msg.role}: {msg.content[:100]}...")
        else:
            print(f"No messages found mentioning '{entity_name}'")
    except Exception as e:
        print(f"Error searching for entity: {e}")

def search_topic(topic: str):
    """Search for messages about a specific topic"""
    try:
        messages = memory_system.search_by_topic(topic, limit=5)
        if messages:
            print(f"\nFound {len(messages)} messages about '{topic}':")
            for i, msg in enumerate(messages, 1):
                print(f"{i}. [{msg.timestamp[:19]}] {msg.role}: {msg.content[:100]}...")
        else:
            print(f"No messages found about '{topic}'")
    except Exception as e:
        print(f"Error searching for topic: {e}")

def cleanup_memory():
    """Clean up old memory data"""
    try:
        print("Cleaning up old memory data...")
        memory_system.cleanup_old_data(days_to_keep=30)
        print("Memory cleanup completed successfully!")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def shutdown_handler(signum, frame):
    """Handle shutdown gracefully"""
    logger.info("Shutdown signal received, cleaning up...")
    try:
        # Perform any necessary cleanup
        stats = perf_monitor.get_stats()
        logger.info(f"Final stats: {stats}")
        print("\nSYSTEM: Graceful shutdown completed.")
    except:
        pass
    sys.exit(0)

def cleanup_on_exit():
    """Cleanup function called on exit"""
    logger.info("Application exiting, performing cleanup...")
    try:
        # Save any pending data
        if CONVERSATION_ID:
            stats = memory_system.get_statistics()
            logger.info(f"Final memory stats: {stats}")
    except:
        pass

async def main():
    """Enhanced main function with better error handling"""
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        atexit.register(cleanup_on_exit)
        
        await initialize_session()
        await process_enhanced_input()
        
    except KeyboardInterrupt:
        print("\nSYSTEM: Interrupted by user")
        logger.info("Application interrupted by user")
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        print(f"ERROR: {error_msg}")
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())