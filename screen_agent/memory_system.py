from typing import Dict, List, Optional, Any, Set, Tuple
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta
import networkx as nx
from dataclasses import dataclass, asdict
import uuid
import re
from collections import defaultdict, Counter
import hashlib
import pickle
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a named entity extracted from conversations"""
    name: str
    entity_type: str  # person, place, organization, concept, etc.
    confidence: float
    first_mentioned: str
    last_mentioned: str
    frequency: int
    context: List[str]  # Surrounding text where entity was found
    
@dataclass
class Relationship:
    """Represents a relationship between entities"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_message_id: str
    timestamp: str

@dataclass
class TopicCluster:
    """Represents a cluster of related topics/concepts"""
    id: str
    name: str
    keywords: List[str]
    messages: List[str]  # message IDs
    strength: float
    created_at: str
    updated_at: str

@dataclass
class Message:
    id: str
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]
    entities: List[Entity]
    topics: List[str]
    sentiment: float
    importance: float
    embedding: Optional[List[float]] = None  # For semantic similarity
    
@dataclass
class ConversationSummary:
    """Summary of conversation segments"""
    id: str
    conversation_id: str
    start_message_id: str
    end_message_id: str
    summary: str
    key_points: List[str]
    entities: List[str]
    timestamp: str
    
@dataclass
class Conversation:
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    entities: Dict[str, Entity]
    topics: Dict[str, TopicCluster]
    summaries: List[ConversationSummary]
    user_preferences: Dict[str, Any]

@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    id: str
    agent_name: str
    action_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    timestamp: str
    success: bool
    feedback: Optional[Dict[str, Any]] = None

@dataclass
class ActionFeedback:
    """Represents feedback for an agent action"""
    action_id: str
    user_feedback: str
    feedback_type: str  # positive, negative, neutral
    confidence_adjustment: float
    timestamp: str
    metadata: Dict[str, Any]

class EntityExtractor(ABC):
    """Abstract base class for entity extraction"""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        pass

class SimpleEntityExtractor(EntityExtractor):
    """Simple rule-based entity extractor"""
    
    def __init__(self):
        # Common patterns for different entity types
        self.patterns = {
            'person': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',  # Title Name
            ],
            'location': [
                r'\b[A-Z][a-z]+ (?:City|Town|Street|Avenue|Road|State|Country)\b',
                r'\bin [A-Z][a-z]+(?:, [A-Z][a-z]+)*\b',
            ],
            'organization': [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
                r'\b(?:Apple|Google|Microsoft|Amazon|Facebook|Meta)\b',
            ],
            'date': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            ],
            'number': [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        timestamp = datetime.utcnow().isoformat()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 2:  # Filter out very short matches
                        entities.append(Entity(
                            name=entity_text,
                            entity_type=entity_type,
                            confidence=0.8,  # Rule-based, so moderate confidence
                            first_mentioned=timestamp,
                            last_mentioned=timestamp,
                            frequency=1,
                            context=[text[max(0, match.start()-50):match.end()+50]]
                        ))
        
        return entities

class EnhancedMemorySystem:
    def __init__(self, storage_dir: str = "enhanced_memory"):
        self.storage_dir = Path(storage_dir)
        self.conversations: Dict[str, Conversation] = {}
        self.knowledge_graph = nx.MultiDiGraph()
        self.entity_extractor = SimpleEntityExtractor()
        self.lock = threading.RLock()
        
        # Database for persistent storage
        self.db_path = self.storage_dir / "memory.db"
        
        # In-memory indexes for fast retrieval
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity -> message_ids
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)   # topic -> message_ids
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> message_ids
        
        self._ensure_storage_dir()
        self._init_database()
        self._load_conversations()
        
    def _ensure_storage_dir(self):
        """Ensure the storage directory exists"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    sentiment REAL,
                    importance REAL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                );
                
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    entity_type TEXT,
                    confidence REAL,
                    first_mentioned TEXT,
                    last_mentioned TEXT,
                    frequency INTEGER,
                    context TEXT
                );
                
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    confidence REAL,
                    source_message_id TEXT,
                    timestamp TEXT
                );
                
                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    start_message_id TEXT,
                    end_message_id TEXT,
                    summary TEXT,
                    key_points TEXT,
                    entities TEXT,
                    timestamp TEXT
                );
                
                CREATE TABLE IF NOT EXISTS agent_actions (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT,
                    action_type TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    success BOOLEAN,
                    feedback TEXT
                );
                
                CREATE TABLE IF NOT EXISTS action_feedback (
                    id TEXT PRIMARY KEY,
                    action_id TEXT,
                    user_feedback TEXT,
                    feedback_type TEXT,
                    confidence_adjustment REAL,
                    timestamp TEXT,
                    metadata TEXT,
                    FOREIGN KEY (action_id) REFERENCES agent_actions (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
                CREATE INDEX IF NOT EXISTS idx_agent_actions_agent ON agent_actions(agent_name);
                CREATE INDEX IF NOT EXISTS idx_agent_actions_type ON agent_actions(action_type);
                CREATE INDEX IF NOT EXISTS idx_action_feedback_action ON action_feedback(action_id);
            """)
            
    def _load_conversations(self):
        """Load conversations from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Load conversations
                conversations = conn.execute("SELECT * FROM conversations").fetchall()
                for conv_row in conversations:
                    conv_id = conv_row['id']
                    
                    # Load messages for this conversation
                    messages = conn.execute(
                        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
                        (conv_id,)
                    ).fetchall()
                    
                    message_objects = []
                    for msg_row in messages:
                        message = Message(
                            id=msg_row['id'],
                            role=msg_row['role'],
                            content=msg_row['content'],
                            timestamp=msg_row['timestamp'],
                            metadata=json.loads(msg_row['metadata'] or '{}'),
                            entities=[],
                            topics=[],
                            sentiment=msg_row['sentiment'] or 0.0,
                            importance=msg_row['importance'] or 0.0
                        )
                        message_objects.append(message)
                    
                    conversation = Conversation(
                        id=conv_id,
                        title=conv_row['title'],
                        messages=message_objects,
                        created_at=conv_row['created_at'],
                        updated_at=conv_row['updated_at'],
                        metadata=json.loads(conv_row['metadata'] or '{}'),
                        entities={},
                        topics={},
                        summaries=[],
                        user_preferences={}
                    )
                    
                    self.conversations[conv_id] = conversation
                    
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
            
    def _save_conversation(self, conversation: Conversation):
        """Save conversation to database"""
        with sqlite3.connect(self.db_path) as conn:
            # Save conversation
            conn.execute("""
                INSERT OR REPLACE INTO conversations 
                (id, title, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation.id,
                conversation.title,
                conversation.created_at,
                conversation.updated_at,
                json.dumps(conversation.metadata)
            ))
            
            # Save messages
            for message in conversation.messages:
                conn.execute("""
                    INSERT OR REPLACE INTO messages
                    (id, conversation_id, role, content, timestamp, metadata, sentiment, importance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    conversation.id,
                    message.role,
                    message.content,
                    message.timestamp,
                    json.dumps(message.metadata),
                    message.sentiment,
                    message.importance
                ))
            
            conn.commit()
            
    def create_conversation(self, title: str = "New Conversation") -> str:
        """Create a new conversation and return its ID"""
        with self.lock:
            conv_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            conversation = Conversation(
                id=conv_id,
                title=title,
                messages=[],
                created_at=now,
                updated_at=now,
                metadata={},
                entities={},
                topics={},
                summaries=[],
                user_preferences={}
            )
            self.conversations[conv_id] = conversation
            self._save_conversation(conversation)
            return conv_id
            
    def add_message(self, conversation_id: str, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to a conversation with enhanced processing"""
        with self.lock:
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Extract entities from content
            entities = self.entity_extractor.extract_entities(content)
            
            # Calculate sentiment (simple word-based approach)
            sentiment = self._calculate_sentiment(content)
            
            # Calculate importance (based on length, entities, keywords)
            importance = self._calculate_importance(content, entities)
            
            # Extract topics
            topics = self._extract_topics(content)
            
            message = Message(
                id=str(uuid.uuid4()),
                role=role,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
                metadata=metadata or {},
                entities=entities,
                topics=topics,
                sentiment=sentiment,
                importance=importance
            )
            
            conversation = self.conversations[conversation_id]
            conversation.messages.append(message)
            conversation.updated_at = datetime.utcnow().isoformat()
            
            # Update indexes
            self._update_indexes(message, conversation_id)
            
            # Update knowledge graph
            self._update_knowledge_graph(message, entities)
            
            # Update entities in conversation
            self._update_conversation_entities(conversation, entities)
            
            # Auto-summarize if conversation is getting long
            if len(conversation.messages) % 20 == 0:  # Every 20 messages
                self._auto_summarize(conversation)
            
            self._save_conversation(conversation)
            
            return message
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on word lists"""
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'happy', 'pleased', 'satisfied', 'love', 'like', 'enjoy'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                         'angry', 'frustrated', 'disappointed', 'sad', 'upset'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_importance(self, content: str, entities: List[Entity]) -> float:
        """Calculate message importance based on various factors"""
        factors = []
        
        # Length factor (longer messages might be more important)
        factors.append(min(len(content) / 1000, 1.0))  # Normalize to 0-1
        
        # Entity factor (more entities = more important)
        factors.append(min(len(entities) / 10, 1.0))
        
        # Question factor (questions might be important)
        if '?' in content:
            factors.append(0.3)
        
        # Keyword factor (presence of important keywords)
        important_keywords = {'important', 'urgent', 'critical', 'remember', 'note'}
        if any(keyword in content.lower() for keyword in important_keywords):
            factors.append(0.5)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content using simple keyword extraction"""
        # This is a simplified approach - in production, you might use TF-IDF or more advanced NLP
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = Counter(words)
        
        # Common stop words to filter out
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 
                     'have', 'were', 'been', 'their', 'said', 'each', 'which', 'what', 
                     'where', 'when', 'more', 'very', 'some', 'like', 'into', 'time', 
                     'only', 'just', 'first', 'after', 'most', 'also', 'other', 'many'}
        
        topics = []
        for word, freq in word_freq.most_common(5):
            if word not in stop_words and len(word) > 3:
                topics.append(word)
        
        return topics
    
    def _update_indexes(self, message: Message, conversation_id: str):
        """Update in-memory indexes for fast retrieval"""
        # Entity index
        for entity in message.entities:
            self.entity_index[entity.name.lower()].add(message.id)
        
        # Topic index
        for topic in message.topics:
            self.topic_index[topic].add(message.id)
        
        # Temporal index
        date_key = message.timestamp[:10]  # YYYY-MM-DD
        self.temporal_index[date_key].append(message.id)
    
    def _update_knowledge_graph(self, message: Message, entities: List[Entity]):
        """Update knowledge graph with enhanced relationship extraction"""
        # Add entities as nodes
        for entity in entities:
            self.knowledge_graph.add_node(entity.name, 
                                        entity_type=entity.entity_type,
                                        confidence=entity.confidence)
        
        # Extract relationships between consecutive entities
        for i in range(len(entities) - 1):
            entity1 = entities[i]
            entity2 = entities[i + 1]
            
            # Add edge with context
            self.knowledge_graph.add_edge(
                entity1.name, 
                entity2.name,
                weight=1,
                message_id=message.id,
                context=message.content[:100]  # First 100 chars as context
            )
        
        # Add topic-entity relationships
        for topic in message.topics:
            for entity in entities:
                self.knowledge_graph.add_edge(
                    topic,
                    entity.name,
                    weight=0.5,
                    relationship_type='topic_entity',
                    message_id=message.id
                )
    
    def _update_conversation_entities(self, conversation: Conversation, new_entities: List[Entity]):
        """Update conversation-level entity tracking"""
        for entity in new_entities:
            if entity.name in conversation.entities:
                # Update existing entity
                existing = conversation.entities[entity.name]
                existing.frequency += 1
                existing.last_mentioned = entity.first_mentioned
                existing.context.extend(entity.context)
            else:
                # Add new entity
                conversation.entities[entity.name] = entity
    
    def _auto_summarize(self, conversation: Conversation):
        """Automatically create summaries for long conversations"""
        if len(conversation.messages) < 10:
            return
        
        # Get last 10 messages for summarization
        recent_messages = conversation.messages[-10:]
        
        # Extract key information
        key_entities = set()
        key_topics = set()
        important_points = []
        
        for msg in recent_messages:
            if msg.importance > 0.5:  # Only important messages
                important_points.append(msg.content[:200])  # First 200 chars
            
            key_entities.update(entity.name for entity in msg.entities)
            key_topics.update(msg.topics)
        
        # Create summary
        summary_text = f"Recent discussion involving {len(key_entities)} entities and {len(key_topics)} topics."
        if important_points:
            summary_text += f" Key points: {'; '.join(important_points[:3])}"
        
        summary = ConversationSummary(
            id=str(uuid.uuid4()),
            conversation_id=conversation.id,
            start_message_id=recent_messages[0].id,
            end_message_id=recent_messages[-1].id,
            summary=summary_text,
            key_points=important_points,
            entities=list(key_entities),
            timestamp=datetime.utcnow().isoformat()
        )
        
        conversation.summaries.append(summary)
    
    def get_conversation_history(self, conversation_id: str, 
                               limit: Optional[int] = None,
                               include_summaries: bool = True) -> List[Message]:
        """Get conversation history with optional summarization"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        messages = conversation.messages
        
        if limit and len(messages) > limit:
            if include_summaries and conversation.summaries:
                # Include recent summary + recent messages
                summary_text = conversation.summaries[-1].summary
                recent_messages = messages[-limit:]
                
                # Add summary as a system message
                summary_message = Message(
                    id="summary_" + str(uuid.uuid4()),
                    role="system",
                    content=f"Previous conversation summary: {summary_text}",
                    timestamp=recent_messages[0].timestamp,
                    metadata={"type": "summary"},
                    entities=[],
                    topics=[],
                    sentiment=0.0,
                    importance=1.0
                )
                
                return [summary_message] + recent_messages
            else:
                return messages[-limit:]
        
        return messages
    
    def search_by_entity(self, entity_name: str, limit: int = 10) -> List[Message]:
        """Search messages by entity name"""
        message_ids = self.entity_index.get(entity_name.lower(), set())
        messages = []
        
        for conv in self.conversations.values():
            for msg in conv.messages:
                if msg.id in message_ids:
                    messages.append(msg)
        
        # Sort by importance and recency
        messages.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return messages[:limit]
    
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Message]:
        """Search messages by topic"""
        message_ids = self.topic_index.get(topic.lower(), set())
        messages = []
        
        for conv in self.conversations.values():
            for msg in conv.messages:
                if msg.id in message_ids:
                    messages.append(msg)
        
        messages.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return messages[:limit]
    
    def search_by_timeframe(self, start_date: str, end_date: str) -> List[Message]:
        """Search messages within a time frame"""
        messages = []
        
        for conv in self.conversations.values():
            for msg in conv.messages:
                if start_date <= msg.timestamp <= end_date:
                    messages.append(msg)
        
        return sorted(messages, key=lambda m: m.timestamp)
    
    def get_related_concepts(self, concept: str, max_depth: int = 2, 
                           limit: int = 10) -> List[Tuple[str, float]]:
        """Get concepts related to the given concept with confidence scores"""
        if concept not in self.knowledge_graph:
            return []
        
        related = []
        
        # Get direct neighbors
        for neighbor in self.knowledge_graph.neighbors(concept):
            edge_data = self.knowledge_graph[concept][neighbor]
            weight = sum(data.get('weight', 1) for data in edge_data.values())
            related.append((neighbor, weight))
        
        # Get neighbors at distance 2 if requested
        if max_depth >= 2:
            for intermediate in self.knowledge_graph.neighbors(concept):
                for neighbor in self.knowledge_graph.neighbors(intermediate):
                    if neighbor != concept:
                        edge_data = self.knowledge_graph[intermediate][neighbor]
                        weight = sum(data.get('weight', 1) for data in edge_data.values()) * 0.5
                        related.append((neighbor, weight))
        
        # Remove duplicates and sort by weight
        unique_related = {}
        for name, weight in related:
            if name in unique_related:
                unique_related[name] += weight
            else:
                unique_related[name] = weight
        
        sorted_related = sorted(unique_related.items(), key=lambda x: x[1], reverse=True)
        return sorted_related[:limit]
    
    def get_conversation_context(self, conversation_id: str, 
                               max_messages: int = 5,
                               include_entities: bool = True,
                               include_topics: bool = True) -> str:
        """Get enhanced conversation context"""
        messages = self.get_conversation_history(conversation_id, max_messages)
        
        context_parts = []
        
        # Add message history
        context_parts.append("Recent conversation:")
        for msg in messages[-max_messages:]:
            context_parts.append(f"{msg.role}: {msg.content}")
        
        # Add entity context
        if include_entities and conversation_id in self.conversations:
            entities = self.conversations[conversation_id].entities
            if entities:
                entity_list = [f"{name} ({entity.entity_type})" 
                              for name, entity in list(entities.items())[:5]]
                context_parts.append(f"\nKey entities mentioned: {', '.join(entity_list)}")
        
        # Add topic context
        if include_topics:
            all_topics = []
            for msg in messages:
                all_topics.extend(msg.topics)
            
            if all_topics:
                topic_freq = Counter(all_topics)
                top_topics = [topic for topic, _ in topic_freq.most_common(5)]
                context_parts.append(f"\nMain topics: {', '.join(top_topics)}")
        
        return "\n".join(context_parts)
    
    def get_user_preferences(self, conversation_id: str) -> Dict[str, Any]:
        """Extract user preferences from conversation history"""
        if conversation_id not in self.conversations:
            return {}
        
        conversation = self.conversations[conversation_id]
        preferences = conversation.user_preferences.copy()
        
        # Analyze message patterns to infer preferences
        user_messages = [msg for msg in conversation.messages if msg.role == "user"]
        
        # Communication style preference
        avg_length = sum(len(msg.content) for msg in user_messages) / len(user_messages) if user_messages else 0
        preferences['communication_style'] = 'detailed' if avg_length > 100 else 'concise'
        
        # Topic preferences based on frequency
        all_topics = []
        for msg in user_messages:
            all_topics.extend(msg.topics)
        
        if all_topics:
            topic_freq = Counter(all_topics)
            preferences['favorite_topics'] = [topic for topic, _ in topic_freq.most_common(3)]
        
        # Update stored preferences
        conversation.user_preferences = preferences
        
        return preferences
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage storage"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.isoformat()
        
        with self.lock:
            for conv_id, conversation in list(self.conversations.items()):
                # Remove old messages but keep important ones
                original_count = len(conversation.messages)
                conversation.messages = [
                    msg for msg in conversation.messages
                    if msg.timestamp > cutoff_str or msg.importance > 0.7
                ]
                
                if len(conversation.messages) != original_count:
                    logger.info(f"Cleaned {original_count - len(conversation.messages)} old messages from conversation {conv_id}")
                    self._save_conversation(conversation)
    
    def export_knowledge_graph(self, format: str = 'json') -> str:
        """Export knowledge graph for analysis"""
        if format == 'json':
            data = nx.node_link_data(self.knowledge_graph)
            return json.dumps(data, indent=2)
        elif format == 'gexf':
            return '\n'.join(nx.generate_gexf(self.knowledge_graph))
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())
        total_entities = sum(len(conv.entities) for conv in self.conversations.values())
        
        return {
            'conversations': len(self.conversations),
            'total_messages': total_messages,
            'total_entities': total_entities,
            'knowledge_graph_nodes': self.knowledge_graph.number_of_nodes(),
            'knowledge_graph_edges': self.knowledge_graph.number_of_edges(),
            'storage_size_mb': self._get_storage_size() / (1024 * 1024)
        }
    
    def _get_storage_size(self) -> int:
        """Get total storage size in bytes"""
        total_size = 0
        for file_path in self.storage_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def record_agent_action(self, agent_name: str, action_type: str, 
                           input_data: Dict[str, Any], output_data: Dict[str, Any],
                           confidence: float, success: bool = True) -> str:
        """Record an action taken by an agent"""
        action_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        action = AgentAction(
            id=action_id,
            agent_name=agent_name,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            timestamp=timestamp,
            success=success
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO agent_actions 
                (id, agent_name, action_type, input_data, output_data, confidence, timestamp, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action.id,
                action.agent_name,
                action.action_type,
                json.dumps(action.input_data),
                json.dumps(action.output_data),
                action.confidence,
                action.timestamp,
                action.success
            ))
            conn.commit()
        
        return action_id
    
    def add_action_feedback(self, action_id: str, user_feedback: str,
                          feedback_type: str, confidence_adjustment: float,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add feedback for an agent action"""
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        feedback = ActionFeedback(
            action_id=action_id,
            user_feedback=user_feedback,
            feedback_type=feedback_type,
            confidence_adjustment=confidence_adjustment,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        with sqlite3.connect(self.db_path) as conn:
            # Add feedback
            conn.execute("""
                INSERT INTO action_feedback
                (id, action_id, user_feedback, feedback_type, confidence_adjustment, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback_id,
                feedback.action_id,
                feedback.user_feedback,
                feedback.feedback_type,
                feedback.confidence_adjustment,
                feedback.timestamp,
                json.dumps(feedback.metadata)
            ))
            
            # Update action with feedback
            conn.execute("""
                UPDATE agent_actions
                SET feedback = ?
                WHERE id = ?
            """, (json.dumps(feedback.__dict__), action_id))
            
            conn.commit()
    
    def get_agent_performance(self, agent_name: str, 
                            action_type: Optional[str] = None,
                            time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Base query
            query = """
                SELECT 
                    COUNT(*) as total_actions,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_actions,
                    AVG(CASE WHEN success THEN confidence ELSE 0 END) as success_confidence,
                    AVG(CASE WHEN NOT success THEN confidence ELSE 0 END) as failure_confidence
                FROM agent_actions
                WHERE agent_name = ?
            """
            params = [agent_name]
            
            # Add filters
            if action_type:
                query += " AND action_type = ?"
                params.append(action_type)
            
            if time_window:
                cutoff = (datetime.utcnow() - timedelta(days=time_window)).isoformat()
                query += " AND timestamp > ?"
                params.append(cutoff)
            
            # Get basic metrics
            metrics = conn.execute(query, params).fetchone()
            
            # Get feedback metrics
            feedback_query = """
                SELECT 
                    feedback_type,
                    COUNT(*) as count,
                    AVG(confidence_adjustment) as avg_adjustment
                FROM action_feedback af
                JOIN agent_actions aa ON af.action_id = aa.id
                WHERE aa.agent_name = ?
            """
            feedback_params = [agent_name]
            
            if action_type:
                feedback_query += " AND aa.action_type = ?"
                feedback_params.append(action_type)
            
            if time_window:
                feedback_query += " AND af.timestamp > ?"
                feedback_params.append(cutoff)
            
            feedback_query += " GROUP BY feedback_type"
            
            feedback_metrics = conn.execute(feedback_query, feedback_params).fetchall()
            
            # Combine results
            result = dict(metrics)
            result['feedback_metrics'] = [dict(f) for f in feedback_metrics]
            
            return result
    
    def should_request_clarification(self, agent_name: str, action_type: str,
                                   confidence: float) -> bool:
        """Determine if human clarification should be requested"""
        # Get historical performance
        performance = self.get_agent_performance(agent_name, action_type, time_window=30)
        
        if not performance['total_actions']:
            # No history, use default threshold
            return confidence < 0.7
        
        # Calculate dynamic threshold based on historical performance
        success_rate = performance['successful_actions'] / performance['total_actions']
        avg_confidence = performance['avg_confidence']
        
        # Adjust threshold based on success rate
        if success_rate < 0.5:
            # Poor performance, be more conservative
            threshold = 0.8
        elif success_rate > 0.9:
            # Excellent performance, can be more aggressive
            threshold = 0.6
        else:
            # Normal performance
            threshold = 0.7
        
        # Further adjust based on average confidence
        if avg_confidence < 0.6:
            threshold += 0.1
        elif avg_confidence > 0.9:
            threshold -= 0.1
        
        return confidence < threshold