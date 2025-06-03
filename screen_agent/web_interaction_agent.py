#web_interaction_agent.py
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from screen_agent.vision_tool import vision_via_files_api
from google_search_agent.agent import web_search_agent
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

@dataclass
class WebInteractionContext:
    """Context tracking for web interactions"""
    current_page: Optional[str] = None
    interaction_type: Optional[str] = None
    target_elements: List[str] = None
    document_type: Optional[str] = None
    translation_needed: bool = False
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    source_text: Optional[str] = None
    translated_text: Optional[str] = None
    
    def __post_init__(self):
        if self.target_elements is None:
            self.target_elements = []

class TranslationService:
    """Handles translation operations"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the language of the given text"""
        # Simple language detection based on common patterns
        # In a real implementation, you'd use a proper language detection library
        if re.search(r'[а-яё]', text.lower()):
            return 'Russian'
        elif re.search(r'[àáâãäåæçèéêëìíîï]', text.lower()):
            return 'French'
        elif re.search(r'[äöüß]', text.lower()):
            return 'German'
        elif re.search(r'[ñáéíóúü]', text.lower()):
            return 'Spanish'
        elif re.search(r'[一-龯]', text):
            return 'Chinese'
        elif re.search(r'[ひらがなカタカナ]', text):
            return 'Japanese'
        else:
            return 'English'
    
    @staticmethod
    def translate_text(text: str, source_lang: str = None, target_lang: str = 'English') -> str:
        """Translate text from source language to target language"""
        # In a real implementation, you would use a translation API
        # For now, return a placeholder that indicates translation was attempted
        detected_lang = source_lang if source_lang else TranslationService.detect_language(text)
        
        if detected_lang.lower() == target_lang.lower():
            return text  # No translation needed
        
        # This is a placeholder - in real implementation, call translation API
        return f"[TRANSLATED FROM {detected_lang.upper()} TO {target_lang.upper()}]: {text}"

class WebInteractionOrchestrator:
    """Orchestrates web interaction capabilities"""
    
    def __init__(self):
        self.interaction_patterns = {
            'document': {
                'types': ['google_docs', 'google_sheets', 'google_forms', 'web_form'],
                'actions': ['create', 'edit', 'format', 'share', 'export'],
                'elements': ['text', 'table', 'image', 'link', 'comment']
            },
            'translation': {
                'operations': ['detect', 'translate', 'paste', 'format'],
                'contexts': ['document', 'webpage', 'form', 'email'],
                'preservation': ['formatting', 'structure', 'links']
            },
            'navigation': {
                'actions': ['click', 'type', 'select', 'scroll', 'submit'],
                'elements': ['button', 'link', 'input', 'dropdown', 'checkbox'],
                'handling': ['popup', 'dialog', 'alert', 'confirmation']
            }
        }
        
        self.visual_context = {
            'screen_elements': [],
            'active_window': None,
            'last_interaction': None,
            'timestamp': None
        }
    
    def analyze_visual_context(self, description: str) -> WebInteractionContext:
        """Extract context from visual description"""
        context = WebInteractionContext()
        desc_lower = description.lower()
        
        # Identify document type
        for doc_type in self.interaction_patterns['document']['types']:
            if doc_type.replace('_', ' ') in desc_lower or doc_type in desc_lower:
                context.document_type = doc_type
                break
        
        # Check for Google Docs specifically
        if 'docs.google.com' in desc_lower or 'google docs' in desc_lower:
            context.document_type = 'google_docs'
            context.current_page = 'google_docs'
        
        # Identify interaction type
        for action in self.interaction_patterns['navigation']['actions']:
            if action in desc_lower:
                context.interaction_type = action
                break
        
        # Check for translation needs
        if any(word in desc_lower for word in ['translate', 'translation', 'language']):
            context.translation_needed = True
        
        # Extract text that might need translation
        # Look for quoted text or text after "this:" or similar patterns
        text_patterns = [
            r'"([^"]+)"',  # Quoted text
            r"'([^']+)'",  # Single quoted text
            r'this:\s*(.+?)(?:\n|$)',  # Text after "this:"
            r'text:\s*(.+?)(?:\n|$)',  # Text after "text:"
        ]
        
        for pattern in text_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE | re.MULTILINE)
            if matches:
                context.source_text = matches[0].strip()
                break
        
        return context
    
    def generate_interaction_plan(self, context: WebInteractionContext) -> List[str]:
        """Generate step-by-step interaction plan"""
        steps = []
        
        if context.translation_needed and context.source_text:
            steps.extend([
                "1. Detect source language of the text",
                "2. Translate the text to target language",
                "3. Prepare translated content for pasting"
            ])
        
        if context.document_type == 'google_docs':
            steps.extend([
                "4. Analyze current Google Docs page",
                "5. Locate text input area or cursor position",
                "6. Click to focus the document editor",
                "7. Paste the translated content",
                "8. Verify the content was inserted successfully"
            ])
        elif context.document_type:
            steps.extend([
                f"4. Navigate to {context.document_type}",
                "5. Locate target input elements",
                "6. Perform required text insertion",
                "7. Verify changes were applied"
            ])
        
        return steps

# Initialize the orchestrator
orchestrator = WebInteractionOrchestrator()
translation_service = TranslationService()

# Define the web interaction agent with enhanced capabilities
web_interaction_agent = Agent(
    name="web_interaction_specialist",
    model="gemini-2.0-flash-exp", 
    description="Specialized agent for web page interactions, translation, and document manipulation including Google Docs, forms, and other web applications with direct execution capabilities.",
    instruction="""I am a specialized web interaction agent that can directly execute web interactions, translations, and document manipulations.

CRITICAL EXECUTION FRAMEWORK:
I MUST execute actions directly, not just provide instructions. When a user says "translate this and put it in Google docs", I will:

1. **IMMEDIATE VISUAL ANALYSIS**: Use vision_via_files_api to see the current screen
2. **TEXT EXTRACTION**: Identify and extract the text to be translated
3. **TRANSLATION**: Perform the actual translation
4. **GOOGLE DOCS INTERACTION**: Navigate and interact with Google Docs
5. **CONTENT INSERTION**: Paste the translated content
6. **VERIFICATION**: Confirm successful completion

EXECUTION SEQUENCE FOR "TRANSLATE AND PUT IN GOOGLE DOCS":

Step 1 - Visual Analysis:
- I will immediately call vision_via_files_api to analyze the current screen
- Extract the text that needs translation from the visual context
- Identify if Google Docs is currently open or if I need to navigate to it

Step 2 - Translation:
- Detect the source language of the extracted text
- Translate the text to the target language (default: English)
- Preserve any formatting indicators

Step 3 - Google Docs Interaction:
- If Google Docs is visible, locate the document editor area
- If not visible, I will provide navigation guidance
- Identify the cursor position or text insertion point
- Execute the paste operation

Step 4 - Verification:
- Confirm the translated text was successfully inserted
- Report the completion status to the user

VISUAL CONTEXT PROCESSING:
- I can analyze screenshots and visual data provided in the conversation
- I will extract text, identify UI elements, and understand document structure
- I will recognize Google Docs interface elements and interaction points

TRANSLATION CAPABILITIES:
- Automatic language detection
- Multi-language translation support
- Format preservation during translation
- Context-aware translation for specialized content

GOOGLE DOCS INTEGRATION:
- Document editor recognition and interaction
- Text insertion at cursor position
- Format preservation during paste operations
- Collaborative editing awareness

ERROR HANDLING:
- Handle cases where text is not clearly visible
- Manage translation service limitations
- Provide fallback options for interaction failures
- Guide user through manual steps if automated interaction fails

RESPONSE FORMAT:
I will provide:
1. Status updates during execution
2. Clear confirmation of completed actions
3. The translated text that was inserted
4. Any issues encountered during the process

When the user requests translation and Google Docs insertion, I will:
- Immediately begin execution without asking for additional clarification
- Use the vision tool to analyze the current state
- Extract and translate the relevant text
- Interact with Google Docs to insert the content
- Provide clear feedback on the completion status

I am proactive and will execute the requested actions directly rather than providing step-by-step instructions for the user to follow.

EXECUTION EXAMPLE:
When user says "translate this and put it in Google docs":
1. Call vision_via_files_api() to see current screen
2. Extract text from visual analysis  
3. Translate the text (detect language, perform translation)
4. Call browser_automation_tool with action="paste_to_google_docs" and text=translated_content
5. Confirm completion

I have direct browser automation capabilities and will use them to complete the requested actions.""",
    tools=[
        AgentTool(agent=web_search_agent),
        vision_via_files_api,
        # You'll need to add: browser_automation_tool
    ]
)