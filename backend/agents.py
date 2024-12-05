from typing import Dict, Any, Optional, List
import logging
from groq import Groq
from langchain_groq import ChatGroq
import json
from enum import Enum

class AgentRole(Enum):
    USER = "user"
    RAG = "rag"
    SYSTEM = "system"

class Message:
    def __init__(self, content: str, role: AgentRole, metadata: Optional[Dict] = None):
        self.content = content
        self.role = role
        self.metadata = metadata or {}

class UserAgent:
    """Agent that handles user interactions and query formatting"""
    
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = """You are a friendly assistant who:
        1. Understands user's conversational style
        2. Adapts responses to match user's tone
        3. Maintains natural conversation flow
        4. Guides the RAG agent appropriately
        """

    def format_query(self, query: str) -> Message:
        """Format user query with conversation-aware prompting"""
        query_analysis = self._analyze_query(query)
        conversation_context = self._get_relevant_history()
        
        # Create the prompt for RAG agent
        rag_prompt = self._create_rag_prompt(
            query=query,
            query_analysis=query_analysis,
            conversation_context=conversation_context
        )

        formatted_query = {
            "type": "query",
            "content": query,
            "require_context": True,
            "query_analysis": query_analysis,
            "conversation_context": conversation_context,
            "rag_prompt": rag_prompt
        }

        message = Message(
            content=query,
            role=AgentRole.USER,
            metadata=formatted_query
        )
        self.conversation_history.append(message)
        return message

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for tone, style, and intent"""
        query_lower = query.lower()
        
        # Detect conversation style
        style = {
            'casual': any(word in query_lower for word in ['hey', 'hi', 'hello', 'thanks', 'please']),
            'formal': any(word in query_lower for word in ['could you', 'would you', 'kindly']),
            'direct': any(word in query_lower for word in ['tell me', 'show', 'what is', 'list']),
            'curious': any(word in query_lower for word in ['why', 'how come', 'what if', 'curious']),
        }

        # Determine response type needed
        response_type = self._determine_response_type(query)

        # Check if query is follow-up
        is_follow_up = any(word in query_lower for word in ['also', 'and', 'what about', 'then', 'so'])

        return {
            'style': style,
            'response_type': response_type,
            'is_follow_up': is_follow_up,
            'tone': 'casual' if style['casual'] else 'formal' if style['formal'] else 'neutral'
        }

    def _create_rag_prompt(self, query: str, query_analysis: Dict[str, Any], conversation_context: List[Dict]) -> str:
        """Create context-aware prompt for RAG agent"""
        style = query_analysis['style']
        tone = query_analysis['tone']
        is_follow_up = query_analysis['is_follow_up']

        # Build style guidance based on analysis
        style_guide = []
        if style['casual']:
            style_guide.append("Keep the tone friendly and casual")
        if style['formal']:
            style_guide.append("Maintain a polite, professional tone")
        if style['direct']:
            style_guide.append("Be clear and straightforward")
        if style['curious']:
            style_guide.append("Provide explanatory details")

        # Create appropriate response format
        response_format = "Brief and direct" if style['direct'] else "Conversational and detailed"

        return f"""Let's help answer this question in a natural way.

        Question: {query}
        Conversation Style: {tone}
        Response Type: {query_analysis['response_type']}
        Follow-up Question: {'Yes' if is_follow_up else 'No'}

        Style Guidelines:
        {chr(10).join(f"- {guide}" for guide in style_guide)}

        Response Format: {response_format}

        Key Instructions:
        1. Match the user's conversational style
        2. {'Acknowledge the follow-up nature' if is_follow_up else 'Start fresh with this topic'}
        3. Keep the response {response_format.lower()}
        4. Use natural language and transitions
        5. Stay focused on the specific question

        Previous Context:
        {json.dumps(conversation_context[-2:], indent=2) if conversation_context else 'No previous context'}

        Please provide a {tone}, focused response that matches the user's style.
        """

    def _determine_response_type(self, query: str) -> str:
        """Determine the type of response needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['explain', 'how', 'why']):
            return 'explanation'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'
        elif any(word in query_lower for word in ['list', 'what are', 'tell me about']):
            return 'list'
        elif any(word in query_lower for word in ['summarize', 'brief', 'quick']):
            return 'summary'
        elif any(word in query_lower for word in ['yes', 'no', 'is it', 'does it']):
            return 'confirmation'
        return 'general'

    def _get_relevant_history(self) -> List[Dict]:
        """Get relevant conversation history"""
        return [
            {
                'role': msg.role.value,
                'content': msg.content,
                'metadata': msg.metadata
            }
            for msg in self.conversation_history[-3:]  # Last 3 messages
        ]

    def process_response(self, response: Message) -> Dict[str, Any]:
        """Process RAG agent response and format for display"""
        self.conversation_history.append(response)
        return {
            "answer": response.content,
            "metadata": response.metadata.get("document_metadata", {}),
            "context": response.metadata.get("context", ""),
            "success": True
        }

class RAGAgent:
    """Agent that handles retrieval and response generation"""
    
    def __init__(self, groq_api_key: str, config: Dict[str, Any] = None):
        self.config = {
            "temperature": 0,
            "model": "llama-3.1-70b-versatile",
            "api_key": groq_api_key
        } if config is None else config

        self.groq_client = Groq(api_key=self.config["api_key"])
        self.groq_llm = ChatGroq(
            temperature=self.config["temperature"],
            model=self.config["model"],
            api_key=self.config["api_key"]
        )
        self.conversation_history = []
        self.system_prompt = """You are a friendly assistant who:
        1. Gives clear, focused answers
        2. Uses natural, conversational language
        3. Keeps things simple and relevant
        4. Maintains a warm, helpful tone

        Style:
        - Be direct but friendly
        - Use clear, everyday language
        - Keep responses focused
        - Be conversational but concise
        """

    def generate_response(self, query: Message, context: Dict[str, Any]) -> Message:
        """Generate response using context and query"""
        try:
            prompt = self._create_prompt(query, context)
            llm_response = self.groq_llm.invoke(prompt)
            
            response = Message(
                content=llm_response.content,
                role=AgentRole.RAG,
                metadata={
                    "document_metadata": context['pdf_metadata'],
                    "context": context['text'],
                    "query_type": query.metadata.get("type", "general"),
                    "response_type": query.metadata.get("response_type", "general")
                }
            )
            self.conversation_history.append(response)
            return response

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return Message(
                content=f"Error: {str(e)}",
                role=AgentRole.SYSTEM,
                metadata={"error": True}
            )

    def _create_prompt(self, query: Message, context: Dict[str, Any]) -> str:
        """Create focused prompt for LLM"""
        user_prompt = query.metadata.get("rag_prompt", "")
        metadata = context['pdf_metadata']
        
        # Format metadata nicely
        metadata_str = "\n".join([
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in metadata.items()
            if value and value != "Unknown"
        ])
        
        return f"""{self.system_prompt}

        {user_prompt}

        Document Information:
        {metadata_str}

        Context:
        {context['text']}

        Please provide a focused, friendly response that directly answers the question.
        Keep it conversational but concise.
        """

class AgentSystem:
    """Manages communication between User and RAG agents"""
    
    def __init__(self, groq_api_key: str):
        self.reset(groq_api_key)

    def reset(self, groq_api_key: str):
        """Reset the agent system to initial state"""
        self.user_agent = UserAgent()
        self.rag_agent = RAGAgent(groq_api_key)

    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user query through the agent system"""
        try:
            # User agent formats the query
            formatted_query = self.user_agent.format_query(query)
            
            # RAG agent generates response
            rag_response = self.rag_agent.generate_response(formatted_query, context)
            
            # User agent processes response for display
            final_response = self.user_agent.process_response(rag_response)
            
            # Include agent outputs
            final_response.update({
                "user_analysis": formatted_query.metadata.get("query_analysis", {}),
                "rag_prompt": formatted_query.metadata.get("rag_prompt", "")
            })
            
            return final_response

        except Exception as e:
            logging.error(f"Error in agent system: {str(e)}")
            return {
                "error": f"Error processing query: {str(e)}",
                "metadata": context['pdf_metadata'],
                "success": False
            }

    def format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response for the frontend"""
        if "error" in response:
            return {
                "answer": f"Error: {response['error']}",
                "metadata": response.get("metadata", {}),
                "success": False
            }

        return {
            "answer": response["answer"],
            "metadata": response["metadata"],
            "context": response.get("context", ""),
            "success": True
        } 