import logging
from typing import Dict, Any
from openai import OpenAI
from config.settings import Config

class LLMGenerator:
    """Handles OpenAI LLM interactions"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.logger = logging.getLogger(__name__)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate detailed answer using OpenAI GPT-4"""
        try:
            system_message = """You are an expert assistant for Arizona State University (ASU) with deep knowledge of campus life, academics, and student resources. 

Your responses should be:
- Comprehensive and detailed (aim for 200-400 words)
- Well-structured with clear sections when appropriate
- Specific and actionable when possible
- Include relevant examples, statistics, or specific details from the context
- Professional yet conversational in tone
- Include practical next steps or recommendations when relevant

Always cite specific information from the provided context and acknowledge when information might be limited or outdated."""

            user_prompt = f"""Based on the following context about ASU, provide a detailed and comprehensive answer to the user's question.

Context:
{context}

Question: {query}

Please provide a thorough response that includes:
1. Direct answer to the question
2. Relevant details and examples from the context
3. Practical implications or next steps (when applicable)
4. Any important caveats or limitations

Answer:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}" 