import os
import json
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
def get_llm():
    """Initialize and return the LLM instance"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback for demo - you can replace with other providers
        raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key
    )

# Prompt templates
ADAPTIVE_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""You are CareerCompass Adaptive Question Generator.

Given the user's current answers (JSON below), propose up to 3 short follow-up questions that are:
- Clarifying and actionable
- Targeted to help suggest a career path
- Different from questions already asked
- Focused on uncovering specific interests, skills, or preferences

Context:
{context}

Output format: Return ONLY a JSON array of question strings, like:
["What specific programming languages interest you most?", "Do you prefer leading teams or working independently?", "What industry problems would you like to solve?"]

Questions:"""
)

ROADMAP_PROMPT = PromptTemplate(
    input_variables=["session_data", "top_careers"],
    template="""You are CareerCompass Career Advisor.

Based on the user's survey responses and top career matches, provide:
1. A personalized summary explaining why these careers fit (2-3 sentences)
2. A detailed learning roadmap for the top career choice

User Session Data:
{session_data}

Top Career Matches:
{top_careers}

Output format: Return a JSON object with this structure:
{{
    "summary": "Personalized explanation of why these careers match the user...",
    "roadmap": {{
        "milestones": ["Month 1-3: Learn Python basics", "Month 4-6: Build first project", "Month 7-12: Get certification"],
        "projects": ["Build a personal portfolio website", "Create a data analysis dashboard", "Develop a machine learning model"],
        "certifications": ["Google Data Analytics Certificate", "AWS Cloud Practitioner", "Python Institute PCAP"],
        "first_job_tasks": ["Data cleaning and preprocessing", "Creating basic reports", "Supporting senior analysts"]
    }}
}}

Response:"""
)

def generate_adaptive_questions(context: str) -> List[str]:
    """
    Generate adaptive follow-up questions based on user's current answers
    
    Args:
        context: JSON string containing user info and current answers
        
    Returns:
        List of follow-up question strings
    """
    try:
        llm = get_llm()
        
        # Format the prompt
        prompt = ADAPTIVE_QUESTIONS_PROMPT.format(context=context)
        
        # Get response from LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse the JSON response
        questions_json = response.content.strip()
        
        # Clean up response if it has extra text
        if questions_json.startswith('```json'):
            questions_json = questions_json.replace('```json', '').replace('```', '').strip()
        
        questions = json.loads(questions_json)
        
        # Ensure it's a list and limit to 3 questions
        if isinstance(questions, list):
            return questions[:3]
        else:
            return []
            
    except Exception as e:
        print(f"Error generating adaptive questions: {str(e)}")
        # Fallback questions if LLM fails
        return [
            "What specific aspects of this field interest you most?",
            "How do you prefer to learn new skills?",
            "What kind of impact do you want to have in your career?"
        ]

def summarize_and_roadmap(session_json: Dict[str, Any], top_careers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate personalized career summary and learning roadmap
    
    Args:
        session_json: Complete user session data
        top_careers: List of top recommended careers
        
    Returns:
        Dictionary with summary and roadmap
    """
    try:
        llm = get_llm()
        
        # Prepare the data for the prompt
        session_data = json.dumps(session_json, indent=2)
        careers_data = json.dumps(top_careers[:3], indent=2)
        
        # Format the prompt
        prompt = ROADMAP_PROMPT.format(
            session_data=session_data,
            top_careers=careers_data
        )
        
        # Get response from LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse the JSON response
        result_json = response.content.strip()
        
        # Clean up response if it has extra text
        if result_json.startswith('```json'):
            result_json = result_json.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(result_json)
        
        return result
        
    except Exception as e:
        print(f"Error generating summary and roadmap: {str(e)}")
        
        # Fallback response if LLM fails
        top_career = top_careers[0] if top_careers else {"title": "Software Developer"}
        
        return {
            "summary": f"Based on your responses, {top_career['title']} appears to be an excellent fit for your interests and skills. Your answers suggest you enjoy problem-solving and working with technology.",
            "roadmap": {
                "milestones": [
                    "Month 1-3: Learn fundamental concepts and basic tools",
                    "Month 4-6: Build your first practical project",
                    "Month 7-9: Gain hands-on experience through internships or volunteer work",
                    "Month 10-12: Prepare for entry-level positions and build your network"
                ],
                "projects": [
                    "Create a personal portfolio website",
                    "Build a project related to your interests",
                    "Contribute to an open-source project"
                ],
                "certifications": [
                    "Industry-relevant certification for your chosen field",
                    "Technical skills certification",
                    "Professional development course"
                ],
                "first_job_tasks": [
                    "Learn company-specific tools and processes",
                    "Work on small, well-defined tasks",
                    "Collaborate with senior team members",
                    "Participate in training and development programs"
                ]
            }
        }

# Additional helper functions for LangChain integration

def validate_api_key() -> bool:
    """Check if Groq API key is properly configured"""
    api_key = os.getenv("GROQ_API_KEY")
    return api_key is not None and len(api_key) > 0

def get_model_info() -> Dict[str, str]:
    """Return information about the current model configuration"""
    return {
        "model": "llama-3.3-70b-versatile",
        "provider": "Groq",
        "temperature": "0.7",
        "api_key_configured": str(validate_api_key())
    }