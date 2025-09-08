import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import random
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings model (cached)
_embeddings_model = None

def load_embeddings_model():
    """Load and cache the embeddings model with Mistral AI integration"""
    global _embeddings_model
    if _embeddings_model is None:
        try:
            from langchain_mistralai import MistralAIEmbeddings
            
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if not mistral_api_key:
                print("⚠️ MISTRAL_API_KEY not found, trying HuggingFace fallback...")
                return _try_huggingface_embeddings()
            
            # Configure Mistral embeddings
            _embeddings_model = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=mistral_api_key
            )
            print("✅ Mistral embeddings initialized successfully")
            
        except ImportError:
            print("⚠️ langchain_mistralai not installed, trying HuggingFace fallback...")
            return _try_huggingface_embeddings()
            
        except Exception as e:
            print(f"⚠️ Error loading Mistral embeddings: {str(e)}, trying HuggingFace fallback...")
            return _try_huggingface_embeddings()
    
    return _embeddings_model

def _try_huggingface_embeddings():
    """Fallback to HuggingFace embeddings if Mistral fails"""
    global _embeddings_model
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        hf_token = os.getenv("HF_TOKEN")
        
        # Configure HuggingFace embeddings through LangChain
        _embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            # Use HF token if available for better rate limits
            **({'huggingfacehub_api_token': hf_token} if hf_token else {})
        )
        print("✅ LangChain HuggingFace embeddings loaded as fallback")
        
    except ImportError:
        print("⚠️ langchain_huggingface not installed, trying HF Inference API...")
        try:
            from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
            
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN required for inference API")
            
            _embeddings_model = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("✅ LangChain HF Inference API embeddings loaded as fallback")
            
        except Exception as e:
            print(f"⚠️ Error loading HF embeddings: {str(e)}, using text-based fallback")
            _embeddings_model = "fallback"
            
    except Exception as e:
        print(f"⚠️ Error loading HF embeddings: {str(e)}, using text-based fallback")
        _embeddings_model = "fallback"
    
    return _embeddings_model

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings using Mistral AI with fallbacks
    
    Args:
        texts: List of texts to embed
        
    Returns:
        numpy array of embeddings
    """
    embeddings_model = load_embeddings_model()
    
    # Try Mistral or HuggingFace embeddings first
    if embeddings_model != "fallback":
        try:
            # Use LangChain's embed_documents method
            embeddings = embeddings_model.embed_documents(texts)
            return np.array(embeddings)
        except Exception as e:
            print(f"Embeddings API error: {str(e)}, using fallback")
    
    # Final fallback to text-based similarity
    print("Using text-based similarity fallback")
    return _get_simple_text_embeddings(texts)

def _get_simple_text_embeddings(texts: List[str]) -> np.ndarray:
    """
    Simple fallback: Create embeddings based on text characteristics
    This provides better similarity than random numbers
    """
    from collections import Counter
    import re
    
    embeddings = []
    
    for text in texts:
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Create a simple feature vector based on:
        # - Word count, char count, common words, etc.
        features = []
        
        # Basic text stats (20 features)
        features.extend([
            len(text),                    # Text length
            len(words),                   # Word count
            len(set(words)),             # Unique words
            sum(len(w) for w in words),  # Total char count
            text.count(' '),             # Space count
        ])
        
        # Common career-related keywords (50 features)
        career_keywords = [
            'python', 'java', 'javascript', 'data', 'analysis', 'machine', 'learning',
            'web', 'development', 'software', 'engineering', 'design', 'management',
            'leadership', 'project', 'team', 'communication', 'problem', 'solving',
            'creative', 'analytical', 'technical', 'business', 'strategy', 'marketing',
            'sales', 'customer', 'service', 'research', 'science', 'technology',
            'computer', 'programming', 'coding', 'database', 'cloud', 'security',
            'network', 'systems', 'applications', 'mobile', 'frontend', 'backend',
            'fullstack', 'devops', 'cybersecurity', 'ai', 'artificial', 'intelligence',
            'statistics', 'visualization', 'reporting'
        ]
        
        keyword_features = []
        for keyword in career_keywords:
            keyword_features.append(1 if keyword in text.lower() else 0)
        features.extend(keyword_features)
        
        # Pad or truncate to 384 dimensions to match MiniLM
        while len(features) < 384:
            features.append(0.0)
        features = features[:384]
        
        embeddings.append(features)
    
    return np.array(embeddings, dtype=np.float32)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {filepath}")
        return {}

def save_json(filepath: str, data: Dict[str, Any]) -> bool:
    """Save data to JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {str(e)}")
        return False

def extract_user_skills(session_data: Dict[str, Any]) -> List[str]:
    """Extract user skills from session answers"""
    skills = []
    skill_keywords = [
        'python', 'javascript', 'java', 'c++', 'sql', 'html', 'css', 'react', 'node.js',
        'machine learning', 'data analysis', 'statistics', 'excel', 'powerbi', 'tableau',
        'project management', 'leadership', 'communication', 'teamwork', 'problem solving',
        'creativity', 'design', 'marketing', 'sales', 'customer service', 'writing',
        'research', 'cybersecurity', 'networking', 'cloud computing', 'aws', 'azure'
    ]
    
    # Get all user answers
    all_answers = ' '.join([
        q.get('answer', '') for q in session_data.get('questions', [])
        if 'answer' in q
    ]).lower()
    
    # Find mentioned skills
    for skill in skill_keywords:
        if skill in all_answers:
            skills.append(skill)
    
    return skills

def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts"""
    try:
        embeddings = get_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error computing similarity: {str(e)}")
        return 0.0

def compute_skill_overlap(user_skills: List[str], career_skills: List[str]) -> float:
    """Compute skill overlap between user and career requirements"""
    if not career_skills:
        return 0.0
    
    user_skills_lower = [skill.lower() for skill in user_skills]
    career_skills_lower = [skill.lower() for skill in career_skills]
    
    overlap = len(set(user_skills_lower) & set(career_skills_lower))
    return overlap / len(career_skills_lower)

def recommend_careers(session_data: Dict[str, Any], career_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate career recommendations based on user session data
    
    Args:
        session_data: User session containing answers
        career_data: List of available careers
        
    Returns:
        List of recommended careers with scores and explanations
    """
    if not career_data:
        return []

    # Extract user information
    user_skills = extract_user_skills(session_data)
    
    # Concatenate all user answers for similarity comparison
    user_text = ' '.join([
        q.get('answer', '') for q in session_data.get('questions', [])
        if 'answer' in q
    ])
    
    if not user_text.strip():
        return career_data[:3]  # Return first 3 if no answers
    
    recommendations = []
    
    for career in career_data:
        # Create career description for comparison
        career_text = f"{career['title']} {' '.join(career.get('key_skills', []))}"
        
        # Compute similarity score using multi-tier fallback system
        similarity_score = compute_text_similarity(user_text, career_text)
        
        # Compute skill overlap
        skill_overlap = compute_skill_overlap(user_skills, career.get('key_skills', []))
        
        # Normalize demand score (0-100 to 0-1)
        demand_score = career.get('demand_score', 50) / 100.0
        
        # Composite score (weighted combination)
        composite_score = (
            0.4 * similarity_score +    # 40% semantic similarity
            0.3 * skill_overlap +       # 30% skill match
            0.3 * demand_score          # 30% market demand
        )
        
        # Add to recommendations with additional info
        career_rec = career.copy()
        career_rec['similarity_score'] = similarity_score
        career_rec['skill_overlap'] = skill_overlap
        career_rec['composite_score'] = composite_score
        career_rec['confidence_score'] = min(composite_score * 100, 95)  # Convert to percentage, cap at 95%
        
        # Generate explanation
        explanation_parts = []
        if skill_overlap > 0.3:
            matching_skills = set([s.lower() for s in user_skills]) & set([s.lower() for s in career.get('key_skills', [])])
            explanation_parts.append(f"You already have experience with {', '.join(list(matching_skills)[:2])}")
        
        if similarity_score > 0.5:
            explanation_parts.append("Your interests align well with this field")
        
        if demand_score > 0.7:
            explanation_parts.append("This field has strong market demand")
        
        career_rec['explanation'] = '. '.join(explanation_parts) if explanation_parts else "This career matches several of your preferences"
        
        recommendations.append(career_rec)
    
    # Sort by composite score
    recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return recommendations

def match_mentors(selected_career: Dict[str, Any], session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Match mentors based on selected career and user profile
    
    Args:
        selected_career: The career the user selected
        session_data: User session data
        
    Returns:
        List of matched mentors ordered by relevance
    """
    try:
        mentors_data = load_json("data/mentors.json")
        mentors_list = mentors_data.get("mentors", [])
    except Exception:
        # Return empty list if mentors file doesn't exist
        return []
    
    if not mentors_list:
        return []
    
    career_id = selected_career.get('career_id', '')
    user_skills = extract_user_skills(session_data)
    
    mentor_scores = []
    
    for mentor in mentors_list:
        score = 0.0
        
        # Check expertise overlap with selected career
        mentor_expertise = mentor.get('expertise', [])
        if career_id in mentor_expertise:
            score += 0.5  # Direct match
        
        # Check skill overlap
        mentor_skills = []
        for exp in mentor_expertise:
            # Find career data for this expertise
            career_match = next((c for c in load_json("data/mock_career_data.json") 
                               if c.get('career_id') == exp), None)
            if career_match:
                mentor_skills.extend(career_match.get('key_skills', []))
        
        skill_overlap = compute_skill_overlap(user_skills, mentor_skills)
        score += 0.3 * skill_overlap
        
        # Add rating factor
        rating = mentor.get('rating', 3.0)
        score += 0.2 * (rating / 5.0)  # Normalize rating to 0-1
        
        mentor_scores.append((mentor, score))
    
    # Sort by score and add some randomization for diversity
    mentor_scores.sort(key=lambda x: x[1] + random.random() * 0.1, reverse=True)
    
    # Return top mentors
    return [mentor for mentor, score in mentor_scores]

# Additional utility functions

def calculate_career_growth_projection(career: Dict[str, Any], years: int = 5) -> List[float]:
    """Calculate career growth projection over specified years"""
    growth_rate = career.get('growth_trend', {}).get('5y_growth_pct', 10) / 100
    base_demand = career.get('demand_score', 50)
    
    projections = []
    for year in range(years + 1):
        projected_demand = base_demand * (1 + growth_rate * (year / 5))
        projections.append(min(projected_demand, 100))  # Cap at 100
    
    return projections

def generate_skill_gap_report(user_skills: List[str], career_skills: List[str]) -> Dict[str, Any]:
    """Generate a skill gap analysis report"""
    user_skills_lower = [skill.lower() for skill in user_skills]
    career_skills_lower = [skill.lower() for skill in career_skills]
    
    has_skills = []
    missing_skills = []
    
    for skill in career_skills:
        if skill.lower() in user_skills_lower:
            has_skills.append(skill)
        else:
            missing_skills.append(skill)
    
    return {
        "has_skills": has_skills,
        "missing_skills": missing_skills,
        "skill_coverage": len(has_skills) / len(career_skills) if career_skills else 0
    }

# Import streamlit for caching - this should be at the top but adding here for clarity
# Note: st.cache_resource was removed and replaced with simple global caching