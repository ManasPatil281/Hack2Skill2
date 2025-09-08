import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import uuid
from pathlib import Path

# Import our custom modules
from langchain_agent import generate_adaptive_questions, summarize_and_roadmap
from utils import load_json, save_json, recommend_careers, match_mentors, extract_user_skills

# Page config
st.set_page_config(
    page_title="CareerCompass - AI Career Guidance",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_session' not in st.session_state:
    st.session_state.current_session = {
        "session_id": str(uuid.uuid4()),
        "user": {"name": "", "email": "", "grade": "", "location": ""},
        "questions": [],
        "metadata": {"langchain_prompt_id": "career_compass_v1", "model": "llama-3.3-70b-versatile", "time": datetime.now().isoformat()}
    }

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'selected_career' not in st.session_state:
    st.session_state.selected_career = None

# Static baseline questions
STATIC_QUESTIONS = [
    {"id": "q1", "text": "What subjects do you enjoy most in school?", "type": "static"},
    {"id": "q2", "text": "What activities make you lose track of time?", "type": "static"},
    {"id": "q3", "text": "Do you prefer working with people, data, or things?", "type": "static"},
    {"id": "q4", "text": "What type of work environment appeals to you? (Remote, office, hybrid, field work)", "type": "static"},
    {"id": "q5", "text": "How important is work-life balance to you?", "type": "static"},
    {"id": "q6", "text": "What skills do you currently have or are developing?", "type": "static"},
    {"id": "q7", "text": "What motivates you most? (Money, impact, creativity, stability, growth)", "type": "static"},
    {"id": "q8", "text": "Do you prefer structured tasks or creative problem-solving?", "type": "static"},
    {"id": "q9", "text": "How comfortable are you with technology and learning new tools?", "type": "static"},
    {"id": "q10", "text": "What does career success look like to you?", "type": "static"}
]

def load_demo_session():
    """Load a prefilled demo session for quick testing"""
    demo_session = {
        "session_id": "demo_" + str(uuid.uuid4()),
        "user": {"name": "Alex Demo", "email": "alex@demo.com", "grade": "12th", "location": "San Francisco"},
        "questions": [
            {"id": "q1", "text": "What subjects do you enjoy most in school?", "type": "static", "answer": "Math, Computer Science, and Physics", "timestamp": datetime.now().isoformat()},
            {"id": "q2", "text": "What activities make you lose track of time?", "type": "static", "answer": "Coding personal projects and solving algorithm problems", "timestamp": datetime.now().isoformat()},
            {"id": "q3", "text": "Do you prefer working with people, data, or things?", "type": "static", "answer": "I love working with data and building things, but also enjoy collaborating with others", "timestamp": datetime.now().isoformat()},
            {"id": "q4", "text": "What type of work environment appeals to you?", "type": "static", "answer": "Hybrid work with some remote flexibility", "timestamp": datetime.now().isoformat()},
            {"id": "q5", "text": "How important is work-life balance to you?", "type": "static", "answer": "Very important - I want to have time for hobbies and personal projects", "timestamp": datetime.now().isoformat()},
        ],
        "metadata": {"langchain_prompt_id": "career_compass_v1", "model": "llama-3.3-70b-versatile", "time": datetime.now().isoformat()}
    }
    return demo_session

def main():
    st.title("🧭 CareerCompass - AI-Powered Career Guidance")
    st.markdown("Discover your ideal career path with personalized AI recommendations")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Demo button in sidebar
    if st.sidebar.button("🚀 Load Demo Session", type="primary"):
        st.session_state.current_session = load_demo_session()
        st.sidebar.success("Demo session loaded! Go to Survey tab to see it.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Survey", "🎯 Recommendations", "👥 Mentors", "⚙️ Admin"])
    
    with tab1:
        survey_tab()
    
    with tab2:
        recommendation_tab()
    
    with tab3:
        mentors_tab()
    
    with tab4:
        admin_tab()

def survey_tab():
    st.header("Career Assessment Survey")
    
    # User info section
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name", value=st.session_state.current_session["user"]["name"])
        grade = st.text_input("Grade/Education Level", value=st.session_state.current_session["user"]["grade"])
    
    with col2:
        email = st.text_input("Email", value=st.session_state.current_session["user"]["email"])
        location = st.text_input("Location", value=st.session_state.current_session["user"]["location"])
    
    # Update session with user info
    st.session_state.current_session["user"] = {
        "name": name, "email": email, "grade": grade, "location": location
    }
    
    st.divider()
    
    # Questions section
    st.subheader("Career Assessment Questions")
    
    # Get existing answers
    existing_answers = {q["id"]: q["answer"] for q in st.session_state.current_session["questions"] if "answer" in q}
    
    # Display static questions
    st.markdown("### Core Questions")
    static_answers = {}
    
    for i, question in enumerate(STATIC_QUESTIONS):
        answer = st.text_area(
            f"**{question['text']}**",
            value=existing_answers.get(question["id"], ""),
            key=f"static_{question['id']}",
            height=80
        )
        static_answers[question["id"]] = answer
    
    # Update session with static answers
    for q_id, answer in static_answers.items():
        if answer.strip():
            # Update existing question or add new
            question_exists = False
            for q in st.session_state.current_session["questions"]:
                if q["id"] == q_id:
                    q["answer"] = answer
                    q["timestamp"] = datetime.now().isoformat()
                    question_exists = True
                    break
            
            if not question_exists:
                question_data = next(q for q in STATIC_QUESTIONS if q["id"] == q_id)
                st.session_state.current_session["questions"].append({
                    "id": q_id,
                    "text": question_data["text"],
                    "type": "static",
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
    
    # Generate adaptive questions
    if st.button("Generate Follow-up Questions", type="primary"):
        if any(static_answers.values()):
            with st.spinner("Generating personalized follow-up questions..."):
                try:
                    context = json.dumps({
                        "user_info": st.session_state.current_session["user"],
                        "answers": static_answers
                    })
                    
                    adaptive_questions = generate_adaptive_questions(context)
                    
                    if adaptive_questions:
                        st.success(f"Generated {len(adaptive_questions)} follow-up questions!")
                        
                        # Add adaptive questions to session
                        for i, q_text in enumerate(adaptive_questions):
                            adaptive_id = f"adaptive_{len([q for q in st.session_state.current_session['questions'] if q['type'] == 'adaptive']) + i + 1}"
                            st.session_state.current_session["questions"].append({
                                "id": adaptive_id,
                                "text": q_text,
                                "type": "adaptive",
                                "timestamp": datetime.now().isoformat()
                            })
                except Exception as e:
                    st.error(f"Error generating adaptive questions: {str(e)}")
        else:
            st.warning("Please answer some questions first before generating follow-ups.")
    
    # Display adaptive questions
    adaptive_questions = [q for q in st.session_state.current_session["questions"] if q["type"] == "adaptive"]
    
    if adaptive_questions:
        st.markdown("### Follow-up Questions")
        for question in adaptive_questions:
            existing_answer = question.get("answer", "")
            answer = st.text_area(
                f"**{question['text']}**",
                value=existing_answer,
                key=f"adaptive_{question['id']}",
                height=80
            )
            
            if answer.strip() and answer != existing_answer:
                question["answer"] = answer
                question["timestamp"] = datetime.now().isoformat()
    
    # Save session
    if st.button("Save Progress"):
        save_session()
        st.success("Progress saved successfully!")

def recommendation_tab():
    st.header("Career Recommendations")
    
    if not st.session_state.current_session["questions"]:
        st.warning("Please complete the survey first to get recommendations.")
        return
    
    # Check if we have answers
    answered_questions = [q for q in st.session_state.current_session["questions"] if "answer" in q and q["answer"].strip()]
    
    if len(answered_questions) < 3:
        st.warning("Please answer at least 3 questions to get meaningful recommendations.")
        return
    
    if st.button("Generate Recommendations", type="primary") or st.session_state.recommendations:
        if not st.session_state.recommendations:
            with st.spinner("Analyzing your responses and generating recommendations..."):
                try:
                    # Load career data
                    career_data = load_json("data/mock_career_data.json")
                    
                    # Get recommendations
                    recommendations = recommend_careers(st.session_state.current_session, career_data)
                    
                    # Get AI summary and roadmap
                    ai_summary = summarize_and_roadmap(st.session_state.current_session, recommendations)
                    
                    st.session_state.recommendations = {
                        "careers": recommendations,
                        "ai_summary": ai_summary
                    }
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    return
        
        # Display recommendations
        if st.session_state.recommendations:
            st.subheader("Your Top Career Matches")
            
            # Display AI summary
            if st.session_state.recommendations.get("ai_summary"):
                st.markdown("### Personalized Analysis")
                st.write(st.session_state.recommendations["ai_summary"].get("summary", ""))
            
            # Display top 3 careers
            for i, career in enumerate(st.session_state.recommendations["careers"][:3]):
                with st.expander(f"#{i+1} {career['title']} - {career['confidence_score']:.1f}% match", expanded=(i==0)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Salary Range:** ${career['avg_salary']:,}")
                        st.write(f"**Demand Score:** {career['demand_score']}/100")
                        st.write(f"**Key Skills:** {', '.join(career['key_skills'])}")
                        st.write(f"**Growth Projection:** {career['growth_trend']['5y_growth_pct']}% over 5 years")
                        st.write(career['growth_trend']['explain'])
                        
                        if career.get('explanation'):
                            st.write(f"**Why this matches you:** {career['explanation']}")
                    
                    with col2:
                        if st.button(f"Select {career['title']}", key=f"select_{i}"):
                            st.session_state.selected_career = career
                            st.success(f"Selected {career['title']}! Check out the Mentors tab.")
            
            # Display roadmap for top choice
            if st.session_state.recommendations.get("ai_summary", {}).get("roadmap"):
                st.subheader("Your Learning Roadmap")
                roadmap = st.session_state.recommendations["ai_summary"]["roadmap"]
                
                if isinstance(roadmap, dict):
                    if roadmap.get("milestones"):
                        st.write("**Learning Milestones:**")
                        for milestone in roadmap["milestones"]:
                            st.write(f"• {milestone}")
                    
                    if roadmap.get("projects"):
                        st.write("**Suggested Projects:**")
                        for project in roadmap["projects"]:
                            st.write(f"• {project}")
                    
                    if roadmap.get("certifications"):
                        st.write("**Recommended Certifications:**")
                        for cert in roadmap["certifications"]:
                            st.write(f"• {cert}")
            
            # Growth visualization
            st.subheader("Career Growth Analysis")
            display_growth_charts(st.session_state.recommendations["careers"])

def display_growth_charts(careers):
    """Display growth trend and skill gap analysis"""
    
    # Growth trend chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Line chart for growth trends
    years = list(range(2024, 2030))
    for career in careers[:3]:
        growth_rate = career['growth_trend']['5y_growth_pct'] / 100
        values = [100 * (1 + growth_rate * (year - 2024) / 5) for year in years]
        ax1.plot(years, values, marker='o', label=career['title'])
    
    ax1.set_title("5-Year Career Growth Projection")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Relative Demand Index")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Skill gap analysis
    if st.session_state.selected_career:
        user_skills = extract_user_skills(st.session_state.current_session)
        career_skills = st.session_state.selected_career['key_skills']
        
        skill_match = []
        for skill in career_skills:
            # Simple keyword matching
            has_skill = any(skill.lower() in answer.lower() 
                          for q in st.session_state.current_session["questions"] 
                          if "answer" in q
                          for answer in [q["answer"]])
            skill_match.append(1 if has_skill else 0)
        
        y_pos = np.arange(len(career_skills))
        ax2.barh(y_pos, skill_match, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(career_skills)
        ax2.set_xlabel('Skill Level (0=Need to Learn, 1=Have Experience)')
        ax2.set_title(f'Skill Gap Analysis: {st.session_state.selected_career["title"]}')
        ax2.set_xlim(0, 1.2)
    
    plt.tight_layout()
    st.pyplot(fig)

def mentors_tab():
    st.header("Find Your Mentor")
    
    if not st.session_state.selected_career:
        st.warning("Please select a career from the Recommendations tab first.")
        return
    
    st.subheader(f"Mentors for {st.session_state.selected_career['title']}")
    
    try:
        # Load mentors data
        mentors_data = load_json("data/mentors.json")
        
        # Match mentors
        matched_mentors = match_mentors(st.session_state.selected_career, st.session_state.current_session)
        
        for i, mentor in enumerate(matched_mentors[:3]):
            with st.expander(f"⭐ {mentor['name']} - {mentor['rating']}/5.0 rating", expanded=(i==0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Expertise:** {', '.join(mentor['expertise'])}")
                    st.write(f"**Bio:** {mentor['bio']}")
                    st.write(f"**Availability:** {mentor['availability']}")
                    
                    # Message template
                    st.write("**Suggested Message:**")
                    message_template = f"""Hi {mentor['name']},

I'm a student exploring a career in {st.session_state.selected_career['title']} and would love to learn from your experience. 

I'm particularly interested in:
- Getting started in this field
- Key skills to develop
- Day-to-day responsibilities

Would you have 15-20 minutes for a brief chat?

Best regards,
{st.session_state.current_session['user']['name']}"""
                    
                    st.text_area("Message:", value=message_template, height=150, key=f"message_{i}")
                
                with col2:
                    st.write(f"**Contact:** {mentor.get('contact', 'mentor@example.com')}")
                    if st.button(f"Connect with {mentor['name']}", key=f"connect_{i}"):
                        st.success("Connection request sent! (Demo)")
        
        # Chat placeholder
        st.subheader("💬 Chat Preview")
        st.info("This is a placeholder for future chat functionality. In the full version, you could chat directly with mentors here.")
        
        with st.container():
            st.text_input("Type a message...", placeholder="Ask your mentor a question", disabled=True)
            if st.button("Send", disabled=True):
                pass
                
    except Exception as e:
        st.error(f"Error loading mentor data: {str(e)}")

def admin_tab():
    st.header("Admin Dashboard")
    
    st.subheader("Session Management")
    
    # Create sessions directory if it doesn't exist
    sessions_dir = Path("data/session_logs")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # List all session files
    session_files = list(sessions_dir.glob("*.json"))
    
    if session_files:
        st.write(f"Total sessions: {len(session_files)}")
        
        # Session list
        for session_file in sorted(session_files, reverse=True):
            with st.expander(f"Session: {session_file.name}"):
                try:
                    session_data = load_json(str(session_file))
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**User:** {session_data.get('user', {}).get('name', 'Anonymous')}")
                        st.write(f"**Questions Answered:** {len([q for q in session_data.get('questions', []) if 'answer' in q])}")
                        st.write(f"**Created:** {session_data.get('metadata', {}).get('time', 'Unknown')}")
                    
                    with col2:
                        if st.button(f"Download", key=f"download_{session_file.name}"):
                            st.download_button(
                                label="💾 Download JSON",
                                data=json.dumps(session_data, indent=2),
                                file_name=session_file.name,
                                mime="application/json",
                                key=f"dl_{session_file.name}"
                            )
                except Exception as e:
                    st.error(f"Error reading {session_file.name}: {str(e)}")
    else:
        st.info("No sessions found yet. Complete a survey to create the first session.")
    
    # Current session info
    st.subheader("Current Session")
    st.json(st.session_state.current_session)
    
    # Save current session manually
    if st.button("Save Current Session"):
        save_session()
        st.success("Session saved!")

def save_session():
    """Save current session to file"""
    sessions_dir = Path("data/session_logs")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    filepath = sessions_dir / filename
    
    save_json(str(filepath), st.session_state.current_session)

if __name__ == "__main__":
    main()