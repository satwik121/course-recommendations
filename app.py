import streamlit as st
import json
import openai
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
client = OpenAI(api_key=openai.api_key)

# --------------------------
# Config
# --------------------------
FEEDBACK_FILE = "feedback.json"

# --------------------------
# Load API Key
# --------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
if openai.api_key is None:
    openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")

client = OpenAI(api_key=openai.api_key)

# Store top courses in session (from recommendation step)
if "top_courses" not in st.session_state:
    st.session_state.top_courses = []

# --------------------------
# Load Course Dataset
# --------------------------
@st.cache_data
def load_courses():
    with open("course.json", "r", encoding="utf-8") as f:
        return json.load(f)

courses = load_courses()

# --------------------------
# Get Embeddings
# --------------------------
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# --------------------------
# Helper Functions
# --------------------------
def filter_and_embed_courses(
    courses,
    selected_providers,
    selected_skill_levels,
    selected_cost_type,
    selected_duration,
    user_embedding
):
    """
    Filters courses based on sidebar selections and computes similarity with user profile.
    Returns sorted list of (course, similarity_score).
    """
    course_similarities = []

    for course in courses:
        if (
            course["provider"] in selected_providers
            and course["skill_level"] in selected_skill_levels
            and str(course["cost"]).lower() in selected_cost_type
            and isinstance(course["duration_hours"], (int, float))
            and course["duration_hours"] <= selected_duration
        ):
            tags_str = ", ".join(course.get("tags", []))
            course_text = f"{course['title']} {course['description']} {tags_str} {course['skill_level']} {course.get('sentiment_score', 0)}"
            emb = get_embedding(course_text)
            sim = cosine_similarity([user_embedding], [emb])[0][0]
            course_similarities.append((course, sim))

    return sorted(course_similarities, key=lambda x: x[1], reverse=True)

def display_recommendations(top_courses, limit=5):
    """Display top recommended courses in Streamlit with semantic reranking."""
    st.subheader("✅ Recommended Courses for You")

    # Take top-N from initial ranking
    if top_courses == []:
        st.write("No courses found with the selected filters.")
        return
    selected_courses = top_courses[:limit]
    print(selected_courses)
    print("-------")

    # Rerank them based on semantic_score (descending order)
    reranked_courses = sorted(
        selected_courses,
        key=lambda x: float(x[0].get("sentiment_score", 0)),
        reverse=True
    )
    print(reranked_courses)
    print("-------")

    # Display courses
    for idx, (course, score) in enumerate(reranked_courses, 1):
        st.markdown(f"### {idx}. {course['title']} ({course['provider']})")
        st.write(f"**Description:** {course['description']}")
        st.write(f"**Skill Level:** {course['skill_level']}")
        st.write(f"**Tags:** {', '.join(course.get('tags', []))}")
        st.write(f"**Duration:** {course['duration_hours']} hours | **Cost:** {course['cost']}")
        st.write(f"**Match Score:** {score:.3f} | **Semantic Score:** {course.get('sentiment_score', 0):.3f}")
        st.markdown("---")

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎓 Intelligent Course Recommender")
st.write("Provide your background, interests, and career goals, and get personalized course recommendations.")

# --- User Inputs ---
user_background = st.text_area("Background (e.g., Final-year CS student)")
user_interests = st.text_area("Interests (e.g., AI, Data Science)")
user_goals = st.text_area("Career Goals (e.g., Become an ML Engineer)")
user_skills = st.text_area("Optional: Skill levels (e.g., Python - intermediate, Math - beginner)")

# --------------------------
# Sidebar Filters
# --------------------------
st.sidebar.header("📌 Filters")

providers = sorted(set([c["provider"] for c in courses]))
skill_levels = sorted(set([c["skill_level"] for c in courses]))

selected_providers = st.sidebar.multiselect("Platform", providers, default=providers)
selected_skill_levels = st.sidebar.multiselect("Difficulty", skill_levels, default=skill_levels)

max_duration = int(max([c["duration_hours"] for c in courses if isinstance(c["duration_hours"], (int, float))]))
selected_duration = st.sidebar.slider("Maximum Duration (hours)", 0, max_duration, max_duration)

selected_cost_type = st.sidebar.multiselect("Cost", ["free", "paid"], default=["free", "paid"])

# --------------------------
# Recommendation Logic
# --------------------------

# Store top courses in session (from recommendation step)
if "top_courses" not in st.session_state:
    st.session_state.top_courses = []

if st.button("Get Recommendations"):
    if not openai.api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        # Build user profile text
        profile_text = f"Background: {user_background}\nInterests: {user_interests}\nGoals: {user_goals}\nSkills: {user_skills}"

        # Embed user profile
        user_embedding = get_embedding(profile_text)

        # Filter + Embed + Similarity
        st.session_state.top_courses = filter_and_embed_courses(
            courses,
            selected_providers,
            selected_skill_levels,
            selected_cost_type,
            selected_duration,
            user_embedding
        )

        # if not st.session_state.top_courses:
        #     st.warning("No courses found with the selected filters.")
        # else:
        #     display_recommendations(top_courses, limit=5)

display_recommendations(st.session_state.top_courses, limit=5)

# Input / Output file paths
input_file1 = "feedback.json"

# Load JSON data
with open(input_file1, "r", encoding="utf-8") as f:
    feedback_course = json.load(f)

# --------------------------
# Streamlit Interface
# --------------------------
from multi_Agent import *
st.subheader("💬 Multi-Agent Learning Path Q&A")

# Initialize the multi-agent system
if "multi_agent_system" not in st.session_state:
    st.session_state.multi_agent_system = MultiAgentLearningSystem()

# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Update session with latest recommended courses
if 'top_courses' in locals() and top_courses:
    st.session_state.top_courses = top_courses

# Display agent status
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🧭 Router Agent\n\nRoutes queries to specialists")
with col2:
    st.success("📚 Learning Path Agent\n\nHandles course recommendations")
with col3:
    st.warning("💬 Feedback Agent\n\nHandles reviews & feedback")

# Chat input
if prompt := st.chat_input("Ask about learning paths or course feedback..."):
    
    # Prepare user profile and course data
    if 'user_background' in locals():
        profile_text = f"Background: {user_background}\nInterests: {user_interests}\nGoals: {user_goals}\nSkills: {user_skills}"
    else:
        profile_text = "Profile not available"
    
    if 'st.session_state.top_courses' in locals() and st.session_state.top_courses:
        course_text = "\n".join(
            [f"- {c[0]['title']} ({c[0]['provider']}, {c[0]['skill_level']})"
             for c in st.session_state.top_courses[:5]]
        )
    else:
        course_text = "No courses available"
    
    # Show processing status
    with st.status("Processing query through multi-agent system...") as status:
        st.write("🧭 Router analyzing query...")
        
        # Process through multi-agent system
        response = st.session_state.multi_agent_system.process_query(
            query=prompt,
            user_profile=profile_text,
            top_courses=course_text,
            chat_history=st.session_state.chat_history,
            feedback_course=feedback_course
        )
        
        st.write("✅ Agent completed processing")
        status.update(label="Query processed successfully!", state="complete")
    
    # Save to history
    st.session_state.chat_history.append({"user": prompt, "assistant": response})
    
    # Display conversation
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)

# Show recent chat history (optional)
if st.button("Show Chat History") and st.session_state.chat_history:
    st.subheader("Recent Conversations")
    for i, turn in enumerate(st.session_state.chat_history[-3:], 1):
        with st.expander(f"Conversation {len(st.session_state.chat_history)-3+i}"):
            st.write("**User:**", turn["user"])
            st.write("**Assistant:**", turn["assistant"])

# import streamlit as st
# from langgraph.graph import StateGraph, END
# from langchain_core.runnables import RunnableLambda
# from langchain_openai import ChatOpenAI

# # ------------------------
# # Agents
# # ------------------------

# def learning_path_agent(state):
#     """Handles learning path Q&A using top courses context + history."""
#     user_query = state["query"]
#     profile_text = f"Background: {user_background}\nInterests: {user_interests}\nGoals: {user_goals}\nSkills: {user_skills}"

#     top_courses = st.session_state.get("top_courses", [])
#     chat_history = st.session_state.get("chat_history", [])

#     # Prepare top courses context
#     course_text = "\n".join(
#         [f"- {c[0]['title']} ({c[0]['provider']}, {c[0]['skill_level']})"
#          for c in top_courses[:5]]
#     )

#     system_prompt = f"""
#     You are a learning path advisor. Use past conversation history and the
#     following top courses to answer clearly and practically.
#     --Profile Text---
#     {profile_text}
#     --- Past Conversation ---
#     {" ".join([f"User: {h['user']} | Assistant: {h['assistant']}" for h in chat_history])}

#     --- Top Recommended Courses ---
#     {course_text}
#     """

#     llm = ChatOpenAI(model="gpt-4o-mini")
#     resp = llm.invoke([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_query}
#     ])
#     return {"answer": resp.content}


# def feedback_agent(state):
#     """Stub feedback agent."""
#     return {"answer": "Feedback Agent stub (to be implemented)"}


# def platform_agent(state):
#     """Stub platform agent."""
#     return {"answer": "Platform Agent stub (to be implemented)"}


# # ------------------------
# # Router Agent
# # ------------------------
# def router_agent(state):
#     """Use LLM to decide which agent should handle the query."""
#     query = state["query"]

#     system_prompt = """
#     You are a router that decides which agent should handle a query. 
#     Possible agents:
#     - learning_path_agent → questions about courses, skills, or career learning paths
#     - feedback_agent → queries about liking/disliking courses or giving feedback
#     - platform_agent → queries about course platforms, providers, or comparisons
    
#     Output ONLY one of: learning_path_agent, feedback_agent, platform_agent
#     """

#     resp = llm_router.invoke([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": query}
#     ])

#     decision = resp.content.strip().lower()

#     # Normalize outputs
#     if "feedback" in decision:
#         return "feedback_agent"
#     elif "platform" in decision:
#         return "platform_agent"
#     else:
#         return "learning_path_agent"


# # ------------------------
# # Graph Setup
# # ------------------------
# def build_graph():
#     workflow = StateGraph(dict)

#     # Agents as runnable nodes
#     workflow.add_node("learning_path_agent", RunnableLambda(learning_path_agent))
#     workflow.add_node("feedback_agent", RunnableLambda(feedback_agent))
#     workflow.add_node("platform_agent", RunnableLambda(platform_agent))

#     # Router as branch, not a runnable
#     workflow.set_entry_point("router")

#     workflow.add_conditional_edges(
#         "router",  # virtual router node
#         router_agent,  # just function, not RunnableLambda
#         {
#             "learning_path_agent": "learning_path_agent",
#             "feedback_agent": "feedback_agent",
#             "platform_agent": "platform_agent"
#         }
#     )

#     # Endpoints
#     workflow.add_edge("learning_path_agent", END)
#     workflow.add_edge("feedback_agent", END)
#     workflow.add_edge("platform_agent", END)

#     return workflow.compile()


# # ------------------------
# # Streamlit UI
# # ------------------------
# st.title("🎓 Multi-Agent Course Advisor")

# if "graph" not in st.session_state:
#     st.session_state.graph = build_graph()

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_query = st.chat_input("Ask about learning paths, feedback, or platform...")

# if user_query:
#     result = st.session_state.graph.invoke({"query": user_query})
#     answer = result["answer"]

#     # Save conversation (but don't display history to user)
#     st.session_state.chat_history.append({
#         "user": user_query,
#         "assistant": answer
#     })

#     # Display only latest response
#     st.chat_message("assistant").write(answer)

