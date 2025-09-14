import streamlit as st
from typing import Dict, List, Any, Literal
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import json
import openai
from openai import OpenAI
import os 

# Initialize OpenAI API key from secrets
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
if openai.api_key is None:
    openai.api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Initialize OpenAI client for LangChain
client = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai.api_key)

@dataclass
class AgentState:
    """State shared between agents"""
    query: str = ""
    user_profile: str = ""
    top_courses: str = ""
    chat_history: List[Dict] = None
    agent_decision: str = ""
    final_response: str = ""
    feedback_course : str = ""
    
    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []
    
class RouterAgent:
    """Routes queries to appropriate specialized agent"""
    
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        routing_prompt = f"""
        You are a router agent. Based on the user query, decide which agent should handle it:
        
        - learning_path_agent: For questions about courses, learning recommendations, career paths, skills development, curriculum advice
        - feedback_agent: For query related about course feedback like - satwik , c_id - c002 , this course is amazing 
        
        User Query: {state.query}
        
        Respond with ONLY one word: either "learning_path_agent" or "feedback_agent"
        """
        
        response = client.invoke([HumanMessage(content=routing_prompt)])
        decision = response.content.strip().lower()
        
        # Ensure valid decision
        if decision not in ["learning_path_agent", "feedback_agent"]:
            decision = "learning_path_agent"  # Default fallback
        print('Router decision:', decision)
        return {"agent_decision": decision}

class LearningPathAgent:
    """Handles learning path and course recommendation queries"""
    
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        learning_prompt = f"""
        You are a helpful career learning advisor specializing in learning paths and course recommendations.
        
        User Profile:
        {state.user_profile}
        
        Top Recommended Courses:
        {state.top_courses}

        Feedback Course:
        {state.feedback_course}
        
        Previous Chat Context:
        {self._format_chat_history(state.chat_history)}
        
        User Question: {state.query}
        
        Provide helpful advice about learning paths, course recommendations, career guidance, or skill development.
        Be specific and reference the recommended courses when relevant.

        Also share the feedback of the courses if available for the top courses in the feedback_course json.
        """
        
        response = client.invoke([HumanMessage(content=learning_prompt)])
        return {"final_response": response.content}
    
    def _format_chat_history(self, history: List[Dict]) -> str:
        if not history:
            return "No previous conversation"
        
        formatted = []
        for turn in history[-3:]:  # Last 3 turns for context
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(formatted)

class FeedbackAgent:
    """Handles feedback and review-related queries"""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Get raw user feedback text from state
        feedback_text = state.query

        feedback_prompt = f"""
        You are a course feedback agent and review specialist. 

        You will get the feedback from users about courses in natural text like:
        "satwik , c_id - c002 , this course is amazing"

        Convert it into the following JSON structure:

        {{
            "student_name": "<name>",
            "course_id": "<course_id>",
            "like": true/false,
            "dislike": true/false,
            "comment": "<user's feedback text>"
        }}

        Input: {feedback_text}

        Respond in STRICT JSON format only, without extra text.
        """

        # Call LLM
        response = client.invoke([HumanMessage(content=feedback_prompt)])
        print('response')
        print(response.content)

        # Parse JSON from LLM output
        try:
            feedback_data = json.loads(response.content)
        except Exception as e:
            raise ValueError(f"Invalid response: {response.content}") from e

        # Save feedback
        save_feedback(
            student_name=feedback_data["student_name"],
            course_id=feedback_data["course_id"],
            like=feedback_data["like"],
            dislike=feedback_data["dislike"],
            comment=feedback_data["comment"],
        )

        return {"final_response": "Feedback saved successfully."}
    
    def _format_chat_history(self, history: List[Dict]) -> str:
        if not history:
            return "No previous conversation"
        
        formatted = []
        for turn in history[-3:]:  # Last 3 turns for context
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(formatted)

class MultiAgentLearningSystem:
    """Main system orchestrating the multi-agent workflow"""
    
    def __init__(self):
        self.router = RouterAgent()
        self.learning_agent = LearningPathAgent()
        self.feedback_agent = FeedbackAgent()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def route_query(state: AgentState) -> Literal["learning_path_agent", "feedback_agent"]:
            """Route based on router decision"""
            return state.agent_decision
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router)
        workflow.add_node("learning_path_agent", self.learning_agent)
        workflow.add_node("feedback_agent", self.feedback_agent)
        
        # Define the flow
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            route_query,
            {
                "learning_path_agent": "learning_path_agent",
                "feedback_agent": "feedback_agent"
            }
        )
        workflow.add_edge("learning_path_agent", END)
        workflow.add_edge("feedback_agent", END)
        
        return workflow.compile()
    
    def process_query(self, query: str, user_profile: str, top_courses: str, chat_history: List[Dict]) -> str:
        """Process a query through the multi-agent system"""
        
        initial_state = AgentState(
            query=query,
            user_profile=user_profile,
            top_courses=top_courses,
            chat_history=chat_history,
            agent_decision="",
            final_response=""
        )
        
        # Run the workflow
        result = self.graph.invoke(initial_state)
        return result["final_response"]

# --------------------------
# Streamlit Interface
# --------------------------

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
            chat_history=st.session_state.chat_history
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




# ----------------------
# Pydantic model for request
# ----------------------
# class Feedback(BaseModel):
#     student_name: str = "Rahul"
#     course_id: str = "c001"
#     like: bool = True
#     dislike: bool = False
#     comment: str = "Great Course!"


# ----------------------
# Feedback saving function
# ----------------------
def save_feedback(student_name: str, course_id: str, like: bool, dislike: bool, comment: str):
    feedback_file = "feedback.json"
    course_file = "course.json"

    # Load feedback
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    updated = False
    max_id = max((record.get("id", 0) for record in data), default=0)

    sentiment_change = 0

    # Update existing feedback
    for record in data:
        if record.get("student_name") == student_name and record.get("course_id") == course_id:
            if record.get("like"):
                sentiment_change -= 1
            elif record.get("dislike"):
                sentiment_change += 1

            record.update({
                "like": like,
                "dislike": dislike,
                "comment": comment
            })
            updated = True
            break

    # Add new feedback
    if not updated:
        new_record = {
            "id": max_id + 1,
            "student_name": student_name,
            "course_id": course_id,
            "like": like,
            "dislike": dislike,
            "comment": comment
        }
        data.append(new_record)

    if like:
        sentiment_change += 1
    elif dislike:
        sentiment_change -= 1

    # Save feedback.json
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=2)

    # Update sentiment_score in course.json
    if os.path.exists(course_file):
        with open(course_file, "r") as f:
            try:
                courses = json.load(f)
            except json.JSONDecodeError:
                courses = []
    else:
        courses = []

    for course in courses:
        if course["id"] == course_id:
            course["sentiment_score"] = course.get("sentiment_score", 0) + sentiment_change
            break

    with open(course_file, "w") as f:
        json.dump(courses, f, indent=2)

    return {"status": "success", "updated": updated, "total_records": len(data)}