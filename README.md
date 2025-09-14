🎓 Intelligent Course Recommender with Multi-Agent Q&A

This project is a Streamlit-based web application that provides personalized course recommendations and a multi-agent learning assistant for interactive Q&A. It combines OpenAI embeddings, semantic similarity search, and a multi-agent system to guide learners in selecting the most suitable courses and exploring structured learning paths.

🚀 Features

Course Recommendations

Uploads and filters courses from a JSON dataset (course.json)

Embeds user profile and course descriptions using OpenAI embeddings

Ranks courses by cosine similarity between user preferences and course embeddings

Semantic reranking using sentiment_score

Interactive UI for filtering by platform, difficulty, duration, and cost

Multi-Agent Q&A

Router Agent: Decides which agent handles the query

Learning Path Agent: Suggests learning paths using top recommended courses

Feedback Agent: Processes user feedback on courses

Platform Agent: Provides information about providers/platforms

Maintains chat history for contextual responses

Streamlit Interface

Sidebar filters for courses

Real-time recommendations

Chat interface with multi-agent responses

Chat history viewer

📂 Project Structure
.
├── course.json          # Dataset of available courses
├── feedback.json        # JSON file storing course feedback
├── multi_Agent.py       # Multi-agent system logic
├── app.py               # Main Streamlit app (this file)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation

⚙️ Setup & Installation

Clone Repository

git clone https://github.com/your-repo/course-recommender.git
cd course-recommender


Create Virtual Environment

python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows


Install Dependencies

pip install -r requirements.txt


Example requirements.txt:

streamlit
openai
scikit-learn
numpy


Prepare Course Data

Add your course.json dataset (course metadata with title, description, provider, duration, etc.)

Ensure feedback.json exists (can be empty list [] initially).

Set OpenAI API Key

Either set in .streamlit/secrets.toml:

OPENAI_API_KEY = "your_api_key_here"


Or enter interactively in the app.

▶️ Run the Application
streamlit run app.py


Then open the provided local URL (default: http://localhost:8501) in your browser.

🧠 How It Works

Course Embeddings

Uses text-embedding-3-small model to encode course metadata

Matches against user profile embedding (background, interests, goals, skills)

Computes cosine similarity to rank courses

Semantic Reranking

Courses are reranked by a sentiment_score field (if available) for better quality ordering

Multi-Agent Chat

Queries are routed to the appropriate agent:

learning_path_agent: Q&A about skills, goals, courses

feedback_agent: Collects course reviews

platform_agent: Info on providers and platform differences

History & Context

Conversation history is stored in st.session_state.chat_history

Responses consider past turns for contextual recommendations

📝 Example Usage

Input:

Background: Final-year CS student
Interests: AI, Data Science
Goals: Become an ML Engineer
Skills: Python - intermediate, Math - beginner


Filters:

Platform: Coursera, Udemy

Difficulty: Beginner, Intermediate

Duration: ≤ 80 hours

Cost: Free + Paid

Output:

Top 5 recommended courses with similarity scores

Interactive chat for learning paths:

User: "Which course helps me with deep learning basics?"

Assistant: Suggests top deep learning courses from recommendations

📌 Notes

 OpenAI API key I have added

Works best with a well-curated course.json.

feedback.json will store user feedback dynamically.