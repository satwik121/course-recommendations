from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os

app = FastAPI(title="Course Feedback API")


# ----------------------
# Pydantic model for request
# ----------------------
class Feedback(BaseModel):
    student_name: str = "Rahul"
    course_id: str = "c001"
    like: bool = True
    dislike: bool = False
    comment: str = "Great Course!"


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


# ----------------------
# FastAPI endpoint
# ----------------------
@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    try:
        result = save_feedback(
            feedback.student_name,
            feedback.course_id,
            feedback.like,
            feedback.dislike,
            feedback.comment
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
