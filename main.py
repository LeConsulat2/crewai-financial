import streamlit as st
import openai
from PyPDF2 import PdfReader
from crewai import Crew, Agent, Task
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Try to get the API key from environment variables or Streamlit secrets
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["credentials"].get(
    "OPENAI_API_KEY"
)

# Raise an error if the API key is not found
if not openai_api_key:
    st.error(
        "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to the Streamlit secrets."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

st.title("Financial Assistance Assessment (Beta)")


# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Initialize CrewAI with agents using ChatOpenAI
crew_ai = Crew(api_key=openai_api_key)
chat_model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")


# Define Agents
class IncomeAgent(Agent):
    def perform_task(self, task: Task):
        pdf_text = task.input_data
        prompt = f"Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 and any monthly income by dividing by 4.\n\n{pdf_text}"
        response = chat_model(messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"].strip()


class ExpenseAgent(Agent):
    def perform_task(self, task: Task):
        pdf_text = task.input_data
        prompt = f"Calculate the total weekly expenses for the student. Convert any fortnightly expenses by dividing by 2 and any monthly expenses by dividing by 4.\n\n{pdf_text}"
        response = chat_model(messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"].strip()


class StoryAgent(Agent):
    def perform_task(self, task: Task):
        data = task.input_data
        pdf_text, income_info, expense_info = (
            data["pdf_text"],
            data["income_info"],
            data["expense_info"],
        )
        prompt = f"Using the following student's story and financial situation, assess their financial situation. Consider any shortfall or surplus of the income, and mention factors such as placements, job loss, or rent arrears.\n\nStory:\n{pdf_text}\n\nWeekly Income:\n{income_info}\n\nWeekly Expenses:\n{expense_info}"
        response = chat_model(messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"].strip()


class RecommendAgent(Agent):
    def perform_task(self, task: Task):
        story_info = task.input_data
        prompt = f"Based on the following financial situation and story, provide a final recommendation on how much financial assistance the student should receive. Provide an exact dollar value and logically assess if the assistance should be granted or declined.\n\n{story_info}"
        response = chat_model(messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"].strip()


# Initialize Crew and add agents
crew = Crew(
    agents=[
        IncomeAgent(name="Income Agent"),
        ExpenseAgent(name="Expense Agent"),
        StoryAgent(name="Story Agent"),
        RecommendAgent(name="Recommend Agent"),
    ]
)

# Streamlit file upload
uploaded_file = st.file_uploader(
    "Upload the Financial Assistance Form Submission PDF", type="pdf"
)

if uploaded_file:
    # Extract text from uploaded PDF
    pdf_text = extract_pdf_text(uploaded_file)

    # Define tasks for each agent
    tasks = [
        Task(name="Calculate Income", input_data=pdf_text, agent="Income Agent"),
        Task(name="Calculate Expenses", input_data=pdf_text, agent="Expense Agent"),
    ]

    # Perform tasks for Income and Expense Agents
    income_task = crew.perform_task(tasks[0])
    expense_task = crew.perform_task(tasks[1])

    income_info = income_task.result
    expense_info = expense_task.result

    # Define task for Story Agent
    story_task = Task(
        name="Compile Story",
        input_data={
            "pdf_text": pdf_text,
            "income_info": income_info,
            "expense_info": expense_info,
        },
        agent="Story Agent",
    )
    story_info = crew.perform_task(story_task).result

    # Define task for Recommend Agent
    recommend_task = Task(
        name="Recommend Assistance", input_data=story_info, agent="Recommend Agent"
    )
    recommendation = crew.perform_task(recommend_task).result

    # Display the assessment results
    st.header("Assessment Report")
    st.subheader("Income Agent Result:")
    st.write(income_info)
    st.subheader("Expense Agent Result:")
    st.write(expense_info)
    st.subheader("Story Agent Result:")
    st.write(story_info)
    st.subheader("Recommend Agent Result:")
    st.write(recommendation)
