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


# Initialize ChatOpenAI Model
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
income_agent = IncomeAgent(
    role="Income Calculation Agent",
    goal="Calculate the total weekly income",
    backstory="Responsible for calculating student income based on financial documents.",
    verbose=True,
)

expense_agent = ExpenseAgent(
    role="Expense Calculation Agent",
    goal="Calculate the total weekly expenses",
    backstory="Responsible for calculating student expenses based on financial documents.",
    verbose=True,
)

story_agent = StoryAgent(
    role="Story Compilation Agent",
    goal="Assess the student's financial situation based on the provided documents",
    backstory="Compiles the student's financial story based on their income and expenses.",
    verbose=True,
)

recommend_agent = RecommendAgent(
    role="Recommendation Agent",
    goal="Provide a recommendation for financial assistance",
    backstory="Recommends the amount of financial assistance based on the compiled financial story.",
    verbose=True,
)

# Streamlit file upload
uploaded_file = st.file_uploader(
    "Upload the Financial Assistance Form Submission PDF", type="pdf"
)

if uploaded_file:
    # Extract text from uploaded PDF
    pdf_text = extract_pdf_text(uploaded_file)

    # Define tasks for each agent
    income_task = Task(
        expected_output="Total weekly income",
        description="Calculate total weekly income based on the provided financial document.",
        agent=income_agent,
        input_data=pdf_text,
    )

    expense_task = Task(
        expected_output="Total weekly expenses",
        description="Calculate total weekly expenses based on the provided financial document.",
        agent=expense_agent,
        input_data=pdf_text,
    )

    story_task = Task(
        expected_output="Compiled financial story",
        description="Compile the financial story based on the income and expenses.",
        agent=story_agent,
        input_data={
            "pdf_text": pdf_text,
            "income_info": income_task,
            "expense_info": expense_task,
        },
    )

    recommend_task = Task(
        expected_output="Final recommendation for financial assistance",
        description="Provide a recommendation for financial assistance based on the compiled story.",
        agent=recommend_agent,
        input_data=story_task,
    )

    # Initialize Crew with agents and tasks
    my_crew = Crew(
        agents=[income_agent, expense_agent, story_agent, recommend_agent],
        tasks=[income_task, expense_task, story_task, recommend_task],
    )

    # Execute tasks
    crew_results = my_crew.kickoff()

    # Display the assessment results
    st.header("Assessment Report")
    st.subheader("Income Agent Result:")
    st.write(crew_results["Total weekly income"])
    st.subheader("Expense Agent Result:")
    st.write(crew_results["Total weekly expenses"])
    st.subheader("Story Agent Result:")
    st.write(crew_results["Compiled financial story"])
    st.subheader("Recommend Agent Result:")
    st.write(crew_results["Final recommendation for financial assistance"])
else:
    st.info("Please upload a PDF file to begin the assessment.")
