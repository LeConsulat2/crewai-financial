import streamlit as st
from crewai import Crew, Agent, Task
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Configure API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["credentials"].get(
    "OPENAI_API_KEY"
)
if not openai_api_key:
    st.error(
        "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable or add it to the Streamlit secrets."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

st.title("Advanced Financial Assistance Assessment (CrewAI)")


# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = " ".join(page.extract_text() for page in reader.pages)
    return extracted_text


# Function to extract specific sections for StoryAgent
def extract_story_info(pdf_text):
    support_type = extract_support_type(pdf_text)
    situation = extract_situation(pdf_text)
    return support_type, situation


# Example extraction functions
def extract_support_type(pdf_text):
    # Logic to find and return the "type(s) of financial support" section
    start = pdf_text.find(
        "Please select the type(s) of financial support you are requesting"
    )
    end = pdf_text.find(
        "Please tell us briefly about your situation and why you are seeking financial support"
    )
    if start != -1 and end != -1:
        return (
            pdf_text[start:end]
            .replace(
                "Please select the type(s) of financial support you are requesting", ""
            )
            .strip()
        )
    return "Unknown"


def extract_situation(pdf_text):
    # Logic to find and return the "situation and why seeking financial support" section
    start = pdf_text.find(
        "Please tell us briefly about your situation and why you are seeking financial support"
    )
    end = pdf_text.find("What is your income?")
    if start != -1 and end != -1:
        return (
            pdf_text[start:end]
            .replace(
                "Please tell us briefly about your situation and why you are seeking financial support",
                "",
            )
            .strip()
        )
    return "Unknown"


# Define the agents and tasks
def define_agents_and_tasks(pdf_text):
    # Extract specific information for the StoryAgent
    support_type, situation = extract_story_info(pdf_text)

    # Initialize ChatOpenAI model
    chat_model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")

    # Define IncomeAgent
    income_agent = Agent(
        role="Income Agent",
        goal="Calculate the total weekly income from the student's financial information.",
        backstory="""
        You are responsible for accurately calculating the total weekly income from the student's financial documents.
        You need to ensure that all sources of income are correctly converted to a weekly basis, taking into account any special circumstances or irregular payments.
        """,
        prompt_template=PromptTemplate(
            template="""
            Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 and any monthly income by dividing by 4.

            Student's financial information:
            {pdf_text}
            """,
            input_variables=["pdf_text"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(pdf_text=task.input_data)
        ),
        verbose=True,
    )

    # Define ExpenseAgent
    expense_agent = Agent(
        role="Expense Agent",
        goal="Calculate the total weekly expenses from the student's financial information.",
        backstory="""
        You are responsible for calculating the student's weekly expenses, ensuring all costs are accounted for, including those that may occur less frequently (monthly or fortnightly).
        Your calculations will be crucial for determining the student's financial need.
        """,
        prompt_template=PromptTemplate(
            template="""
            Calculate the total weekly expenses for the student. Convert any fortnightly expenses by dividing by 2 and any monthly expenses by dividing by 4.

            Student's financial information:
            {pdf_text}
            """,
            input_variables=["pdf_text"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(pdf_text=task.input_data)
        ),
        verbose=True,
    )

    # Define StoryAgent
    story_agent = Agent(
        role="Story Agent",
        goal="Analyze the student's financial situation based on their story and financial data.",
        backstory="""
        Your role is to analyze the student's overall financial situation by synthesizing their narrative with the calculated income and expenses.
        Your analysis should highlight any financial challenges, gaps, or noteworthy circumstances that might affect the student's ability to support themselves.
        """,
        prompt_template=PromptTemplate(
            template="""
            Analyze the student's financial situation based on their story and financial data. Identify any income shortfall or surplus, and highlight factors such as job loss, placements, or other significant financial challenges.

            Type(s) of Financial Support Requested: {support_type}

            Reason for Seeking Financial Support:
            {situation}

            Weekly Income: {income}
            Weekly Expenses: {expenses}
            """,
            input_variables=["support_type", "situation", "income", "expenses"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(
                support_type=task.input_data["support_type"],
                situation=task.input_data["situation"],
                income=task.input_data["income"],
                expenses=task.input_data["expenses"],
            )
        ),
        verbose=True,
    )

    # Define RecommendAgent
    recommend_agent = Agent(
        role="Recommend Agent",
        goal="Provide a final recommendation on financial assistance, including amount and justification.",
        backstory="""
        Your task is to review the student's financial situation analysis and make a final recommendation on the amount of financial assistance that should be provided.
        Your recommendation must be well-justified, clearly stating the amount and the rationale, ensuring that it aligns with both the student's needs and the assistance guidelines.
        """,
        prompt_template=PromptTemplate(
            template="""
            Based on the following financial situation and story, provide a final recommendation on how much financial assistance the student should receive.
            Provide an exact dollar value and logically assess if the assistance should be granted or declined.

            Financial Situation Analysis:
            {story_info}
            """,
            input_variables=["story_info"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(story_info=task.input_data)
        ),
        verbose=True,
    )

    # Define tasks
    financial_assessment_task = Task(
        description="Assess the financial needs of each student based on their application, documentation, and financial situation. Make a decision to approve, decline, or recommend a specific amount of assistance.",
        agent=income_agent,  # First step in assessment
    )

    expense_assessment_task = Task(
        description="Assess the student's weekly expenses based on their provided financial information.",
        agent=expense_agent,  # Second step in assessment
    )

    story_analysis_task = Task(
        description="Synthesize the financial story with the calculated income and expenses to provide a comprehensive analysis.",
        agent=story_agent,  # Third step in assessment
    )

    final_decision_task = Task(
        description="Document the final decision on the financial assistance request, including the rationale. Prepare the case for review by a manager if the recommended amount exceeds your approval limit.",
        agent=recommend_agent,  # Final step in decision-making
    )

    return [income_agent, expense_agent, story_agent, recommend_agent], [
        financial_assessment_task,
        expense_assessment_task,
        story_analysis_task,
        final_decision_task,
    ]


# Streamlit file upload
uploaded_file = st.file_uploader(
    "Upload the Financial Assistance Form Submission PDF", type="pdf"
)

if uploaded_file:
    # Extract text from uploaded PDF
    pdf_text = extract_pdf_text(uploaded_file)

    # Initialize the agents and tasks with the extracted text
    agents, tasks = define_agents_and_tasks(pdf_text)

    # Create the crew instance
    crew = Crew(
        tasks=tasks,
        agents=agents,
        verbose=2,
    )

    # Display spinner and run CrewAI
    with st.spinner(
        "Analysis in progress... Outcome and recommendation are being made. Please wait."
    ):
        result = crew.kickoff()

    # Display the result after the spinner is done
    st.subheader("CrewAI Assessment Result")
    st.write(result)
