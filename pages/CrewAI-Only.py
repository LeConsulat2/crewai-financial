import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Crew, Agent, Task
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()

# Configure API key
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["openai"]["api_key"]

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
    return " ".join(page.extract_text() for page in reader.pages)


# Initialize ChatOpenAI
chat_model = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview")

# Define output parsers for structured responses
income_schema = ResponseSchema(
    name="weekly_income",
    description="Total weekly income calculated from the student's financial information",
)
income_parser = StructuredOutputParser.from_response_schemas([income_schema])

expense_schema = ResponseSchema(
    name="weekly_expenses",
    description="Total weekly expenses calculated from the student's financial information",
)
expense_parser = StructuredOutputParser.from_response_schemas([expense_schema])

story_schema = ResponseSchema(
    name="financial_situation",
    description="Analysis of the student's financial situation based on their story and financial data",
)
story_parser = StructuredOutputParser.from_response_schemas([story_schema])

recommendation_schema = ResponseSchema(
    name="recommendation",
    description="Final recommendation on financial assistance, including amount and justification",
)
recommendation_parser = StructuredOutputParser.from_response_schemas(
    [recommendation_schema]
)


# Define Agents with improved prompts and output parsing
class IncomeAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = PromptTemplate(
            template="Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 and any monthly income by dividing by 4.\n\n{format_instructions}\n\nStudent's financial information:\n{pdf_text}",
            input_variables=["pdf_text"],
            partial_variables={
                "format_instructions": income_parser.get_format_instructions()
            },
        )

    def perform_task(self, task: Task):
        prompt = self.prompt_template.format(pdf_text=task.input_data)
        response = chat_model.predict(prompt)
        return income_parser.parse(response)


class ExpenseAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = PromptTemplate(
            template="Calculate the total weekly expenses for the student. Convert any fortnightly expenses by dividing by 2 and any monthly expenses by dividing by 4.\n\n{format_instructions}\n\nStudent's financial information:\n{pdf_text}",
            input_variables=["pdf_text"],
            partial_variables={
                "format_instructions": expense_parser.get_format_instructions()
            },
        )

    def perform_task(self, task: Task):
        prompt = self.prompt_template.format(pdf_text=task.input_data)
        response = chat_model.predict(prompt)
        return expense_parser.parse(response)


class StoryAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = PromptTemplate(
            template="Analyze the student's financial situation based on their story and financial data. Consider any shortfall or surplus of income, and mention factors such as placements, job loss, or rent arrears.\n\n{format_instructions}\n\nStudent's story:\n{pdf_text}\n\nWeekly Income: {income}\nWeekly Expenses: {expenses}",
            input_variables=["pdf_text", "income", "expenses"],
            partial_variables={
                "format_instructions": story_parser.get_format_instructions()
            },
        )

    def perform_task(self, task: Task):
        data = task.input_data
        prompt = self.prompt_template.format(
            pdf_text=data["pdf_text"], income=data["income"], expenses=data["expenses"]
        )
        response = chat_model.predict(prompt)
        return story_parser.parse(response)


class RecommendAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = PromptTemplate(
            template="Based on the following financial situation and story, provide a final recommendation on how much financial assistance the student should receive. Provide an exact dollar value and logically assess if the assistance should be granted or declined.\n\n{format_instructions}\n\nFinancial Situation Analysis:\n{story_info}",
            input_variables=["story_info"],
            partial_variables={
                "format_instructions": recommendation_parser.get_format_instructions()
            },
        )

    def perform_task(self, task: Task):
        prompt = self.prompt_template.format(story_info=task.input_data)
        response = chat_model.predict(prompt)
        return recommendation_parser.parse(response)


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

    # Define and perform task for Income Agent
    income_task = Task(
        name="Calculate Income", input_data=pdf_text, agent="Income Agent"
    )
    income_result = crew.perform_task(income_task)

    # Define and perform task for Expense Agent
    expense_task = Task(
        name="Calculate Expenses", input_data=pdf_text, agent="Expense Agent"
    )
    expense_result = crew.perform_task(expense_task)

    # Define and perform task for Story Agent
    story_task = Task(
        name="Compile Story",
        input_data={
            "pdf_text": pdf_text,
            "income": income_result.result["weekly_income"],
            "expenses": expense_result.result["weekly_expenses"],
        },
        agent="Story Agent",
    )
    story_result = crew.perform_task(story_task)

    # Define and perform task for Recommend Agent
    recommend_task = Task(
        name="Recommend Assistance",
        input_data=story_result.result["financial_situation"],
        agent="Recommend Agent",
    )
    recommendation_result = crew.perform_task(recommend_task)

    # Display the assessment results
    st.header("Assessment Report")
    st.subheader("Income Analysis:")
    st.write(f"Weekly Income: {income_result.result['weekly_income']}")
    st.subheader("Expense Analysis:")
    st.write(f"Weekly Expenses: {expense_result.result['weekly_expenses']}")
    st.subheader("Financial Situation Analysis:")
    st.write(story_result.result["financial_situation"])
    st.subheader("Recommendation:")
    st.write(recommendation_result.result["recommendation"])

    # Add a download button for the report
    report = f"""Assessment Report

Income Analysis:
Weekly Income: {income_result.result['weekly_income']}

Expense Analysis:
Weekly Expenses: {expense_result.result['weekly_expenses']}

Financial Situation Analysis:
{story_result.result['financial_situation']}

Recommendation:
{recommendation_result.result['recommendation']}
"""
    st.download_button(
        label="Download Report",
        data=report,
        file_name="financial_assistance_report.txt",
        mime="text/plain",
    )

    # Provide control buttons for the user to proceed or stop
    if st.button("Stop Processing"):
        st.warning("Processing stopped by user.")
        st.stop()
