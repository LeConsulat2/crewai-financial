import streamlit as st
from crewai import Crew, Agent, Task
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import pdfplumber
import re

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

st.title("Financial Hardship Assessment (CrewAI)")


# Function to extract text from PDF using pdfplumber
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        extracted_text = " ".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )
    return extracted_text


# Improved extraction functions using refined regular expressions
def extract_support_type(pdf_text):
    match = re.search(
        r"Please\s*select\s*the\s*type\(s\)\s*of\s*financial\s*support\s*you\s*are\s*requesting\s*([\s\S]*?)\s*Please\s*tell\s*us\s*briefly\s*about\s*your\s*situation\s*and\s*why\s*you\s*are\s*seeking\s*financial\s*support",
        pdf_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return "Unknown"


def extract_situation(pdf_text):
    match = re.search(
        r"Please\s*tell\s*us\s*briefly\s*about\s*your\s*situation\s*and\s*why\s*you\s*are\s*seeking\s*financial\s*support\s*([\s\S]*?)\s*What\s*is\s*your\s*income\?",
        pdf_text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return "Unknown"


# Custom function to handle progress updates
class ProgressHandler:
    def __init__(self, placeholder):
        self.placeholder = placeholder

    def log(self, message):
        self.placeholder.text(message)


# Define the agents and tasks
def define_agents_and_tasks(pdf_text):
    support_type, situation = extract_support_type(pdf_text), extract_situation(
        pdf_text
    )

    chat_model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

    income_agent = Agent(
        role="Income Agent",
        goal="Calculate the total weekly income from the student's financial information.",
        backstory="""
        You are responsible for accurately calculating the total weekly income from the student's financial documents.
        You need to ensure that all sources of income are correctly converted to a weekly basis, taking into account any special circumstances or irregular payments.
        """,
        prompt_template=PromptTemplate(
            template="""
            Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 and any monthly income by dividing by 4 and once all converted to weekly, add all the weekly incomes which will show the average weekly income.

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

    expense_agent = Agent(
        role="Expense Agent",
        goal="Calculate the total weekly expenses from the student's financial information.",
        backstory="""
        You are responsible for calculating the student's weekly expenses, ensuring all costs are accounted for, including fortnightly and monthly. Once converted into weekly, add all weekly expenses that will show the average weekly expenses.
        Your calculations will be crucial for assessing and determining the student's financial need.
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

    story_agent = Agent(
        role="Story Agent",
        goal="Analyze the student's financial situation based on their story and financial data.",
        backstory="""
        Your role is to analyze the student's overall financial situation by synthesizing their narrative with the calculated income and expenses.
        Your analysis should highlight any financial challenges, gaps, or noteworthy circumstances.
        """,
        prompt_template=PromptTemplate(
            template="""
            Analyze the student's financial situation based on their story and financial data. Identify any overall shortfall or surplus (weekly income - weekly expense), and highlight factors such as job loss, placements, or other significant financial challenges.

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

    recommend_agent = Agent(
        role="Recommend Agent",
        goal="Provide a final recommendation on financial assistance, including amount and justification.",
        backstory="""
        Your task is to review the student's financial situation analysis and make a final recommendation on the amount of financial assistance that should be provided.
        """,
        prompt_template=PromptTemplate(
            template="""
            Based on the following financial situation and story, provide a final recommendation on how much financial assistance the student should receive.
            """,
            input_variables=["story_info"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(story_info=task.input_data)
        ),
        verbose=True,
    )

    financial_assessment_task = Task(
        description="Assess the financial needs of each student based on their application, documentation, and financial situation.",
        agent=income_agent,
        expected_output="""
        A detailed report containing:
        - A breakdown of all income sources, including type, amount, and frequency (weekly, fortnightly, monthly).
        - Conversion of all non-weekly income amounts to a weekly basis with clear calculations.
        - Total calculated weekly income amount in a clear and concise format.
        - Identification of any irregularities or special considerations affecting income calculations (e.g., one-off payments or variable income).
        - Additional notes or assumptions made during calculations.
        """,
    )

    expense_assessment_task = Task(
        description="Assess the student's weekly expenses based on their provided financial information.",
        agent=expense_agent,
        expected_output="""
        A comprehensive breakdown of the student's weekly expenses, including:
        - Itemized list of all regular expenses with type, amount, and frequency (weekly, fortnightly, monthly).
        - Conversion of all non-weekly expenses to a weekly basis, including detailed calculation steps.
        - Total calculated weekly expense amount formatted clearly.
        - Identification of any discrepancies or unusual expenses that may require further explanation.
        - Comments on any cost-saving opportunities or potential financial management strategies.
        """,
    )

    story_analysis_task = Task(
        description="Synthesize the financial story with the calculated income and expenses to provide a comprehensive analysis.",
        agent=story_agent,
        expected_output="""
        A narrative analysis that includes:
        - A summary of the student's financial situation, capturing their personal story and context.
        - Identification of any shortfall or surplus (weekly income - weekly expenses) with the weekly calculations done.
        - Detailed insights into specific financial challenges or factors affecting their situation (e.g., recent job loss, upcoming expenses, life events).
        - An assessment of the student's overall financial stability and risk factors.
        """,
    )

    final_decision_task = Task(
        description="Document the final decision on the financial assistance request, including the rationale.",
        agent=recommend_agent,
        expected_output="""
        A final recommendation report containing:
        - A proposed amount of financial assistance to be provided, with a specific dollar value (e.g., "$400 for 4 weeks of food and transport costs").
        - A clear and logical justification for the recommended amount, referencing the student's financial situation and needs.
        - Contingency plans or alternative recommendations if the decision requires managerial review (e.g., "Recommend $800, but refer to manager for review as it exceeds the $500 threshold. Reason for recommending $800... ie student needs rent, petrol, food support for this 6 week period where no income from the part-time job can be gained due to placement internship taking up all work hours. ")
        - A concise summary of key points that influenced the decision, including any identified risks, special circumstances, or needs.
        - Any recommendations for future follow-ups or additional support the student may require ie keyworker but this is not a must, only if seems necessary.
        """,
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
    pdf_text = extract_pdf_text(uploaded_file)
    agents, tasks = define_agents_and_tasks(pdf_text)
    crew = Crew(tasks=tasks, agents=agents, verbose=2)

    progress_placeholder = st.empty()
    progress_handler = ProgressHandler(progress_placeholder)

    # Display spinner and run CrewAI
    with st.spinner(
        "Analysis in progress... Outcome and recommendation are being made. Please wait."
    ):
        result = crew.kickoff(progress_callback=progress_handler.log)

    # Display the final result
    st.subheader("CrewAI Assessment Result")
    st.write(result)
