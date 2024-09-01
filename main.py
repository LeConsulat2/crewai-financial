import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import os
from crewai import Crew, Agent, Task
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

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
        extracted_text = "\n".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )
    return extracted_text


# Functions to extract each section using case-insensitive search
def extract_student_details(text):
    text_lower = text.lower()
    start = text_lower.find("student details")
    end = text_lower.find("māori advisor required?")
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


def extract_support_types(text):
    text_lower = text.lower()
    start = text_lower.find(
        "please select the type(s) of financial support you are requesting"
    )
    end = text_lower.find("please tell us briefly about your situation")
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


def extract_support_reason(text):
    text_lower = text.lower()
    start = text_lower.find("please tell us briefly about your situation")
    end = text_lower.find("what is your income?")
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


def extract_income_details(text):
    text_lower = text.lower()
    start = text_lower.find("what is your income?")
    end = text_lower.find("what are your regular essential living costs?")
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


def extract_living_costs(text):
    text_lower = text.lower()
    start = text_lower.find("what are your regular essential living costs?")
    end = text_lower.find("what is your current living situation?")
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


def extract_living_situation(text):
    text_lower = text.lower()
    start = text_lower.find("what is your current living situation?")
    end = text_lower.find("do you have any children or other dependants in your care?")
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


def extract_dependants(text):
    text_lower = text.lower()
    start = text_lower.find(
        "do you have any children or other dependants in your care?"
    )
    end = len(text)
    return text[start:end].strip() if start != -1 else "Not found"


# Function to combine all extracted sections
def combine_extracted_sections(text):
    details = {
        "Student Details": extract_student_details(text),
        "Types of Financial Support": extract_support_types(text),
        "Reason for Support": extract_support_reason(text),
        "Income Details": extract_income_details(text),
        "Living Costs": extract_living_costs(text),
        "Living Situation": extract_living_situation(text),
        "Dependants": extract_dependants(text),
    }
    return details


# Function to define agents and tasks based on raw text input
def define_agents_and_tasks(text):
    # Combine extracted sections directly inside this function
    extracted_data = combine_extracted_sections(text)
    chat_model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")

    # Extracted data from the sections
    support_type = extracted_data.get("Types of Financial Support", "Not found")
    situation = extracted_data.get("Reason for Support", "Not found")
    income = extracted_data.get("Income Details", "Not found")
    living_cost = extracted_data.get("Living Costs", "Not found")

    income_agent = Agent(
        role="Income Agent",
        goal="Calculate the total weekly income from the student's financial information.",
        backstory="You are responsible for accurately calculating the total weekly income from the student's financial documents.",
        prompt_template=PromptTemplate(
            template="""
            Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 
            and any monthly income by dividing by 4. Once all converted to weekly, add all the weekly incomes 
            to show the average weekly income.

            Student's financial information:
            {income}
            """,
            input_variables=["income"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(income=task.input_data)
        ),
        verbose=True,
    )

    living_cost_agent = Agent(
        role="Living Cost Agent",
        goal="Calculate the total weekly living costs from the student's financial information.",
        backstory="You are responsible for calculating the student's weekly living costs, ensuring all costs are accounted for, including fortnightly and monthly.",
        prompt_template=PromptTemplate(
            template="""
            Calculate the total weekly living costs for the student. Convert any fortnightly living costs by dividing by 2 
            and any monthly living costs by dividing by 4.

            Student's financial information:
            {living_cost}
            """,
            input_variables=["living_cost"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(living_cost=task.input_data)
        ),
        verbose=True,
    )

    story_agent = Agent(
        role="Story Agent",
        goal="Analyze the student's financial situation based on their story and financial data.",
        backstory="Your role is to analyze the student's overall financial situation by synthesizing their narrative with the calculated income and living costs.",
        prompt_template=PromptTemplate(
            template="""
            Analyze the student's financial situation based on their story and financial data. Identify any overall 
            shortfall or surplus (weekly income - weekly living costs), and highlight factors such as job loss, placements, 
            or other significant financial challenges.

            Type(s) of Financial Support Requested: {support_type}

            Reason for Seeking Financial Support:
            {situation}

            Weekly Income: {income}
            Weekly Living Costs: {living_cost}
            """,
            input_variables=["support_type", "situation", "income", "living_cost"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(
                support_type=support_type,
                situation=situation,
                income=income,
                living_cost=living_cost,
            )
        ),
        verbose=True,
    )

    recommend_agent = Agent(
        role="Recommend Agent",
        goal="Provide a final recommendation on financial assistance, including amount and justification.",
        backstory="Your task is to review the student's financial situation analysis and make a final recommendation on the amount of financial assistance that should be provided.",
        prompt_template=PromptTemplate(
            template="""
            Based on the following financial situation and story, provide a final recommendation on how much financial assistance 
            the student should receive.
            """,
            input_variables=["story_info"],
        ),
        perform_task=lambda task: chat_model.predict(
            task.agent.prompt_template.format(story_info=task.input_data)
        ),
        verbose=True,
    )

    # Define tasks for each agent
    financial_assessment_task = Task(
        description="Assess the financial needs of each student based on their application, documentation, and financial situation.",
        agent=income_agent,
        expected_output="A detailed report of the total weekly income.",
    )

    living_cost_assessment_task = Task(
        description="Assess the student's weekly living_cost based on their provided financial information.",
        agent=living_cost_agent,
        expected_output="A comprehensive breakdown of the student's weekly living_cost.",
    )

    story_analysis_task = Task(
        description="Synthesize the financial story with the calculated income and living_cost to provide a comprehensive analysis.",
        agent=story_agent,
        expected_output="A narrative analysis that includes the summary of the student's financial situation.",
    )

    final_decision_task = Task(
        description="Document the final decision on the financial assistance request, including the rationale.",
        agent=recommend_agent,
        expected_output="A final recommendation report containing the proposed amount of financial assistance.",
    )

    # Return agents and tasks for CrewAI
    return [income_agent, living_cost_agent, story_agent, recommend_agent], [
        financial_assessment_task,
        living_cost_assessment_task,
        story_analysis_task,
        final_decision_task,
    ]


# Upload PDF file
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    # Extract text from the uploaded PDF
    extracted_text = extract_pdf_text(pdf_file)

    # Combine and extract relevant sections
    extracted_sections = combine_extracted_sections(extracted_text)

    # Display extracted sections
    for section, content in extracted_sections.items():
        st.subheader(section)
        st.write(content)

    # Add a stop button for the user to stop execution if needed
    if st.button("Stop Processing"):
        st.warning("Processing stopped by user.")
        st.stop()

    # Define agents and tasks with the extracted data
    agents, tasks = define_agents_and_tasks(
        extracted_text
    )  # Pass raw text if required by the function

    # Create Crew instance with defined tasks and agents
    crew = Crew(tasks=tasks, agents=agents, verbose=2)

    progress_placeholder = st.empty()

    # Display spinner and run CrewAI
    with st.spinner(
        "Analysis in progress... Outcome and recommendation are being made. Please wait."
    ):
        try:
            result = crew.kickoff()

            # Manually simulate progress logging
            for i, task in enumerate(tasks):
                st.write(f"Completed task {i + 1}/{len(tasks)}: {task.description}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Display the final result
    st.subheader("CrewAI Assessment Result")
    st.write(result)
