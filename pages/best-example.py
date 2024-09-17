import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import os
from crewai import Crew, Agent, Task
from functools import lru_cache
import openai  # Direct use of OpenAI API

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
            filter(None, [page.extract_text() for page in pdf.pages])
        )
    return extracted_text


# Utility function to extract sections from the text
def extract_section(text, start_keyword, end_keyword):
    start = text.find(start_keyword)
    end = text.find(end_keyword) if end_keyword else len(text)
    return text[start:end].strip() if start != -1 and end != -1 else "Not found"


# Combine extraction functions using the utility function
def combine_extracted_sections(text):
    text_lower = text.lower()
    details = {
        "Student Details": extract_section(
            text_lower, "student details", "māori advisor required?"
        ),
        "Types of Financial Support": extract_section(
            text_lower,
            "please select the type(s) of financial support you are requesting",
            "please tell us briefly about your situation",
        ),
        "Reason for Support": extract_section(
            text_lower,
            "please tell us briefly about your situation",
            "what is your income?",
        ),
        "Income Details": extract_section(
            text_lower,
            "what is your income?",
            "what are your regular essential living costs?",
        ),
        "Living Costs": extract_section(
            text_lower,
            "what are your regular essential living costs?",
            "what is your current living situation?",
        ),
        "Living Situation": extract_section(
            text_lower,
            "what is your current living situation?",
            "do you have any children or other dependants in your care?",
        ),
        "Dependants": extract_section(
            text_lower,
            "do you have any children or other dependants in your care?",
            None,
        ),
    }
    return details


# Define agents with detailed verbose output for advanced insights
income_agent = Agent(
    role="Income Agent",
    goal="Calculate the total weekly income from the student's financial information.",
    backstory="""
    You are an expert at accurately calculating the total weekly income from the combine_extracted_sections function details.
    You convert any fortnightly income by dividing by 2 and monthly income by 4.
    Finally, you then add all the weekly incomes to show the average weekly income.
    """,
    verbose=True,
    allow_delegation=False,
)

living_cost_agent = Agent(
    role="Living Cost Agent",
    goal="Calculate the total weekly living costs (EXPENSES) from the student's financial information.",
    backstory="""
    You are an expert at accurately calculating the total weekly Living Cost (EXPENSES) from the combine_extracted_sections function details.
    Calculate the total weekly living costs (EXPENSES) for the student.
    You convert any fortnightly income by dividing by 2 and monthly living cost (EXPENSES) by 4.
    Finally, you then add all the weekly living cost (EXPENSES) to show the average weekly living cost (EXPENSES).
    """,
    verbose=True,
    allow_delegation=False,
)

story_agent = Agent(
    role="Story Agent",
    goal="Analyze the student's financial situation based on their story and financial data received from income and living_cost Agent.",
    backstory="""
    You are a senior student advisor who expertize in student financial hardship and emergency requests and applications.
    You analyze the student's financial situation based on their story from the combine_extracted_sections function details and financial data received from income and living_cost Agent. Your excellency in the role shows when you not only consider the shortfall or surplus of the (weekly income - weekly living costs) but also to highlight the factors that involve with the student's financial challenge such as job loss, placements (intership), family issues or any other significant financial challenges. Your story compilation is exceptional that the summary of the story is concise but does not miss any points of the story.
    """,
    verbose=True,
    allow_delegation=False,
)

recommend_agent = Agent(
    role="Recommend Agent",
    goal="Provide a final recommendation on financial assistance, including amount and justification.",
    backstory="""
    You are a senior student advisor who compile all the information from the income agent, living_cost agent and story_agent to provide a final recommendation on how much financial assistance the student should receive based on financial situation and their individual stories that involve with the financial challenges. Your recommendation is so sound that any managers or senior managers would agree with your recommendation. 
    """,
    verbose=True,
    allow_delegation=False,
)

# Upload PDF file and process
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    extracted_text = extract_pdf_text(pdf_file)
    extracted_sections = combine_extracted_sections(extracted_text)

    # Display extracted sections
    for section, content in extracted_sections.items():
        st.subheader(section)
        st.write(content)

    if st.button("Stop Processing"):
        st.warning("Processing stopped by user.")
        st.stop()

    # Define tasks with the extracted data
    income_task = Task(
        description="Calculate the total weekly income from the provided financial document.",
        agent=income_agent,
        input_data=extracted_sections["Income Details"],
        expected_output="Total weekly income calculated.",
    )

    living_cost_task = Task(
        description="Calculate the total weekly living costs from the provided financial document.",
        agent=living_cost_agent,
        input_data=extracted_sections["Living Costs"],
        expected_output="Total weekly living costs (EXPENSES) calculated.",
    )

    story_task = Task(
        description="Compile a comprehensive story based on income, expenses, and additional financial data.",
        agent=story_agent,
        input_data={
            "income_info": income_task,
            "living_cost_info": living_cost_task,
            "support_type": extracted_sections["Types of Financial Support"],
            "situation": extracted_sections["Reason for Support"],
        },
        expected_output="Compiled financial story.",
    )

    recommend_task = Task(
        description="Provide a recommendation for financial assistance based on the financial story and MUST give an exact dollar value ie $450 financial hardship fund approved with strong rationale.",
        agent=recommend_agent,
        input_data={
            "story_info": story_task,
        },
        expected_output="Final recommendation for financial assistance. An exact dollar value ie $400 support is given out. If declined, then MUST state declined with the rationale.",
    )

    # Create Crew instance and run tasks
    agents = [income_agent, living_cost_agent, story_agent, recommend_agent]
    tasks = [income_task, living_cost_task, story_task, recommend_task]

    crew = Crew(tasks=tasks, agents=agents, verbose=True)

    with st.spinner(
        "Analysis in progress... Outcome and recommendation are being made. Please wait."
    ):
        try:
            result = crew.kickoff()

            # Log task progress
            for i, task in enumerate(tasks):
                st.write(f"Completed task {i + 1}/{len(tasks)}: {task.description}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Display the final result
    st.subheader("CrewAI Assessment Result")
    st.write(result)

    # Stop execution after displaying the result to avoid unintended reruns
    st.stop()

    if st.button("Stop Processing"):
        st.warning("Processing stopped by user.")
        st.stop()
