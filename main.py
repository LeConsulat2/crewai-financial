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


# Use caching to avoid redundant processing for similar inputs
@lru_cache(maxsize=10)
def get_cached_analysis(prompt):
    response = openai.Completion.create(
        model="gpt-4o-mini", prompt=prompt, max_tokens=500, temperature=0.5
    )
    return response.choices[0].text.strip()


# Define agents with the backstory directly containing all instructions
income_agent = Agent(
    role="Income Agent",
    goal="Calculate the total weekly income from the student's financial information.",
    backstory="""
    ## Income Calculation
    You are responsible for accurately calculating the total weekly income from the student's financial documents.
    - Convert any fortnightly income by dividing by 2.
    - Convert any monthly income by dividing by 4.
    - Add all the weekly incomes to show the average weekly income.

    **Student's financial information:**
    {income}
    """,
    perform_task=lambda task: get_cached_analysis(
        task.agent.backstory.format(income=task.input_data)
    ),
    verbose=True,
    allow_delegation=False,
)

living_cost_agent = Agent(
    role="Living Cost Agent",
    goal="Calculate the total weekly living costs from the student's financial information.",
    backstory="""
    ## Living Cost Calculation
    Calculate the total weekly living costs for the student.
    - Convert any fortnightly living costs by dividing by 2.
    - Convert any monthly living costs by dividing by 4.
    - Add all the weekly living costs to show the average weekly living costs.

    **Student's financial information:**
    {living_cost}
    """,
    perform_task=lambda task: get_cached_analysis(
        task.agent.backstory.format(living_cost=task.input_data)
    ),
    verbose=True,
    allow_delegation=False,
)

story_agent = Agent(
    role="Story Agent",
    goal="Analyze the student's financial situation based on their story and financial data.",
    backstory="""
    ## Financial Situation Analysis
    Analyze the student's financial situation based on their story and financial data. Identify any overall shortfall or surplus (weekly income - weekly living costs).
    - Highlight factors such as job loss, placements, or other significant financial challenges.

    **Type(s) of Financial Support Requested:** {support_type}
    
    **Reason for Seeking Financial Support:**
    {situation}

    **Weekly Income:** {income}
    **Weekly Living Costs:** {living_cost}
    """,
    perform_task=lambda task: get_cached_analysis(
        task.agent.backstory.format(
            support_type=task.input_data["support_type"],
            situation=task.input_data["situation"],
            income=task.input_data["income"],
            living_cost=task.input_data["living_cost"],
        )
    ),
    verbose=True,
    allow_delegation=False,
)

recommend_agent = Agent(
    role="Recommend Agent",
    goal="Provide a final recommendation on financial assistance, including amount and justification.",
    backstory="""
    ## Final Recommendation
    Based on the following financial situation and story, provide a final recommendation on how much financial assistance the student should receive.
    {story_info}
    """,
    perform_task=lambda task: get_cached_analysis(
        task.agent.backstory.format(story_info=task.input_data)
    ),
    verbose=True,
    allow_delegation=False,
)

# Define tasks
financial_assessment_task = Task(
    description="Assess the financial needs of each student based on their application, documentation, and financial situation.",
    agent=income_agent,
    expected_output="A detailed report of the total weekly income.",
)

living_cost_assessment_task = Task(
    description="Assess the student's weekly living costs based on their provided financial information.",
    agent=living_cost_agent,
    expected_output="A comprehensive breakdown of the student's weekly living costs.",
)

story_analysis_task = Task(
    description="Synthesize the financial story with the calculated income and living costs to provide a comprehensive analysis.",
    agent=story_agent,
    expected_output="A narrative analysis that includes the summary of the student's financial situation.",
)

final_decision_task = Task(
    description="Document the final decision on the financial assistance request, including the rationale.",
    agent=recommend_agent,
    expected_output="A final recommendation report containing the proposed amount of financial assistance.",
)


# Information function to define agents and tasks
def information(extracted_data):
    return [income_agent, living_cost_agent, story_agent, recommend_agent], [
        financial_assessment_task,
        living_cost_assessment_task,
        story_analysis_task,
        final_decision_task,
    ]


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

    # Define agents and tasks with the extracted data
    agents, tasks = information(extracted_sections)

    # Create Crew instance and run tasks
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
