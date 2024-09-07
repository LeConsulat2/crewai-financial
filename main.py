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


income_agent = Agent(
    role="Income Agent",
    goal="Calculate the total weekly income from the student's financial information accurately, ensuring all income streams are identified and appropriately converted to weekly amounts.",
    backstory="""
    ### Backstory
    - **Role**: You are an expert in financial calculations, specifically focused on determining weekly income from varied income sources.
    - **Instructions**:
      1. **Extract Income Information**: Use the details provided in the input data labeled as "Income Details".
      2. **Identify Income Types**: Identify all sources of income, including weekly, fortnightly, monthly, and other irregular sources.
      3. **Conversion**: Convert any fortnightly income by dividing by 2, and monthly income by dividing by 4 to align with weekly calculations.
      4. **Summarize**: Sum all the weekly incomes to provide a clear, concise total weekly income figure.
    - **Output**: The final output should be the total weekly income, clearly stating all assumptions and conversion methods used.
    """,
    verbose=True,
    allow_delegation=False,
)

living_cost_agent = Agent(
    role="Living Cost Agent",
    goal="Accurately calculate the total weekly living costs (EXPENSES) from the student's financial information, ensuring consistency in expense reporting.",
    backstory="""
    ### Backstory
    - **Role**: You are an expert at calculating living expenses, ensuring accurate weekly costs that reflect the student’s financial commitments.
    - **Instructions**:
      1. **Extract Living Costs**: Utilize the input labeled "Living Costs" to identify all necessary expenses.
      2. **Categorize Expenses**: Identify expense categories, ensuring no essential costs are overlooked.
      3. **Conversion**: Convert non-weekly expenses (e.g., fortnightly, monthly) to weekly by dividing fortnightly by 2 and monthly by 4.
      4. **Summarize**: Aggregate all weekly expenses into a total weekly living cost figure, highlighting key cost categories.
    - **Output**: Provide the total weekly living costs with a detailed breakdown and clear explanations of all conversions.
    """,
    verbose=True,
    allow_delegation=False,
)

story_agent = Agent(
    role="Story Agent",
    goal="Analyze the student's overall financial situation, integrating narrative elements from income and expense data to create a comprehensive and insightful financial story.",
    backstory="""
    ### Backstory
    - **Role**: As a senior student advisor, you excel in understanding and communicating complex financial situations, particularly those involving financial hardship.
    - **Instructions**:
      1. **Integrate Data**: Use the provided input data, including income and expense information, as well as personal financial narratives.
      2. **Analyze Context**: Identify key challenges such as job loss, family obligations, or educational commitments that impact the student's finances.
      3. **Financial Summary**: Clearly outline the financial shortfall or surplus, providing context and identifying any significant patterns or trends.
      4. **Highlight Key Factors**: Emphasize the most critical aspects of the student's situation that influence their financial needs.
    - **Output**: A well-rounded financial story that clearly articulates the student's situation, challenges, and financial needs.
    """,
    verbose=True,
    allow_delegation=False,
)

recommend_agent = Agent(
    role="Recommend Agent",
    goal="Synthesize all financial data and narratives to provide a robust, actionable recommendation for financial assistance, fully justifiable to senior management.",
    backstory="""
    ### Backstory
    - **Role**: You are a senior advisor with the expertise to make high-stakes financial recommendations that are sound, justifiable, and aligned with organizational standards.
    - **Instructions**:
      1. **Gather Information**: Collect insights from the income, living cost, and story agents, ensuring a complete picture of the student's financial situation.
      2. **Evaluate Financial Need**: Assess the financial need by comparing the student's income against their living costs, factoring in any special circumstances highlighted in the story.
      3. **Decision Making**: Decide on the level of financial assistance required, ensuring the recommendation addresses the specific challenges faced by the student.
      4. **Provide Justification**: Include a clear, detailed rationale for the recommended amount, explaining why this level of support is appropriate given the student’s circumstances.
      5. **Alternative Options**: If recommending against assistance, provide a compassionate rationale and suggest alternative pathways or resources.
    - **Output**: A final recommendation that includes a specific dollar amount for financial assistance, supported by detailed reasoning and aligned with organizational criteria.
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

    income_task = Task(
        description="Calculate the total weekly income from the provided financial document.",
        agent=income_agent,
        input_data=extracted_sections["Income Details"],
        expected_output="""
    ### Expected Output
    - **Accurate Calculation**: Calculate the student's total weekly income, including a detailed breakdown of all income sources (weekly, fortnightly, monthly, and other).
    - **Conversion**: Ensure that all income amounts are converted to weekly equivalents for standardized comparison.
    - **Summary**: Provide a clear summary highlighting key income streams and any potential anomalies or irregularities in the data.
    """,
    )

    living_cost_task = Task(
        description="Calculate the total weekly living costs from the provided financial document.",
        agent=living_cost_agent,
        input_data=extracted_sections["Living Costs"],
        expected_output="""
        ### Expected Output
        - **Precise Calculation**: Calculate the student's total weekly living costs (EXPENSES), including a detailed breakdown of all essential expenses.
        - **Conversion**: Convert non-weekly expenses to weekly equivalents to ensure consistent comparison.
        - **Summary**: Include a summary of the most significant expense categories, and highlight any potential concerns or opportunities for cost reduction.
        """,
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
        expected_output="""
        ### Expected Output
        - **Comprehensive Narrative**: Integrate the student's financial story with detailed income and expense data.
        - **Context**: Identify key challenges, such as employment status, family obligations, or other relevant factors.
        - **Financial Summary**: Clearly outline the student’s financial shortfall or surplus, and discuss the impact on their financial stability.
        - **Insights**: Highlight any critical insights or trends influencing the student's financial health.
        """,
    )

    recommend_task = Task(
        description="Provide a recommendation for financial assistance based on the financial story and MUST give an exact dollar value ie $450 financial hardship fund approved with strong rationale.",
        agent=recommend_agent,
        input_data={
            "story_info": story_task,
        },
        expected_output="""
        ### Expected Output
        - **Well-Reasoned Recommendation**: Synthesize information from income, living costs, and story tasks to provide a recommendation for financial assistance.
        - **Specific Amount**: Include a specific dollar amount for the recommended assistance, with detailed justification.
        - **Alignment with Senior Criteria**: Ensure the recommendation aligns with criteria used by senior managers, making it defensible and equitable.
        - **Impact**: Clearly outline how the recommended financial support will impact the student's situation.
        - **Alternative Guidance**: If declining support, provide a compassionate rationale and suggest alternative options.
        """,
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
