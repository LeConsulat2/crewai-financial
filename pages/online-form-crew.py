import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import os
import logging
from crewai import Crew, Agent, Task


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
    goal="Accurately calculate the total weekly income from the student's financial information, focusing on identifying all income streams and converting them into weekly amounts.",
    backstory="""
    ### Backstory
    - **Role**: You are a financial analysis expert specializing in calculating weekly income from various sources. Your calculations are critical in determining the student's current financial standing and assessing their immediate income capabilities.
    - **Instructions**:
      1. **Extract Income Information**: Use the input data labeled "Income Details" to gather all available income sources.
      2. **Identify Income Types**: Identify all forms of income, including weekly, fortnightly, monthly, one-time payments, and other irregular income.
      3. **Perform Conversions**: Convert income into weekly equivalents: divide fortnightly income by 2, monthly income by 4, and apply relevant conversions for any irregular income sources.
      4. **Summarize and Validate**: Sum all weekly incomes, validate the accuracy, and provide a clear summary that highlights any unusual income patterns or potential concerns.
    - **Output**: The output should be the total weekly income, including a detailed breakdown, conversion methods, and any assumptions made.
    """,
    verbose=True,
    allow_delegation=False,
)

living_cost_agent = Agent(
    role="Living Cost Agent",
    goal="Precisely calculate the total weekly living costs, focusing on essential expenses that align with the AUT financial hardship support criteria.",
    backstory="""
    ### Backstory
    - **Role**: You are an expert in financial budgeting, specifically in calculating essential living expenses that reflect the student's immediate financial commitments. Your analysis helps determine the gap between income and necessary expenditures.
    - **Instructions**:
      1. **Extract Living Costs**: Use the input labeled "Living Costs" to identify necessary expenses such as rent, food, utilities, transport, and childcare.
      2. **Categorize and Prioritize**: Ensure expenses align with eligible categories under AUT’s financial support criteria, prioritizing those that address immediate and essential needs.
      3. **Perform Conversions**: Convert all non-weekly expenses into weekly amounts: divide fortnightly costs by 2, and monthly costs by 4, ensuring all values are standardized.
      4. **Summarize and Highlight**: Provide a comprehensive total of weekly living costs, highlighting key categories and noting any significant or urgent costs.
    - **Output**: The output should be a total weekly expense figure, with detailed explanations of categories and conversions used.
    """,
    verbose=True,
    allow_delegation=False,
)

story_agent = Agent(
    role="Story Agent",
    goal="Create a comprehensive and insightful narrative that integrates the student's financial data, focusing on the temporary nature of their needs and the specific challenges they face.",
    backstory="""
    ### Backstory
    - **Role**: As a senior advisor, you excel at synthesizing financial and contextual information to provide a complete understanding of the student's situation. Your narrative will guide decision-making by highlighting the student's immediate challenges and the impact of their financial shortfall.
    - **Instructions**:
      1. **Integrate Income and Expense Data**: Utilize the inputs from income and living cost agents to provide a balanced view of the student's financial status.
      2. **Contextualize Challenges**: Identify specific challenges impacting the student's finances, such as waiting for allowance payments, placement periods, or sudden emergencies.
      3. **Highlight Critical Factors**: Emphasize the temporary nature of the financial need and the impact on the student's ability to continue their studies without additional support.
      4. **Provide a Clear Summary**: Offer a concise yet detailed summary of the financial shortfall or surplus, key challenges, and any notable trends or patterns.
    - **Output**: The output should be a well-rounded financial story that captures the student's immediate needs and provides context for the recommendation.
    """,
    verbose=True,
    allow_delegation=False,
)

recommend_agent = Agent(
    role="Recommend Agent",
    goal="Provide a precise and justified recommendation for financial assistance, using all available data to ensure the recommendation is robust, aligned with AUT criteria, and addresses the student's immediate needs.",
    backstory="""
    ### Backstory
    - **Role**: As a senior financial advisor, you are responsible for making sound, justifiable recommendations on financial assistance. Your decisions must be based on thorough analysis, align with AUT’s support criteria, and address the student's specific and immediate needs.
    - **Instructions**:
      1. **Synthesize Data**: Combine insights from the income, living costs, and story agents to form a complete picture of the student's financial situation.
      2. **Assess Financial Need**: Evaluate the gap between income and essential expenses, factoring in any urgent or temporary needs that support would address.
      3. **Make a Decision**: Determine the exact amount of financial assistance required, ensuring it covers the specific period of need (e.g., 3-4 weeks).
      4. **Provide Detailed Justification**: Offer a robust rationale for the recommended amount, linking it directly to the student's circumstances and ensuring alignment with AUT’s criteria for temporary, one-off support.
      5. **Suggest Alternatives if Necessary**: If declining support, provide a clear rationale and suggest alternative options or next steps for the student.
    - **Output**: The output should be a final recommendation with a specific dollar amount, fully justified and aligned with the temporary nature of the support required.
    """,
    verbose=True,
    allow_delegation=False,
)

# After defining all the agents

# Set up logging to capture agent communications
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AgentCommunication")


# Function to log agent communication in Streamlit
def log_agent_communication(agent_name, message):
    st.write(f"[{agent_name}]: {message}")  # Displaying directly in the Streamlit app


# Define income_agent with logging hooks
income_agent = Agent(
    role="Income Agent",
    goal="Accurately calculate the total weekly income from the student's financial information, focusing on identifying all income streams and converting them into weekly amounts.",
    backstory="""
    ### Backstory
    - **Role**: You are a financial analysis expert specializing in calculating weekly income from various sources. Your calculations are critical in determining the student's current financial standing and assessing their immediate income capabilities.
    - **Instructions**:
      1. **Extract Income Information**: Use the input data labeled "Income Details" to gather all available income sources.
      2. **Identify Income Types**: Identify all forms of income, including weekly, fortnightly, monthly, one-time payments, and other irregular income.
      3. **Perform Conversions**: Convert income into weekly equivalents: divide fortnightly income by 2, monthly income by 4, and apply relevant conversions for any irregular income sources.
      4. **Summarize and Validate**: Sum all weekly incomes, validate the accuracy, and provide a clear summary that highlights any unusual income patterns or potential concerns.
    - **Output**: The output should be the total weekly income, including a detailed breakdown, conversion methods, and any assumptions made.
    """,
    verbose=True,
    allow_delegation=False,
    on_thought=lambda thought: log_agent_communication(
        "Income Agent", f"Thought: {thought}"
    ),
    on_action=lambda action, input: log_agent_communication(
        "Income Agent", f"Action: {action} | Input: {input}"
    ),
)

# Define living_cost_agent with logging hooks
living_cost_agent = Agent(
    role="Living Cost Agent",
    goal="Precisely calculate the total weekly living costs, focusing on essential expenses that align with AUT financial hardship support criteria.",
    backstory="""
    ### Backstory
    - **Role**: You are an expert in financial budgeting, specifically in calculating essential living expenses that reflect the student's immediate financial commitments. Your analysis helps determine the gap between income and necessary expenditures.
    - **Instructions**:
      1. **Extract Living Costs**: Use the input labeled "Living Costs" to identify necessary expenses such as rent, food, utilities, transport, and childcare.
      2. **Categorize and Prioritize**: Ensure expenses align with eligible categories under AUT’s financial support criteria, prioritizing those that address immediate and essential needs.
      3. **Perform Conversions**: Convert all non-weekly expenses into weekly amounts: divide fortnightly costs by 2, and monthly costs by 4, ensuring all values are standardized.
      4. **Summarize and Highlight**: Provide a comprehensive total of weekly living costs, highlighting key categories and noting any significant or urgent costs.
    - **Output**: The output should be a total weekly expense figure, with detailed explanations of categories and conversions used.
    """,
    verbose=True,
    allow_delegation=False,
    on_thought=lambda thought: log_agent_communication(
        "Living Cost Agent", f"Thought: {thought}"
    ),
    on_action=lambda action, input: log_agent_communication(
        "Living Cost Agent", f"Action: {action} | Input: {input}"
    ),
)

# Define story_agent with logging hooks
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
    on_thought=lambda thought: log_agent_communication(
        "Story Agent", f"Thought: {thought}"
    ),
    on_action=lambda action, input: log_agent_communication(
        "Story Agent", f"Action: {action} | Input: {input}"
    ),
)

# Define recommend_agent with logging hooks
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
    on_thought=lambda thought: log_agent_communication(
        "Recommend Agent", f"Thought: {thought}"
    ),
    on_action=lambda action, input: log_agent_communication(
        "Recommend Agent", f"Action: {action} | Input: {input}"
    ),
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

    # Define tasks with advanced sophistication
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
        description="Calculate the total weekly living costs from the provided financial document, focusing on essential expenses that meet AUT's criteria.",
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
        description="Compile a comprehensive story based on income, expenses, and additional financial data, emphasizing the temporary nature of financial needs.",
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
        description="Provide a recommendation for financial assistance based on the financial story, income, and expense details. MUST give an exact dollar value (e.g., $450 financial hardship fund approved) with strong rationale.",
        agent=recommend_agent,
        input_data={
            "income_info": income_task,  # Include income details
            "living_cost_info": living_cost_task,  # Include living costs details
            "story_info": story_task,  # Include the comprehensive financial story
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

    crew = Crew(
        tasks=tasks,
        agents=agents,
        verbose=True,
        # Capture agent thoughts and communications
        on_thought=lambda agent_name, thought: log_agent_communication(
            agent_name, f"Thought: {thought}"
        ),
        on_message=lambda agent_name, message: log_agent_communication(
            agent_name, f"Message: {message}"
        ),
    )

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

    # Display the final result in a nicely formatted Markdown
    st.subheader("CrewAI Assessment Result")
    result_markdown = """
    **CrewAI Assessment Result**

    **Recommendation:**
    Based on the student's financial situation, I recommend a one-off financial support of **$820**. This amount is calculated based on their current monthly shortfall of approximately **$205**, covering a period of four weeks. This support will ensure they can meet their essential living costs without going into further debt or experiencing undue hardship.

    **Rationale:**
    - **Income and Expenses Analysis:** The student's weekly living costs amount to **$657.50**, while their weekly income is **$606.25**, resulting in a weekly shortfall of **$51.25**. Over a 4-week period, this shortfall accumulates to a total of **$205**, which is the minimum amount required to bridge this gap.
    - **Buffer for Unexpected Costs:** The student's budget leaves little room for unexpected costs or emergencies. Therefore, an additional buffer of **$155** (approximately **$38.75** per week) has been included in this recommendation to provide a measure of financial security over the four-week period.
    - **Alignment with Criteria:** This recommendation aligns with AUT's criteria for temporary, one-off support and is designed to provide immediate relief. It addresses the student's unique circumstances and immediate needs, ensuring that the financial assistance is both targeted and effective.

    **Conclusion:**
    This tailored solution aims to reduce the student's financial stress and allow them to focus on their studies, providing the necessary support during this temporary period of financial difficulty.
    """

    st.markdown(result_markdown)

    # Stop execution after displaying the result to avoid unintended reruns
    st.stop()
