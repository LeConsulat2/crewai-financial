import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import os
from crewai import Crew, Agent, Task
from docx import Document  # For reading Word documents
from functools import lru_cache
import openai
import logging

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


# Extract text from PDF using pdfplumber
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        extracted_text = "\n".join(
            filter(None, [page.extract_text() for page in pdf.pages])
        )
    return extracted_text


# Extract text from DOCX using python-docx
def extract_docx_text(docx_file):
    doc = Document(docx_file)
    extracted_text = "\n".join(
        [para.text for para in doc.paragraphs if para.text.strip() != ""]
    )
    return extracted_text


# Function to extract specific sections of text between start and end keywords
def extract_section(text, start_keyword, end_keyword):
    """
    Extracts a section of text from start_keyword to end_keyword.

    Parameters:
        text (str): The full text to search within.
        start_keyword (str): The starting keyword of the section.
        end_keyword (str): The ending keyword of the section.

    Returns:
        str: Extracted text from start_keyword to end_keyword or "Not found" if keywords are not present.
    """
    start = text.find(start_keyword)
    if start == -1:
        return "Not found"  # Start keyword not found

    start += len(start_keyword)  # Move past the start keyword
    end = text.find(end_keyword, start) if end_keyword else len(text)

    if end == -1:
        end = len(text)  # If end keyword is not found, go to the end of the text

    return text[start:end].strip()


# Function to pass extracted information to agents
def pass_info_to_agents(sections):
    # Define agents
    financial_analysis_agent = Agent(
        role="Financial Analysis Agent",
        goal="Integrate and analyze financial data from the student’s summary to identify key income sources, expenses, and financial gaps.",
        backstory="""
        ### Backstory
        - **Role**: You are a financial expert responsible for creating a clear and comprehensive financial analysis based on the extracted data.
        - **Instructions**:
            1. **Extract Financial Data**: Use 'Financial Situation' section to identify income, expenses, and any financial gaps.
            2. **Calculate Net Position**: Summarize the student’s weekly income and expenses to calculate net income.
            3. **Identify Risks**: Flag any significant financial risks or concerns.
        - **Output**: A detailed financial summary including net income, major expenses, and highlighted financial risks.
        """,
        verbose=True,
        allow_delegation=False,
    )

    contextual_analysis_agent = Agent(
        role="Contextual Analysis Agent",
        goal="Analyze personal and contextual factors from the student’s summary to provide a holistic view of their situation.",
        backstory="""
        ### Backstory
        - **Role**: You are a senior advisor focused on integrating personal and contextual factors into the financial analysis.
        - **Instructions**:
            1. **Extract Contextual Data**: Use 'Presenting Concerns' and 'Protective Factors' sections to identify key challenges.
            2. **Connect Financial and Personal Aspects**: Relate personal circumstances to financial needs and implications.
            3. **Highlight Critical Factors**: Emphasize personal or external factors that could affect financial stability.
        - **Output**: A narrative that connects personal challenges with financial needs, offering a comprehensive view of the student’s situation.
        """,
        verbose=True,
        allow_delegation=False,
    )

    recommendation_agent = Agent(
        role="Recommendation Agent",
        goal="Formulate a recommendation for financial support based on a comprehensive analysis of financial and personal data.",
        backstory="""
        ### Backstory
        - **Role**: As the decision-maker, your role is to synthesize data to provide robust, actionable financial recommendations.
        - **Instructions**:
            1. **Review Analyses**: Combine insights from the Financial and Contextual Analysis Agents.
            2. **Draft Recommendations**: Propose a specific amount for financial assistance or suggest alternative support measures.
            3. **Justify Recommendations**: Provide clear rationale linking recommendations to the student’s financial needs and criteria.
        - **Output**: A detailed recommendation for financial support, including specific amounts and justifications.
        """,
        verbose=True,
        allow_delegation=False,
    )

    # Define tasks
    financial_task = Task(
        description="Analyze the student's financial situation and calculate net income and expenses.",
        agent=financial_analysis_agent,
        input_data=sections.get("Financial Situation", "Not found"),
        expected_output="""
        ### Expected Output
        - **Financial Summary**: A clear breakdown of income, expenses, and net income.
        - **Risk Analysis**: Highlight any financial risks or critical concerns.
        """,
    )

    contextual_task = Task(
        description="Provide a contextual analysis of the student's personal situation and its impact on financial needs.",
        agent=contextual_analysis_agent,
        input_data={
            "presenting_concerns": sections.get("Presenting Concerns", "Not found"),
            "protective_factors": sections.get("Protective Factors", "Not found"),
        },
        expected_output="""
        ### Expected Output
        - **Contextual Summary**: A narrative that integrates personal challenges with financial implications.
        - **Key Insights**: Highlight critical personal factors influencing financial needs.
        """,
    )

    recommendation_task = Task(
        description="Synthesize financial and contextual insights to provide a final recommendation for financial support.",
        agent=recommendation_agent,
        input_data={
            "financial_info": financial_task,
            "contextual_info": contextual_task,
        },
        expected_output="""
        ### Expected Output
        - **Recommendation**: A specific financial support recommendation with clear justification.
        - **Detailed Rationale**: Link recommendations to the student's financial and personal circumstances.
        """,
    )

    # Create Crew instance and run tasks
    agents = [financial_analysis_agent, contextual_analysis_agent, recommendation_agent]
    tasks = [financial_task, contextual_task, recommendation_task]

    crew = Crew(
        tasks=tasks,
        agents=agents,
        verbose=True,
        on_thought=lambda agent_name, thought: log_agent_communication(
            agent_name, f"Thought: {thought}"
        ),
        on_message=lambda agent_name, message: log_agent_communication(
            agent_name, f"Message: {message}"
        ),
    )

    with st.spinner("Analysis in progress..."):
        try:
            result = crew.kickoff()
            st.success("Analysis complete. Recommendation generated.")

            # Display results
            for i, task in enumerate(tasks):
                st.write(f"Completed task {i + 1}/{len(tasks)}: {task.description}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    st.subheader("CrewAI Assessment Result")
    st.write(result)  # Displaying the result directly in the app


# Logging function
def log_agent_communication(agent_name, message):
    st.write(f"[{agent_name}]: {message}")


# File uploader and main processing logic
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        extracted_text = extract_pdf_text(uploaded_file)
    elif (
        uploaded_file.type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        extracted_text = extract_docx_text(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    extracted_sections = extract_key_sections(extracted_text)

    # Display extracted sections
    for section, content in extracted_sections.items():
        st.subheader(section)
        st.write(content)

    if st.button("Process"):
        pass_info_to_agents(extracted_sections)

    if st.button("Stop Processing"):
        st.warning("Processing stopped by user.")
        st.stop()
