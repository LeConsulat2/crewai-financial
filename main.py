import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from crewai import Crew, Agent, Task
import pdfplumber

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
    st.stop()  # Stop the Streamlit script here if the key is missing

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
st.title("Financial Assistance Assessment")

# File upload
uploaded_file = st.file_uploader("Upload your bank statement PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_statement.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract data from PDF
    with pdfplumber.open("uploaded_statement.pdf") as pdf:
        page = pdf.pages[0]
        text = page.extract_text()

    # Process the extracted text
    data = {"Income": [], "Expenses": [], "Documentation": []}
    lines = text.split("\n")

    for line in lines:
        if "TRANSFER" in line or "PAY" in line or "DIRECT" in line or "POS" in line:
            parts = line.split(" ")
            date = parts[0] + " " + parts[1]
            description = " ".join(parts[2:-2])
            if len(parts) >= 4:
                try:
                    withdrawal_amount = float(
                        parts[-1].replace(",", "").replace("$", "")
                    )
                    deposit_amount = float(parts[-2].replace(",", "").replace("$", ""))
                except ValueError:
                    deposit_amount = 0
                    withdrawal_amount = float(
                        parts[-1].replace(",", "").replace("$", "")
                    )
            else:
                deposit_amount = 0
                withdrawal_amount = 0

            if deposit_amount > 0:
                data["Income"].append(
                    {"category": description, "amount": deposit_amount}
                )
            if withdrawal_amount > 0:
                data["Expenses"].append(
                    {"category": description, "amount": withdrawal_amount}
                )

    # Convert the extracted data into DataFrames
    income_df = pd.DataFrame(data["Income"])
    expenses_df = pd.DataFrame(data["Expenses"])

    # Display the extracted data
    st.subheader("Extracted Income Data")
    st.write(income_df)

    st.subheader("Extracted Expenses Data")
    st.write(expenses_df)

    # Calculate totals and shortfall
    total_income = income_df["amount"].sum()
    total_expenses = expenses_df["amount"].sum()
    shortfall = total_expenses - total_income

    # Assessment logic based on AUT guidelines
    def assess_financial_needs(shortfall, documentation):
        if shortfall <= 500:
            return "Approve: $500"
        elif shortfall > 500:
            recommendation = f"Recommend approval of ${shortfall} due to significant shortfall in income during placement period. Documentation provided: {documentation}"
            return recommendation

    assessment = assess_financial_needs(shortfall, data["Documentation"])

    # Create a DataFrame for the final assessment
    assessment_data = {
        "Total Income": [total_income],
        "Total Expenses": [total_expenses],
        "Shortfall": [shortfall],
        "Assessment": [assessment],
    }
    assessment_df = pd.DataFrame(assessment_data)

    # Display assessments
    st.subheader("Financial Assessment")
    st.write(assessment_df)

    # Define the senior student advisor agent
    senior_advisor_agent = Agent(
        role="Senior Student Advisor",
        goal="Assess and make fair and compassionate decisions on financial assistance requests, ensuring each decision aligns with the guidelines and student needs.",
        backstory="""
        You are a senior student advisor with expertise in student financial assistance. You are deeply familiar with the financial assistance guidelines and have a strong understanding of student needs. 
        Your role is to make compassionate decisions for students while maintaining professionalism, ensuring that all assessments are fair and justified. You are capable of approving, declining, or recommending specific amounts of assistance, and your rationale is so sound that any manager reviewing your decisions would agree.
        """,
        verbose=True,
        allow_delegation=False,
    )

    # Define the financial assessment task
    financial_assessment_task = Task(
        description="Assess the financial needs of each student based on their application, documentation, and financial situation. Make a decision to approve, decline, or recommend a specific amount of assistance.",
        agent=senior_advisor_agent,
        expected_output="""
        Your output should include a detailed assessment of the student's financial situation, the decision (approve/decline/recommend), the exact amount to give out if approved or recommended, and a rationale that justifies your decision. The amount should be clearly stated in the format: 'Approved Amount: $X' or 'Recommended Amount: $X'. The rationale should be comprehensive enough that any manager would agree with your assessment.
        """,
        output_file="financial_assessment.md",
    )

    # Define the final decision and documentation task
    final_decision_task = Task(
        description="Document the final decision on the financial assistance request, including the rationale. Prepare the case for review by a manager if the recommended amount exceeds your approval limit.",
        agent=senior_advisor_agent,
        expected_output="""
        Your output should include a clear and concise documentation of the decision made, the exact amount approved or recommended, and the rationale. The amount should be clearly stated in the format: 'Approved Amount: $X' or 'Recommended Amount: $X'. If the amount recommended exceeds $500, include a note that the case requires managerial review.
        """,
        output_file="final_decision.md",
    )

    # Create the crew instance
    crew = Crew(
        tasks=[
            financial_assessment_task,
            final_decision_task,
        ],
        agents=[
            senior_advisor_agent,
        ],
        verbose=2,
    )

    # Kickoff the crew without using 'inputs' argument
    result = crew.kickoff()

    # Display the result
    st.subheader("CrewAI Assessment Result")
    st.write(result)
