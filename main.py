import streamlit as st
import openai
from PyPDF2 import PdfReader
from crewai import CrewAI
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI

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
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

st.title("Financial Assistance Assessment (Beta)")


# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Initialize CrewAI with agents using ChatOpenAI
crew_ai = CrewAI(api_key=openai_api_key)
chat_model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")


# Define Income Agent
def income_agent(pdf_text):
    prompt = f"Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 and any monthly income by dividing by 4.\n\n{pdf_text}"
    response = chat_model(completion(prompt))
    return response["choices"][0]["message"]["content"].strip()


# Define Expense Agent
def expense_agent(pdf_text):
    prompt = f"Calculate the total weekly expenses for the student. Convert any fortnightly expenses by dividing by 2 and any monthly expenses by dividing by 4.\n\n{pdf_text}"
    response = chat_model(completion(prompt))
    return response["choices"][0]["message"]["content"].strip()


# Define Story Agent
def story_agent(pdf_text, income_info, expense_info):
    prompt = f"Using the following student's story and financial situation, assess their financial situation. Consider any shortfall or surplus of the income, and mention factors such as placements, job loss, or rent arrears.\n\nStory:\n{pdf_text}\n\nWeekly Income:\n{income_info}\n\nWeekly Expenses:\n{expense_info}"
    response = chat_model(completion(prompt))
    return response["choices"][0]["message"]["content"].strip()


# Define Recommend Agent
def recommend_agent(story_info):
    prompt = f"Based on the following financial situation and story, provide a final recommendation on how much financial assistance the student should receive. Provide an exact dollar value and logically assess if the assistance should be granted or declined.\n\n{story_info}"
    response = chat_model(completion(prompt))
    return response["choices"][0]["message"]["content"].strip()


# Streamlit file upload
uploaded_file = st.file_uploader(
    "Upload the Financial Assistance Form Submission PDF", type="pdf"
)

if uploaded_file:
    # Extract text from uploaded PDF
    pdf_text = extract_pdf_text(uploaded_file)

    # Use agents for assessment
    income_info = income_agent(pdf_text)
    expense_info = expense_agent(pdf_text)
    story_info = story_agent(pdf_text, income_info, expense_info)
    recommendation = recommend_agent(story_info)

    # Display the assessment results
    st.header("Assessment Report")
    st.subheader("Income Agent Result:")
    st.write(income_info)
    st.subheader("Expense Agent Result:")
    st.write(expense_info)
    st.subheader("Story Agent Result:")
    st.write(story_info)
    st.subheader("Recommend Agent Result:")
    st.write(recommendation)
