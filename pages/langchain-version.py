import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from typing import List, Union
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

st.title("Advanced Financial Assistance Assessment")


# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join(page.extract_text() for page in reader.pages)


# Initialize ChatOpenAI
chat_model = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")


# Define custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\nThought: I now know the result of using {action.tool}. "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        return self.template.format(**kwargs)


# Define output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        match = re.match(
            r"Action: (.*?)[\n]*Action Input:[\s]*(.*)", llm_output, re.DOTALL
        )
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


# Define tools
tools = [
    Tool(
        name="CalculateIncome",
        func=lambda x: chat_model.predict(
            f"Calculate the total weekly income for the student. Convert any fortnightly income by dividing by 2 and any monthly income by dividing by 4.\n\n{x}"
        ),
        description="Calculates the total weekly income",
    ),
    Tool(
        name="CalculateExpenses",
        func=lambda x: chat_model.predict(
            f"Calculate the total weekly expenses for the student. Convert any fortnightly expenses by dividing by 2 and any monthly expenses by dividing by 4.\n\n{x}"
        ),
        description="Calculates the total weekly expenses",
    ),
    Tool(
        name="AnalyzeFinancialSituation",
        func=lambda x: chat_model.predict(
            f"Analyze the student's financial situation based on their story, income, and expenses. Consider any shortfall or surplus of the income, and mention factors such as placements, job loss, or rent arrears.\n\n{x}"
        ),
        description="Analyzes the overall financial situation",
    ),
    Tool(
        name="RecommendAssistance",
        func=lambda x: chat_model.predict(
            f"Based on the financial situation and story, provide a final recommendation on how much financial assistance the student should receive. Provide an exact dollar value and logically assess if the assistance should be granted or declined.\n\n{x}"
        ),
        description="Recommends financial assistance amount",
    ),
]

# Define prompt
prompt = CustomPromptTemplate(
    template="""You are an AI assistant designed to process financial assistance applications. Your goal is to analyze the provided information and make a recommendation.

Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to provide a final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: To process this financial assistance application, I need to gather all the necessary information and analyze it step by step.
{agent_scratchpad}""",
    tools=tools,
    input_variables=["input", "intermediate_steps", "tool_names"],
)

# Initialize the agent
llm_chain = LLMSingleActionAgent(
    llm_chain=chat_model,
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools],
    prompt=prompt,
)

# Set up the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=llm_chain,
    tools=tools,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history"),
)

# Streamlit file upload
uploaded_file = st.file_uploader(
    "Upload the Financial Assistance Form Submission PDF", type="pdf"
)

if uploaded_file:
    # Extract text from uploaded PDF
    pdf_text = extract_pdf_text(uploaded_file)

    # Process the application
    with st.spinner("Analyzing the application..."):
        result = agent_executor.run(pdf_text)

    # Display the assessment results
    st.header("Assessment Report")
    st.write(result)

    # Add a download button for the report
    st.download_button(
        label="Download Report",
        data=result,
        file_name="financial_assistance_report.txt",
        mime="text/plain",
    )

    if st.button("Stop Processing"):
        st.warning("Processing stopped by user.")
        st.stop()
