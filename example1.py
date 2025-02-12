import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage,ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from info import csv_database


@tool
def csv_db(question):
    """
    This tool searches a database of CSVs for the best answer to a question.
    CSV is about tourism information in pakistan
    """
    return csv_database(question)


tools = [csv_db]

PROMPT = """You are a chatbot that speaks english.
You will answer every question which is related to the data in tools {tools}.
And if offtopic question is asked, the say "I am sorry".
"""

model = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools)
messages = [SystemMessage(PROMPT)]

def chatbot(question):
    messages.append(HumanMessage(question))
    response = model.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "csv_db":
                pdf_data = csv_db.invoke(tool_call["args"]["question"])
                messages.append(ToolMessage(pdf_data,tool_call_id = tool_call["id"] ))
                
        response = model.invoke(messages)
        messages.append(response)
    return response.content



