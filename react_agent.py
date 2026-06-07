import os
from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """This is an addition function that adds 2 numbers together."""

    return a + b

@tool
def multiply(a:int, b:int):
    """This tool is used to multiply 2 numbers."""

    return a*b

tools = [add, multiply]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY_K")).bind_tools(tools)
# model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", api_key=os.getenv("GOOGLE_API_KEY")).bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=
     "You are my AI assistant, please answer my query to the best of your abilities."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {'messages':[response]}


def should_continue(state:AgentState):
    messages = state['messages']
    last_message = messages[-1]

    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)



graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
    )



graph.add_edge("tools","our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {'messages':[("user", "Add 40 + 12 and multiply the result by 2, also tell me a joke plz")]}
# inputs = {'messages':[("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke plz.")]}

print_stream(app.stream(inputs, stream_mode="values"))