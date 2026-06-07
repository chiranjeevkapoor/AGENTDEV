import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

def process(state: AgentState)->AgentState:
    """This node will solve the request you input."""
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(response.content))
    print(f"\n AI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    print(conversation_history)
    result = agent.invoke({'messages':conversation_history})
    conversation_history = result['messages']
    user_input = input("Enter :")
    

with open("logging.txt", "w") as file:
    file.write("Your conversation log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You : {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI : {message.content}\n\n")
    file.write("Conversation is over.")

print("Conversation saved to logging.txt")

    