from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

load_dotenv()

def get_app():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # Define a new graph
    workflow = StateGraph(state_schema=State)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are helpfull assistance and answering in warhammer 40k style.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt_template | llm

    # Define the function that calls the model
    def call_model(state: State):
        response = chain.invoke({'messages': state["messages"]})
        return {"messages": response}


    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
