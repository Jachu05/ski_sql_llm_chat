from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from typing_extensions import TypedDict, Optional, Literal
from typing import Literal
from pydantic import BaseModel
from langgraph.graph import START, StateGraph, END
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase

load_dotenv()

def get_app():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    db = SQLDatabase.from_uri("sqlite:///ski_resorts.db")

    class State(BaseModel):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        query: str = ''
        result: str = ''

    workflow = StateGraph(state_schema=State)

    class QueryOutput(BaseModel):
        f"""Generated SQL query. Matching {db.dialect} dialect"""

        query: Annotated[str, ..., f"Syntactically valid SQL query. For {db.dialect}"]

    class ShouldSeekInDb(BaseModel):
        """Decision if agent should look for information in database or generate answer based only on llm.
        
        generate_answer - if should generate answer only based on llm.
        write_query - if the agent should look for the information in database.
        """

        content: Literal["generate_answer", "search_db_and_answer"]

    # maybe not that necessary if this is a chain...
    # better would be to use query translation to transform question to be better suited for agent
    # e.g. depends on abstraction to 
    def should_seek_in_db(state: State) -> Literal["generate_answer", "write_query"]:
        """Check if prompt requires to look for some information in database. 
        If yes look for them, if not generate answer only from llm"""

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are expert in routing. Based on conversation messages you have to choose if user's prompt require to look for some information in database.
                    The database schema is as follows:
                    {db.get_table_info()}

                    If the messages require to look for information in database then answer with `write_query`.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt_template | llm.with_structured_output(ShouldSeekInDb)
        response = chain.invoke({'messages': state.messages})
        if response.content == 'generate_answer': # type: ignore
            return "generate_answer"
        return "write_query"

    # change it to tool
    # other more specyfic tools could be added e.g. weather tool that will take info from db and from internet
    def write_query(state: State):
        """Generate SQL query to fetch information."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    Given an input question, create a syntactically correct {db.dialect} query to run to help find the answer.
                    Limit results to resonable ammout based on your assumption. 
                    You can order the results by a relevant column to return the most interesting examples in the database.
                    Never query for all the columns from a specific table, but be adised to use multiple necessary columns.
                    Pay attention to use only the column names that you can see in the schema description:
                    Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
                    Only have single quotes on any sql command sent to the engine.
                    Only use the following tables:
                    {db.get_table_info()}

                    Based on messages conversation generate sql query.
                    Keep track about of what entities the user was asking and use them in query if that might help the user.

                    """
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt_template | llm.with_structured_output(QueryOutput)
        response = chain.invoke({'messages': state.messages})
        return response

    # instead of error better would be validates query according to the question asked ect.
    def rewrite_error_query(state: State):
        query_check_system = f"""
            You are a SQL expert with a strong attention to detail.
            Check if the query is valid.
            Analyze the results. If the error happend before, follow the recent results and make correction based on root cause
            recent results: 
            ```
            {state.result}
            ```

            If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
            query: 
            ```
            {state.query}
            ```
            """
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(query_check_system)
        return result

    def should_check_query(state: State) -> Literal["rewrite_error_query", "generate_answer"]:
        """Check is results contains error, if so check the query."""
        if state.result.startswith('Error'):
            return "rewrite_error_query"
        return "generate_answer"

    def execute_query(state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDataBaseTool(db=db)
        return {"result": execute_query_tool.invoke(state.query.replace('\\', ''))}

    # would be good to make to self reflecting, check if the answer asnwers the question, if the query
    # is correct according to question. 
    # add chat history database, message cleanup like trim, summary
    def generate_answer(state: State):
        """Answer question using retrieved information as context."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    You are helpfull ski resort assistant that helps answer user's questions.
                    You will help user with questions about ski resort, about which information you will have in SQL Query and SQL Result.
                    SQL Result is based on SQL Query and they are related to each other.
                    Be kind. Answer only if you are sure. Say do do not know if you are not sure.
                    Only answer to question related to ski resorts. Do not provide information about other topics.

                    Try yours best and base you answer on following messages conversion, potential corresponding SQL query
                    and SQL database results.
                    Look for potential answer in messages conversation before you move to SQL.
                    If there are no SQL Query and SQL Result you can skip this part and mention to users that you have no data about it.

                    SQL Query: {state.query if state.query else ''}

                    SQL Result: {state.result if state.result else ''}

                    User does not see SQL Query and SQL Result. These are results from your tools so mention to the user what you have found. 
                    Based on that inforamtion, answer the user's question.
                    Present answer in bullet point if there are some options."""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt_template | llm
        response = chain.invoke({'messages': state.messages})
        return {"messages": response}

    workflow.add_conditional_edges(START, should_seek_in_db)
    workflow.add_edge("write_query", "execute_query")
    workflow.add_edge("rewrite_error_query", "execute_query")
    workflow.add_edge("generate_answer", END)
    workflow.add_conditional_edges("execute_query", should_check_query)

    workflow.add_node("write_query", write_query)
    workflow.add_node("rewrite_error_query", rewrite_error_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("generate_answer", generate_answer)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
