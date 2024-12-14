# streamlit run c:/repos/ski_sql_llm_chat/main.py

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
import llm_utils
import streamlit as st

st.title("Ski resorts assistant")

if "config" not in st.session_state:
    st.session_state["config"] = RunnableConfig({"configurable": {"thread_id": "ab12"}})

if "llm" not in st.session_state:
    st.session_state["llm"] = llm_utils.get_app()

state = st.session_state.llm.get_state(st.session_state.config)
if state.values:
    for message in state.values['messages']:
        with st.chat_message(message.type):
            st.markdown(message.content)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # add streaming option for messages
        response = st.session_state.llm.invoke(
            {"messages": [HumanMessage(prompt)]},
            st.session_state.config,
        )
        st.markdown(response['messages'][-1].content)
        