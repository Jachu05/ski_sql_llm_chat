from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
import llm_utils
import streamlit as st

st.title("ChatGPT-like clone")

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
        stream = st.session_state.llm.stream(
            {"messages": [HumanMessage(prompt)]},
            st.session_state.config,
            stream_mode="messages",
        )

        def format_stream(stream):
            for message, _ in stream:
                yield message.content

        response = st.write_stream(format_stream(stream))
        