import streamlit as st
import pipe

st.title("Assistant Virtuel")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("question sur les écouteurs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = pipe.run_pipeline(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)