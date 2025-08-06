import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file
gemini_api_key = os.getenv('GEMINI_API_KEY')

# gemini_api_key = st.secrets.GEMINI_API_KEY
# openai_api_key = st.secrets.OPENAI_API_KEY

genai.configure(api_key=gemini_api_key)


# streaming responses to the user
def stream_gemini_response(prompt):
    model = genai.GenerativeModel(
        st.session_state.gemini_model)
    response = model.generate_content(
        prompt,
        stream=True,
    )
    return response


st.set_page_config(
    page_title="Study Assistant",
    page_icon="🤖"
)

st.title("AI Study Helper")

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"

if "general_messages" not in st.session_state:
    st.session_state.general_messages = []

# display the chat history
for msg in st.session_state.general_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# getting the user input and feeding it to the model
if user_msg := st.chat_input("Ask me anything"):
    st.session_state.general_messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            streaming_response = stream_gemini_response(user_msg)
            for chunk in streaming_response:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + " ")
            placeholder.markdown(full_response)

        except Exception as e:
            st.error(e)

    # adding the assistant's final response to the chat history
    st.session_state.general_messages.append({"role": "assistant", "content": full_response})

