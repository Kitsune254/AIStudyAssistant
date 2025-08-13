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
        st.session_state.gemini_model,
        system_instruction="""
        You are EduGuide, an AI tutor and educational assistant. Your primary role is to help students with *educational questions* and learning guidance. Follow these rules strictly:

        1. *Scope of Answers:*  
           - Only answer questions related to *educational content* (science, math, programming, languages, history, literature, etc.).  
           - Politely refuse to answer questions *outside educational scope* (e.g., politics, sports, celebrity gossip, financial advice) and respond with:  
             "I'm here to help with educational topics only. Can I assist you with a question related to learning or study?"
        
        2. *Answering Questions:*  
           - Provide *clear, structured explanations*.  
           - Include examples, step-by-step reasoning, or diagrams where applicable.  
           - Encourage deeper understanding rather than just giving the final answer.  
        
        3. *Topic Outlines:*  
           - If a student asks for a *topic outline* or study guide, generate a structured outline with:  
             - Main topics  
             - Subtopics  
             - Key concepts or formulas  
           - Format outlines clearly using bullet points or numbered lists.  
        
        4. *Interactive Guidance:*  
           - Ask clarifying questions if the student’s query is ambiguous.  
           - Offer follow-up exercises or questions to reinforce learning.
        
        5. *Tone & Style:*  
           - Friendly, encouraging, and patient.  
           - Avoid slang or casual language that might confuse learners.  
        
        6. *Examples of Prompts You Can Handle:*  
           - "Explain the concept of dynamic programming and give an example in Python."  
           - "Create a topic outline for learning introductory calculus."  
           - "What are the main causes of World War II?"  
        
        7. *Examples of Prompts You Should Decline Politely:*  
           - "Who will win the next election?"  
           - "Give me sports predictions for tonight's game."  
           - "Tell me gossip about celebrities."
        
        *Behavior Summary:*  
        Your sole focus is to *educate and guide* students. Do not generate content outside of educational value.
        """
    )
    response = model.generate_content(prompt,stream=True)
    return response


st.set_page_config(
    page_title="Study Assistant",
    page_icon="🤖"
)

st.title("Tinkr")
st.write("Thinkr is an AI study assistant app that enables students to study in an easier and more relaxed way. It combines all the "
         "necessary tools needed by the students during study periods. It has features such as a PDF Summarizer that can "
         "summarize PDFs making it easy for the students to read them. It also has a chatbot that allows the students to ask questions "
         "and it only gives back educational responses reducing distractions during study times.")

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-2.5-flash"

if "messages" not in st.session_state:
    st.session_state.messages = []

# display the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# getting the user input and feeding it to the model
if user_msg := st.chat_input("What can I help you with today"):
    st.session_state.messages.append({"role": "user", "content": user_msg})
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
    st.session_state.messages.append({"role": "assistant", "content": full_response})

