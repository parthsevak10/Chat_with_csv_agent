from dotenv import load_dotenv
import streamlit as st
from langchain_experimental.agents import create_csv_agent
from htmlTemplate import css, bot_template, user_template
import os
from langchain_openai import AzureChatOpenAI
 
# Load environment variables
load_dotenv()
 

 
# Initialize LLM
llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)
 

 
# Streamlit app configuration
st.set_page_config(
    page_title="QnA with CSV",
    page_icon=":robot_face:",
    layout="wide"
)
    
st.write(css, unsafe_allow_html=True)
st.title("QnA with CSV 🤖")
 
# Chat history and context storage
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'ticket_context' not in st.session_state:
    st.session_state['ticket_context'] = None
 
# File uploader
user_csv = st.file_uploader(label="", type="csv")
if not user_csv:
    st.warning("Please Upload your CSV file 🚨")
 
# Display the chat messages
if st.session_state['chat_history']:
    for message in st.session_state['chat_history']:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        elif message['role'] == 'bot':
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
 
# User input
user_input = st.chat_input("Ask your Question about CSV files",
                           disabled=not user_csv)
 
# Sidebar example questions
with st.sidebar:
    st.subheader("Example Questions:")
    example_questions = [
        "What is the total number of rows in the CSV?",
        "Count the total number of tickets with high priority?",
        "What are the column names in the CSV?",
        "How many columns does the CSV have?",
    ]
 
    selected_example = st.selectbox("Select an example question:", example_questions, disabled=not user_csv)
 
    if not user_csv:
        st.warning("Please Upload your CSV file 🚨")
 
    if st.button("Use Selected Example"):
        user_input = selected_example
 
# Main logic for query processing
if user_csv is not None:
    agent = create_csv_agent(
        llm=llm,
        path=user_csv,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )
 
    if user_input is not None and user_input.strip() != "":
        # Add user input to chat history
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
 
        st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)
 
        try:
            with st.spinner("Processing..."):
                response = None
 
                # Process the user query
                if "ticket id" in user_input.lower():
                    # Extract ticket ID from the user input (assumes a specific format for simplicity)
                    words = user_input.split()
                    ticket_id = next((word for word in words if word.isdigit()), None)
                    if ticket_id:
                        st.session_state['ticket_context'] = ticket_id
                        response = agent.run(user_input)
 
                elif "that ticket" in user_input.lower() and st.session_state['ticket_context']:
                    ticket_id = st.session_state['ticket_context']
                    response = agent.run(user_input.replace("that ticket", f"ticket ID {ticket_id}"))
 
                else:
                    # General query handling
                    st.session_state['ticket_context'] = None
                    response = agent.run(user_input)
 
                # Store bot response in chat history
                st.session_state['chat_history'].append({'role': 'bot', 'content': response})
 
                # Display bot response
                st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
 
        except ValueError:
            response = "I'm sorry, I can only answer questions related to the uploaded CSV file."
            st.session_state['chat_history'].append({'role': 'bot', 'content': response})
            st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

