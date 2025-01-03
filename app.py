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
 
# # Templates for predefined sub-category responses
# TEMPLATES = {
#     'GDPR': "Ensure that the data processing aligns with GDPR standards and protects user privacy.",
#     'BCP/DR': "Review the Business Continuity Plan and Disaster Recovery measures to mitigate risks.",
#     'Firewall': "Verify the firewall rules and configurations to ensure secure network boundaries.",
#     'Password non-expiry': "Address the potential risks of non-expiring passwords and enforce password policies.",
#     'Malware Attack': "Investigate the malware attack source and follow standard response protocols.",
#     'New Application Launch': "Validate the security and compliance requirements for the new application.",
#     'Application Offboarding': "Ensure proper decommissioning of applications, including data removal.",
#     'Insider Threat': "Assess the insider threat indicators and strengthen access controls.",
#     'HIPAA': "Ensure that patient data handling complies with HIPAA regulations.",
#     'Segregation of Duties': "Verify proper role assignments to prevent conflicts of interest.",
#     'Database Security': "Assess the security measures in place for database protection.",
#     'Secure Development': "Review coding practices to ensure secure software development.",
#     'PCI-DSS': "Ensure compliance with PCI-DSS for handling payment card data securely.",
#     'Privacy Assessment': "Conduct a thorough privacy impact assessment.",
#     'Obsolete Software': "Identify and replace obsolete software to mitigate vulnerabilities.",
#     'Spyware': "Investigate the spyware detection and follow response protocols.",
#     'Unauthorized software': "Review and remove any unauthorized software from the system.",
#     'Trojan Attack': "Identify and isolate systems affected by the trojan attack.",
#     'Phishing': "Educate users and review email security measures to mitigate phishing risks.",
#     'Vishing': "Address vishing incidents by validating phone-based interactions.",
#     'Encryption': "Ensure robust encryption practices for sensitive data.",
#     'Access Management': "Review and enhance access management policies.",
#     'Unauthorized access': "Investigate unauthorized access incidents and strengthen security controls.",
#     'Third Party Engagement': "Assess third-party engagements for compliance and security.",
#     'Network Security': "Evaluate network security measures and identify gaps."
# }
 
 
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


    











# from dotenv import load_dotenv
# import streamlit as st
# from langchain_experimental.agents import create_csv_agent
# from htmlTemplate import css, bot_template, user_template
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain.schema import Document
# from PyPDF2 import PdfReader

# load_dotenv()
# # Set up the Streamlit interface
# st.set_page_config(
#     page_title="QnA with CSV and Knowledge Base",
#     page_icon=":robot_face:",
#     layout="wide"
# )
# # Set environment variables for Azure OpenAI
# os.environ["AZURE_OPENAI_API_KEY"] = "56aa01103e8b40359783aa5ce46e5ae8"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-test-vraj.openai.azure.com/"
# os.environ["OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# # Initialize LLM for CSV
# csv_llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)

# # Function to process PDF

# def process_pdf(file_path):
#     documents = []
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     documents.append(Document(page_content=text))

#     # Split the documents into smaller chunks
#     text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=500)
#     split_docs = text_splitter.split_documents(documents)

#     # Create Hugging Face embeddings
#     huggingface_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#     # Create a FAISS vector database
#     vectordb = FAISS.from_documents(documents=split_docs, embedding=huggingface_embeddings)
#     return vectordb

# # Process the default knowledge base PDF
# knowledge_base_path = "Knowledge_base.pdf"
# if os.path.exists(knowledge_base_path):
#     with st.spinner("Processing the knowledge base PDF..."):
#         pdf_vectordb = process_pdf(knowledge_base_path)
#         pdf_retriever = pdf_vectordb.as_retriever(search_kwargs={"k": 3})
#         pdf_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         pdf_llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)
#         pdf_qa_chain = ConversationalRetrievalChain.from_llm(llm=pdf_llm, retriever=pdf_retriever, memory=pdf_memory)
#     st.success("Knowledge base loaded successfully!")
# else:
#     st.error("Knowledge base PDF not found. Please place a file named 'Knowledge_base.pdf' in the root folder.")


# st.write(css, unsafe_allow_html=True)
# st.title("QnA with CSV and Knowledge Base 🤖")

# # Information about the knowledge base
# st.markdown("""
# ### 🛠️ Knowledge Base:
# The knowledge base PDF is being used to answer questions related to tickets. 

# If you want to update the knowledge base, replace the current file in the root folder with a new PDF named **'Knowledge_base.pdf'**.
# """, unsafe_allow_html=True)

# # File uploader for CSV
# user_csv = st.file_uploader(label="Upload CSV File", type="csv")

# if not user_csv:
#     st.warning("Please upload a CSV file to proceed.")

# # Sidebar with example questions
# with st.sidebar:
#     st.subheader("Example Questions:")
#     example_questions = [
#         "What is the total number of rows in the CSV?",
#         "Count the total number of tickets with high priority?",
#         "What are the column names in the CSV?",
#         "How many columns does the CSV have?",
#         "What details are mentioned in the knowledge base?",
#     ]
#     selected_example = st.selectbox("Select an example question:", example_questions, disabled=not user_csv)

#     if st.button("Use Selected Example"):
#         st.session_state.user_input = selected_example

# # Process CSV file
# if user_csv:
#     csv_agent = create_csv_agent(
#         llm=csv_llm,
#         path=user_csv,
#         verbose=True,
#         handle_parsing_errors=True,
#         allow_dangerous_code=True
#     )

# # User input
# user_input = st.chat_input("Ask your question about CSV or Knowledge Base:", disabled=not (user_csv or os.path.exists(knowledge_base_path)))

# if user_input:
#     st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

#     if user_csv and "csv" in user_input.lower():
#         try:
#             with st.spinner("Querying CSV..."):
#                 csv_response = csv_agent.run(user_input)
#                 st.write(bot_template.replace("{{MSG}}", csv_response), unsafe_allow_html=True)
#         except Exception as e:
#             st.write(bot_template.replace("{{MSG}}", f"Error querying CSV: {str(e)}"), unsafe_allow_html=True)

#     elif os.path.exists(knowledge_base_path) and "pdf" in user_input.lower():
#         try:
#             with st.spinner("Querying Knowledge Base PDF..."):
#                 pdf_response = pdf_qa_chain({"question": user_input})
#                 st.write(bot_template.replace("{{MSG}}", pdf_response['answer']), unsafe_allow_html=True)
#         except Exception as e:
#             st.write(bot_template.replace("{{MSG}}", f"Error querying Knowledge Base: {str(e)}"), unsafe_allow_html=True)

#     else:
#         st.write(bot_template.replace("{{MSG}}", "Please specify whether the question relates to the CSV or Knowledge Base."), unsafe_allow_html=True)

# Working code which can query both csv and pdf


# from dotenv import load_dotenv
# import streamlit as st
# from langchain_experimental.agents import create_csv_agent
# from htmlTemplate import css, bot_template, user_template
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain.schema import Document
# from PyPDF2 import PdfReader

# load_dotenv()


# # Set up the Streamlit interface
# st.set_page_config(
#     page_title="QnA with CSV and Knowledge Base",
#     page_icon=":robot_face:",
#     layout="wide"
# )


# # Set environment variables for Azure OpenAI
# os.environ["AZURE_OPENAI_API_KEY"] = "56aa01103e8b40359783aa5ce46e5ae8"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-test-vraj.openai.azure.com/"
# os.environ["OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# # Initialize LLM for CSV
# csv_llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)

# # Function to process PDF

# def process_pdf(file_path):
#     documents = []
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         text += page.extract_text()
#     documents.append(Document(page_content=text))

#     # Split the documents into smaller chunks
#     text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1500, chunk_overlap=500)
#     split_docs = text_splitter.split_documents(documents)

#     # Create Hugging Face embeddings
#     huggingface_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#     # Create a FAISS vector database
#     vectordb = FAISS.from_documents(documents=split_docs, embedding=huggingface_embeddings)
#     return vectordb

# # Process the default knowledge base PDF
# knowledge_base_path = "Knowledge_base.pdf"
# if os.path.exists(knowledge_base_path):
#     with st.spinner("Processing the knowledge base PDF..."):
#         pdf_vectordb = process_pdf(knowledge_base_path)
#         pdf_retriever = pdf_vectordb.as_retriever(search_kwargs={"k": 3})
#         pdf_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         pdf_llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)
#         pdf_qa_chain = ConversationalRetrievalChain.from_llm(llm=pdf_llm, retriever=pdf_retriever, memory=pdf_memory)
#     st.success("Knowledge base loaded successfully!")
# else:
#     st.error("Knowledge base PDF not found. Please place a file named 'Knowledge_base.pdf' in the root folder.")


# st.write(css, unsafe_allow_html=True)
# st.title("QnA with CSV and Knowledge Base 🤖")

# # Information about the knowledge base
# st.markdown("""
# ### 🛠️ Knowledge Base:
# The knowledge base PDF is being used to answer questions related to tickets. 

# If you want to update the knowledge base, replace the current file in the root folder with a new PDF named **'Knowledge_base.pdf'**.
# """, unsafe_allow_html=True)

# # File uploader for CSV
# user_csv = st.file_uploader(label="Upload CSV File", type="csv")

# if not user_csv:
#     st.warning("Please upload a CSV file to proceed.")

# # # Sidebar with example questions
# # with st.sidebar:
# #     st.subheader("Example Questions:")
# #     example_questions = [
# #         "What is the total number of rows in the CSV?",
# #         "Count the total number of tickets with high priority?",
# #         "What are the column names in the CSV?",
# #         "How many columns does the CSV have?",
# #         "What details are mentioned in the knowledge base?",
# #     ]
# #     selected_example = st.selectbox("Select an example question:", example_questions, disabled=not user_csv)

# #     if st.button("Use Selected Example"):
# #         st.session_state.user_input = selected_example

# # Process CSV file
# if user_csv:
#     csv_agent = create_csv_agent(
#         llm=csv_llm,
#         path=user_csv,
#         verbose=True,
#         handle_parsing_errors=True,
#         allow_dangerous_code=True
#     )

# # User input
# user_input = st.chat_input("Ask your question about CSV or Knowledge Base:", disabled=not (user_csv or os.path.exists(knowledge_base_path)))

# if user_input:
#     st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

#     if user_csv and "csv" in user_input.lower():
#         try:
#             with st.spinner("Querying CSV..."):
#                 csv_response = csv_agent.run(user_input)
#                 st.write(bot_template.replace("{{MSG}}", csv_response), unsafe_allow_html=True)
#         except Exception as e:
#             st.write(bot_template.replace("{{MSG}}", f"Error querying CSV: {str(e)}"), unsafe_allow_html=True)

#     elif os.path.exists(knowledge_base_path) and "pdf" in user_input.lower():
#         try:
#             with st.spinner("Querying Knowledge Base PDF..."):
#                 pdf_response = pdf_qa_chain({"question": user_input})
#                 st.write(bot_template.replace("{{MSG}}", pdf_response['answer']), unsafe_allow_html=True)
#         except Exception as e:
#             st.write(bot_template.replace("{{MSG}}", f"Error querying Knowledge Base: {str(e)}"), unsafe_allow_html=True)

#     else:
#         st.write(bot_template.replace("{{MSG}}", "Please specify whether the question relates to the CSV or Knowledge Base."), unsafe_allow_html=True)


# from dotenv import load_dotenv
# import streamlit as st
# from langchain_experimental.agents import create_csv_agent
# from htmlTemplate import css, bot_template, user_template
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import Document
# from PyPDF2 import PdfReader

# load_dotenv()

# # Set up Streamlit interface
# st.set_page_config(
#     page_title="QnA with CSV and Knowledge Base",
#     page_icon=":robot_face:",
#     layout="wide"
# )

# # Load environment variables
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
# os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
# os.environ["OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# # Initialize LLM
# llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)

# # Function to process PDF
# def process_pdf(file_path):
#     """Reads and processes a PDF file into a FAISS vector database."""
#     documents = []
#     reader = PdfReader(file_path)
#     for page in reader.pages:
#         text = page.extract_text()
#         if text:
#             documents.append(Document(page_content=text))

#     # Split text into manageable chunks
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=50, chunk_overlap=20)
#     split_docs = text_splitter.split_documents(documents)

#     # Create embeddings and FAISS vector database
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(split_docs, embedding=embeddings)

#     return vectordb

# # Load knowledge base
# knowledge_base_path = "Knowledge_base.pdf"
# pdf_qa_chain = None

# if os.path.exists(knowledge_base_path):
#     with st.spinner("Processing the knowledge base PDF..."):
#         pdf_vectordb = process_pdf(knowledge_base_path)
#         pdf_retriever = pdf_vectordb.as_retriever(search_kwargs={"k": 3})
#         pdf_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         pdf_qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm, retriever=pdf_retriever, memory=pdf_memory
#         )
#     st.success("Knowledge base loaded successfully!")
# else:
#     st.error("Knowledge base PDF not found. Please place a file named 'Knowledge_base.pdf' in the root folder.")

# st.write(css, unsafe_allow_html=True)
# st.title("QnA with CSV and Knowledge Base 🤖")

# # File uploader for CSV
# user_csv = st.file_uploader(label="Upload CSV File", type="csv")

# csv_agent = None
# if user_csv:
#     csv_agent = create_csv_agent(
#         llm=llm,
#         path=user_csv,
#         verbose=True,
#         handle_parsing_errors=True,
#         allow_dangerous_code=True
#     )

# # Sidebar example questions
# with st.sidebar:
#     st.subheader("Example Questions:")
#     example_questions = [
#         "What is the total number of rows in the CSV?",
#         "Count the total number of tickets with high priority?",
#         "What are the column names in the CSV?",
#         "How many columns does the CSV have?",
#         "What details are mentioned in the knowledge base?",
#     ]
#     selected_example = st.selectbox("Select an example question:", example_questions, disabled=not (csv_agent or pdf_qa_chain))

#     if st.button("Use Selected Example"):
#         st.session_state.user_input = selected_example

# # User input
# user_input = st.chat_input(
#     "Ask your question about CSV or Knowledge Base:",
#     disabled=not (csv_agent or pdf_qa_chain)
# )

# if user_input:
#     st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

#     response = None

#     # Query CSV
#     if csv_agent:
#         try:
#             with st.spinner("Querying CSV..."):
#                 response = csv_agent.run(user_input)
#         except Exception as e:
#             st.error(f"Error querying CSV: {e}")

#     # Query Knowledge Base
#     if not response and pdf_qa_chain:
#         try:
#             with st.spinner("Querying Knowledge Base PDF..."):
#                 pdf_response = pdf_qa_chain({"question": user_input})
#                 if pdf_response["answer"].strip():  # If an answer exists
#                     response = pdf_response["answer"]
#                 else:  # Fallback if no answer found in PDF
#                     response = (
#                         "The question couldn't be answered from the knowledge base. "
#                         "Switching to general knowledge...\n\n"
#                     )
#                     response += llm(user_input)
#         except Exception as e:
#             st.error(f"Error querying Knowledge Base: {e}")

#     # Display the response
#     if response:
#         st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
#     else:
#         st.write(
#             bot_template.replace(
#                 "{{MSG}}", "I'm sorry, I couldn't find an answer in either the CSV or the Knowledge Base."
#             ),
#             unsafe_allow_html=True
#         )

#########################################################################
#########################################################################

########## Working Code Demo ##########

#########################################################################
#########################################################################


# from dotenv import load_dotenv
# import streamlit as st
# from langchain_experimental.agents import create_csv_agent
# from htmlTemplate import css, bot_template, user_template
# import os
# from langchain_openai import AzureChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import Document
# from PyPDF2 import PdfReader

# # Load environment variables
# load_dotenv()

# # Set up Streamlit interface
# st.set_page_config(
#     page_title="QnA with CSV and Knowledge Base",
#     page_icon=":robot_face:",
#     layout="wide"
# )




# # Set environment variables for Azure OpenAI
# os.environ["AZURE_OPENAI_API_KEY"] = "56aa01103e8b40359783aa5ce46e5ae8"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-test-vraj.openai.azure.com/"
# os.environ["OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# # Initialize LLM
# llm = AzureChatOpenAI(model="gpt-35-turbo", temperature=0)

# # Function to process PDF
# def process_pdf(file_path):
#     """Reads and processes a PDF file into a FAISS vector database."""
#     documents = []
#     reader = PdfReader(file_path)
#     for page in reader.pages:
#         text = page.extract_text()
#         if text:
#             documents.append(Document(page_content=text))

#     # Split text into manageable chunks
#     text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=200)
#     split_docs = text_splitter.split_documents(documents)

#     # Create embeddings and FAISS vector database
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(split_docs, embedding=embeddings)

#     return vectordb

# # Load knowledge base
# knowledge_base_path = "Knowledge_base.pdf"
# pdf_qa_chain = None

# if os.path.exists(knowledge_base_path):
#     with st.spinner("Processing the knowledge base PDF..."):
#         pdf_vectordb = process_pdf(knowledge_base_path)
#         pdf_retriever = pdf_vectordb.as_retriever(search_kwargs={"k": 3})
#         pdf_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         pdf_qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm, retriever=pdf_retriever, memory=pdf_memory
#         )
#     st.success("Knowledge base loaded successfully!")
# else:
#     st.error("Knowledge base PDF not found. Please place a file named 'Knowledge_base.pdf' in the root folder.")

# st.write(css, unsafe_allow_html=True)
# st.title("QnA with CSV and Knowledge Base 🤖")

# # File uploader for CSV
# user_csv = st.file_uploader(label="Upload CSV File", type="csv")

# csv_agent = None
# if user_csv:
#     csv_agent = create_csv_agent(
#         llm=llm,
#         path=user_csv,
#         verbose=True,
#         handle_parsing_errors=True,
#         allow_dangerous_code=True
#     )



# # User input
# user_input = st.chat_input(
#     "Ask your question about CSV or Knowledge Base:",
#     disabled=not (csv_agent or pdf_qa_chain)
# )

# if user_input:
#     st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

#     response = None

#     # Query CSV
#     if csv_agent:
#         try:
#             with st.spinner("Querying CSV..."):
#                 response = csv_agent.run(user_input)
#                 # If response is explicitly "N/A", treat it as no response
#                 if response.strip().upper() == "N/A":
#                     response = None
#         except Exception as e:
#             st.error(f"Error querying CSV: {e}")

#     # Query Knowledge Base if response is None or "N/A"
#     if not response and pdf_qa_chain:
#         try:
#             with st.spinner("Querying Knowledge Base PDF..."):
#                 pdf_response = pdf_qa_chain({"question": user_input})
#                 if pdf_response["answer"].strip():  # If an answer exists
#                     response = pdf_response["answer"]
#                 else:  # Fallback if no answer found in PDF
#                     response = (
#                         "The question couldn't be answered from the knowledge base. "
#                         "Switching to general knowledge...\n\n"
#                     )
#                     response += llm(user_input)
#         except Exception as e:
#             st.error(f"Error querying Knowledge Base: {e}")

#     # Display the response
#     if response:
#         st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
#     else:
#         st.write(
#             bot_template.replace(
#                 "{{MSG}}", "I'm sorry, I couldn't find an answer in either the CSV or the Knowledge Base."
#             ),
#             unsafe_allow_html=True
#         )
