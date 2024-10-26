# import streamlit as st
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Set the page title
# st.title("Investment Options Advisor")

# # Load the dataset
# @st.cache_resource
# def load_data():
#     data = pd.read_csv("Finance_data.csv")
#     return data.to_dict(orient='records')

# data_fin = load_data()

# # Prepare prompt-response format
# def prepare_prompt_response(entry):
#     prompt = f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#     response = (
#         f"Based on your preferences, here are your investment options:\n"
#         f"- Mutual Funds: {entry['Mutual_Funds']}\n"
#         f"- Equity Market: {entry['Equity_Market']}\n"
#         f"- Debentures: {entry['Debentures']}\n"
#         f"- Government Bonds: {entry['Government_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
#         f"- PPF: {entry['PPF']}\n"
#         f"- Gold: {entry['Gold']}\n"
#         f"Factors considered: {entry['Factor']}\n"
#         f"Objective: {entry['Objective']}\n"
#         f"Expected returns: {entry['Expect']}\n"
#         f"Investment monitoring: {entry['Invest_Monitor']}\n"
#         f"Reasons for choices:\n"
#         f"- Equity: {entry['Reason_Equity']}\n"
#         f"- Mutual Funds: {entry['Reason_Mutual']}\n"
#         f"- Bonds: {entry['Reason_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Reason_FD']}\n"
#         f"Source of information: {entry['Source']}\n"
#     )
#     return prompt, response

# # Create prompts and responses for each entry in the dataset
# prompt_response_data = [prepare_prompt_response(entry) for entry in data_fin]

# # Convert into Document format
# documents = [Document(page_content=f"Prompt: {prompt}\nResponse: {response}") for prompt, response in prompt_response_data]

# # Split documents using Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# # Load Hugging Face embedding model
# hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Directory for storing vector database
# persist_directory = 'chroma_db'
# vectordb_fin = Chroma.from_documents(
#     documents=texts,
#     embedding=hg_embeddings,
#     persist_directory=persist_directory
# )

# # Initialize the Hugging Face LLM
# HUGGINGFACE_API_KEY = "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"  # Replace with your actual API key

# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", 
#                      model_kwargs={"temperature": 0.7}, 
#                      huggingfacehub_api_token=HUGGINGFACE_API_KEY)

# # Set up the RetrievalQA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectordb_fin.as_retriever(),
#     return_source_documents=False
# )

# # User input
# st.header("Get Investment Options")
# age = st.number_input("Enter your age:", min_value=18, max_value=100, value=30)
# gender = st.selectbox("Select your gender:", ("Male", "Female", "Other"))
# avenue = st.text_input("What avenue are you considering? (e.g., Mutual Funds, Equity Market)")
# purpose = st.text_input("What is your investment purpose? (e.g., Wealth Creation, Retirement)")
# duration = st.text_input("What is the duration of your investment? (e.g., 1-3 years, 5 years)")

# if st.button("Get Options"):
#     # Create query based on user input
#     query = f"I'm a {age}-year-old {gender} looking to invest in {avenue} for {purpose} over the next {duration}. What are my options?"
    
#     # Get result from the QA chain
#     result = qa({"query": query})
    
#     # Display the result
#     st.subheader("Suggested Investment Options:")
#     st.write(result['result'])




# import streamlit as st
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Set the page title
# st.title("Investment Options Advisor")

# # Load the dataset
# @st.cache_resource
# def load_data():
#     data = pd.read_csv("Finance_data.csv")
#     return data.to_dict(orient='records')

# data_fin = load_data()

# # Prepare prompt-response format
# def prepare_prompt_response(entry):
#     prompt = f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#     response = (
#         f"Based on your preferences, here are your investment options:\n"
#         f"- Mutual Funds: {entry['Mutual_Funds']}\n"
#         f"- Equity Market: {entry['Equity_Market']}\n"
#         f"- Debentures: {entry['Debentures']}\n"
#         f"- Government Bonds: {entry['Government_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
#         f"- PPF: {entry['PPF']}\n"
#         f"- Gold: {entry['Gold']}\n"
#         f"Factors considered: {entry['Factor']}\n"
#         f"Objective: {entry['Objective']}\n"
#         f"Expected returns: {entry['Expect']}\n"
#         f"Investment monitoring: {entry['Invest_Monitor']}\n"
#         f"Reasons for choices:\n"
#         f"- Equity: {entry['Reason_Equity']}\n"
#         f"- Mutual Funds: {entry['Reason_Mutual']}\n"
#         f"- Bonds: {entry['Reason_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Reason_FD']}\n"
#         f"Source of information: {entry['Source']}\n"
#     )
#     return prompt, response

# # Create prompts and responses for each entry in the dataset
# prompt_response_data = [prepare_prompt_response(entry) for entry in data_fin]

# # Convert into Document format
# documents = [Document(page_content=f"Prompt: {prompt}\nResponse: {response}") for prompt, response in prompt_response_data]

# # Split documents using Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# # Load Hugging Face embedding model
# hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Directory for storing vector database
# persist_directory = 'chroma_db'
# vectordb_fin = Chroma.from_documents(
#     documents=texts,
#     embedding=hg_embeddings,
#     persist_directory=persist_directory
# )

# # Initialize the Hugging Face LLM
# HUGGINGFACE_API_KEY = "YOUR_HUGGINGFACE_API_KEY"  # Replace with your actual API key

# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", 
#                      model_kwargs={"temperature": 0.7}, 
#                      huggingfacehub_api_token=HUGGINGFACE_API_KEY)

# # Set up the RetrievalQA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectordb_fin.as_retriever(),
#     return_source_documents=False
# )

# # User input
# st.header("Get Investment Options")
# age = st.number_input("Enter your age:", min_value=18, max_value=100, value=30)
# gender = st.selectbox("Select your gender:", ("Male", "Female", "Other"))
# avenue = st.text_input("What avenue are you considering? (e.g., Mutual Funds, Equity Market)")
# purpose = st.text_input("What is your investment purpose? (e.g., Wealth Creation, Retirement)")
# duration = st.text_input("What is the duration of your investment? (e.g., 1-3 years, 5 years)")

# if st.button("Get Options"):
#     # Validate user input
#     if avenue and purpose and duration:
#         # Create query based on user input
#         query = f"I'm a {age}-year-old {gender} looking to invest in {avenue} for {purpose} over the next {duration}. What are my options?"

#         # Get result from the QA chain
#         result = qa({"query": query})

#         # Display the result
#         st.subheader("Suggested Investment Options:")
#         st.write(result['result'])
#     else:
#         st.warning("Please fill in all the fields.")








# import streamlit as st
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Set the page title
# st.title("Investment Options Advisor")

# # Load the dataset
# @st.cache_resource
# def load_data():
#     data = pd.read_csv("Finance_data.csv")
#     return data.to_dict(orient='records')

# data_fin = load_data()

# # Prepare prompt-response format
# def prepare_prompt_response(entry):
#     prompt = f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#     response = (
#         f"Based on your preferences, here are your investment options:\n"
#         f"- Mutual Funds: {entry['Mutual_Funds']}\n"
#         f"- Equity Market: {entry['Equity_Market']}\n"
#         f"- Debentures: {entry['Debentures']}\n"
#         f"- Government Bonds: {entry['Government_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
#         f"- PPF: {entry['PPF']}\n"
#         f"- Gold: {entry['Gold']}\n"
#         f"Factors considered: {entry['Factor']}\n"
#         f"Objective: {entry['Objective']}\n"
#         f"Expected returns: {entry['Expect']}\n"
#         f"Investment monitoring: {entry['Invest_Monitor']}\n"
#         f"Reasons for choices:\n"
#         f"- Equity: {entry['Reason_Equity']}\n"
#         f"- Mutual Funds: {entry['Reason_Mutual']}\n"
#         f"- Bonds: {entry['Reason_Bonds']}\n"
#         f"- Fixed Deposits: {entry['Reason_FD']}\n"
#         f"Source of information: {entry['Source']}\n"
#     )
#     return prompt, response

# # Create prompts and responses for each entry in the dataset
# prompt_response_data = [prepare_prompt_response(entry) for entry in data_fin]

# # Convert into Document format
# documents = [Document(page_content=f"Prompt: {prompt}\nResponse: {response}") for prompt, response in prompt_response_data]

# # Split documents using Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# # Load Hugging Face embedding model
# hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Directory for storing vector database
# persist_directory = 'chroma_db'
# vectordb_fin = Chroma.from_documents(
#     documents=texts,
#     embedding=hg_embeddings,
#     persist_directory=persist_directory
# )

# # Replace this with your actual Hugging Face API key
# HUGGINGFACE_API_KEY = "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"

# # Initialize the Hugging Face LLM
# llm = HuggingFaceHub(
#     repo_id="tiiuae/falcon-7b",
#     model_kwargs={"temperature": 0.7},
#     huggingfacehub_api_token=HUGGINGFACE_API_KEY
# )

# # Set up the RetrievalQA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectordb_fin.as_retriever(),
#     return_source_documents=False
# )

# # User input
# st.header("Get Investment Options")
# age = st.number_input("Enter your age:", min_value=18, max_value=100, value=30)
# gender = st.selectbox("Select your gender:", ("Male", "Female", "Other"))
# avenue = st.text_input("What avenue are you considering? (e.g., Mutual Funds, Equity Market)")
# purpose = st.text_input("What is your investment purpose? (e.g., Wealth Creation, Retirement)")
# duration = st.text_input("What is the duration of your investment? (e.g., 1-3 years, 5 years)")

# if st.button("Get Options"):
#     # Create query based on user input
#     query = f"I'm a {age}-year-old {gender} looking to invest in {avenue} for {purpose} over the next {duration}. What are my options?"
    
#     # Get result from the QA chain
#     try:
#         result = qa({"query": query})
#         # Display the result
#         st.subheader("Suggested Investment Options:")
#         st.write(result['result'])
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}. Please check your API key and try again.")











# import streamlit as st
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Set the page title
# st.title("Investment Options Advisor")

# # Load the dataset
# @st.cache_resource
# def load_data():
#     data = pd.read_csv("Finance_data.csv")
#     return data.to_dict(orient='records')

# data_fin = load_data()

# # Prepare prompt-response format
# def prepare_prompt_response(entry):
#     prompt = (
#         f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} "
#         f"for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#     )
#     response = (
#         f"Based on your preferences, here are your tailored investment options:\n\n"
#         f"- **Mutual Funds**: {entry['Mutual_Funds']}\n"
#         f"- **Equity Market**: {entry['Equity_Market']}\n"
#         f"- **Debentures**: {entry['Debentures']}\n"
#         f"- **Government Bonds**: {entry['Government_Bonds']}\n"
#         f"- **Fixed Deposits**: {entry['Fixed_Deposits']}\n"
#         f"- **PPF**: {entry['PPF']}\n"
#         f"- **Gold**: {entry['Gold']}\n\n"
#         f"**Factors considered**: {entry['Factor']}\n"
#         f"**Objective**: {entry['Objective']}\n"
#         f"**Expected returns**: {entry['Expect']}\n"
#         f"**Investment monitoring**: {entry['Invest_Monitor']}\n\n"
#         f"**Reasons for choices**:\n"
#         f"- **Equity**: {entry['Reason_Equity']}\n"
#         f"- **Mutual Funds**: {entry['Reason_Mutual']}\n"
#         f"- **Bonds**: {entry['Reason_Bonds']}\n"
#         f"- **Fixed Deposits**: {entry['Reason_FD']}\n"
#         f"**Source of information**: {entry['Source']}\n"
#     )
#     return prompt, response

# # Generate prompts and responses
# prompt_response_data = [prepare_prompt_response(entry) for entry in data_fin]

# # Convert into Document format
# documents = [Document(page_content=f"Prompt: {prompt}\nResponse: {response}") for prompt, response in prompt_response_data]

# # Split documents using Text Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# # Load Hugging Face embedding model
# hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # Directory for storing vector database
# persist_directory = 'chroma_db'
# vectordb_fin = Chroma.from_documents(
#     documents=texts,
#     embedding=hg_embeddings,
#     persist_directory=persist_directory
# )

# # Initialize the Hugging Face LLM
# HUGGINGFACE_API_KEY = "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"  # Replace with your actual API key
# llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", 
#                      model_kwargs={"temperature": 0.7}, 
#                      huggingfacehub_api_token=HUGGINGFACE_API_KEY)

# # Set up the RetrievalQA chain
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectordb_fin.as_retriever(),
#     return_source_documents=False
# )

# # Single prompt input
# st.header("Get Investment Options")
# user_prompt = st.text_input("Enter your investment query :")

# if st.button("Get Options") and user_prompt:
#     # Pass the user prompt directly into the QA chain
#     result = qa({"query": user_prompt})
    
#     # Display the result
#     st.subheader("Suggested Investment Options:")
#     st.write(result)






# Install all required libraries
# Uncomment the below lines if you need to install the packages
# !pip install -q langchain langchain-community langchain-core transformers langchain-text-splitters
# !pip install -qU sentence-transformers chromadb bitsandbytes

# # Import necessary libraries
# import streamlit as st
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Load and prepare data
# @st.cache_data
# def load_data():
#     data = pd.read_csv("Finance_data.csv")
#     data_fin = data.to_dict(orient='records')

#     prompt_response_data = []
#     for entry in data_fin:
#         prompt = f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#         response = (
#             f"Based on your preferences, here are your investment options:\n"
#             f"- Mutual Funds: {entry['Mutual_Funds']}\n"
#             f"- Equity Market: {entry['Equity_Market']}\n"
#             f"- Debentures: {entry['Debentures']}\n"
#             f"- Government Bonds: {entry['Government_Bonds']}\n"
#             f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
#             f"- PPF: {entry['PPF']}\n"
#             f"- Gold: {entry['Gold']}\n"
#             f"Factors considered: {entry['Factor']}\n"
#             f"Objective: {entry['Objective']}\n"
#             f"Expected returns: {entry['Expect']}\n"
#             f"Investment monitoring: {entry['Invest_Monitor']}\n"
#             f"Reasons for choices:\n"
#             f"- Equity: {entry['Reason_Equity']}\n"
#             f"- Mutual Funds: {entry['Reason_Mutual']}\n"
#             f"- Bonds: {entry['Reason_Bonds']}\n"
#             f"- Fixed Deposits: {entry['Reason_FD']}\n"
#             f"Source of information: {entry['Source']}\n"
#         )
#         prompt_response_data.append({"prompt": prompt, "response": response})

#     return prompt_response_data

# # Prepare the vector store
# def create_vector_store(prompt_response_data):
#     # Convert into Document format
#     documents = []
#     for entry in prompt_response_data:
#         combined_text = f"Prompt: {entry['prompt']}\nResponse: {entry['response']}"
#         documents.append(Document(page_content=combined_text))

#     # Split documents using Text Splitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)

#     # Load Hugging Face embedding model
#     hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#     # Directory for storing vector database
#     persist_directory = 'chroma_db'
#     vectordb_fin = Chroma.from_documents(
#         documents=texts,
#         embedding=hg_embeddings,
#         persist_directory=persist_directory
#     )

#     return vectordb_fin

# # Initialize the Hugging Face LLM
# def create_qa_chain(vectordb_fin):
#     HUGGINGFACE_API_KEY = "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"
#     llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", 
#                          model_kwargs={"temperature": 0.7}, 
#                          huggingfacehub_api_token=HUGGINGFACE_API_KEY)

#     # Set up the RetrievalQA chain
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectordb_fin.as_retriever(),  # Ensure this is correctly fetching relevant context
#         return_source_documents=False
#     )
#     return qa

# # Streamlit app layout
# st.title("Investment Options Advisor")

# # Load data
# prompt_response_data = load_data()
# vectordb_fin = create_vector_store(prompt_response_data)
# qa_chain = create_qa_chain(vectordb_fin)

# # Input text box for user prompt
# user_input = st.text_input("Enter your investment query:")

# # Check if the user has entered a prompt
# if user_input:
#     result = qa_chain({"query": user_input})
#     st.write("**Helpful Answer:**")
#     st.write(result['result'])




# import streamlit as st
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load pre-trained model and tokenizer
# model_name = "gpt2"  # You can also try "gpt2-medium" or others
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # Function to generate financial advice
# def generate_financial_advice(prompt, max_length=100):
#     # Encode the input prompt
#     inputs = tokenizer.encode(prompt, return_tensors="pt")

#     # Generate a response
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_length=max_length,
#             num_return_sequences=1,
#             no_repeat_ngram_size=2,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     # Decode the generated response
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Streamlit app layout
# st.title("AI Financial Advisor")
# st.write("Ask any investment-related question, and get advice from our AI model!")

# # User input
# user_input = st.text_input("Your Investment Question:", "What investment strategy should I consider for long-term growth?")

# # Button to generate advice
# if st.button("Get Advice"):
#     if user_input:
#         advice = generate_financial_advice(user_input)
#         st.write("### AI Financial Advisor Response:")
#         st.write(advice)
#     else:
#         st.write("Please enter a question to get advice.")






# Install all required libraries
# Uncomment the below lines if you need to install the packages
# !pip install -q langchain langchain-community langchain-core transformers langchain-text-splitters
# !pip install -qU sentence-transformers chromadb bitsandbytes

# Import necessary libraries
# import streamlit as st
# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import HuggingFaceHub
# from langchain.chains import RetrievalQA

# # Load and prepare data
# @st.cache_data
# def load_data():
#     data = pd.read_csv("Finance_data.csv")
#     data_fin = data.to_dict(orient='records')

#     prompt_response_data = []
#     for entry in data_fin:
#         prompt = f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
#         response = (
#             f"Based on your preferences, here are your investment options:\n"
#             f"- Mutual Funds: {entry['Mutual_Funds']}\n"
#             f"- Equity Market: {entry['Equity_Market']}\n"
#             f"- Debentures: {entry['Debentures']}\n"
#             f"- Government Bonds: {entry['Government_Bonds']}\n"
#             f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
#             f"- PPF: {entry['PPF']}\n"
#             f"- Gold: {entry['Gold']}\n"
#             f"Factors considered: {entry['Factor']}\n"
#             f"Objective: {entry['Objective']}\n"
#             f"Expected returns: {entry['Expect']}\n"
#             f"Investment monitoring: {entry['Invest_Monitor']}\n"
#             f"Reasons for choices:\n"
#             f"- Equity: {entry['Reason_Equity']}\n"
#             f"- Mutual Funds: {entry['Reason_Mutual']}\n"
#             f"- Bonds: {entry['Reason_Bonds']}\n"
#             f"- Fixed Deposits: {entry['Reason_FD']}\n"
#             f"Source of information: {entry['Source']}\n"
#         )
#         prompt_response_data.append({"prompt": prompt, "response": response})

#     return prompt_response_data

# # Prepare the vector store
# def create_vector_store(prompt_response_data):
#     # Convert into Document format
#     documents = []
#     for entry in prompt_response_data:
#         combined_text = f"Prompt: {entry['prompt']}\nResponse: {entry['response']}"
#         documents.append(Document(page_content=combined_text))

#     # Split documents using Text Splitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)

#     # Load Hugging Face embedding model
#     hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#     # Directory for storing vector database
#     persist_directory = 'chroma_db'
#     vectordb_fin = Chroma.from_documents(
#         documents=texts,
#         embedding=hg_embeddings,
#         persist_directory=persist_directory
#     )

#     return vectordb_fin

# # Initialize the Hugging Face LLM
# def create_qa_chain(vectordb_fin):
#     HUGGINGFACE_API_KEY = "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"
#     llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b", 
#                          model_kwargs={"temperature": 0.7}, 
#                          huggingfacehub_api_token=HUGGINGFACE_API_KEY)

#     # Set up the RetrievalQA chain
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectordb_fin.as_retriever(),  # Ensure this is correctly fetching relevant context
#         return_source_documents=False
#     )
#     return qa

# # Streamlit app layout
# st.title("Investment Options Advisor")

# # Load data
# prompt_response_data = load_data()
# vectordb_fin = create_vector_store(prompt_response_data)
# qa_chain = create_qa_chain(vectordb_fin)

# # Input text box for user prompt
# user_input = st.text_input("Enter your investment query:")

# # Check if the user has entered a prompt
# if user_input:
#     result = qa_chain({"query": user_input})
    
#     # Extract response after the separator "Helpful Answer:"
#     separator = 'Helpful Answer:'
#     before, sep, after = result['result'].rpartition(separator)
    
#     # Display response
#     st.write("**Helpful Answer:**")
#     st.write(after.strip())  # Strip to remove any leading/trailing whitespace


















import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Set the page title
st.title("Investment Options Advisor")

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv("Finance_data.csv")
    return data.to_dict(orient='records')

data_fin = load_data()

# Prepare prompt-response format for each entry in the dataset
def prepare_prompt_response(entry):
    prompt = f"I'm a {entry['age']}-year-old {entry['gender']} looking to invest in {entry['Avenue']} for {entry['Purpose']} over the next {entry['Duration']}. What are my options?"
    response = (
        f"Based on your preferences, here are your investment options:\n"
        f"- Mutual Funds: {entry['Mutual_Funds']}\n"
        f"- Equity Market: {entry['Equity_Market']}\n"
        f"- Debentures: {entry['Debentures']}\n"
        f"- Government Bonds: {entry['Government_Bonds']}\n"
        f"- Fixed Deposits: {entry['Fixed_Deposits']}\n"
        f"- PPF: {entry['PPF']}\n"
        f"- Gold: {entry['Gold']}\n"
        f"Factors considered: {entry['Factor']}\n"
        f"Objective: {entry['Objective']}\n"
        f"Expected returns: {entry['Expect']}\n"
        f"Investment monitoring: {entry['Invest_Monitor']}\n"
        f"Reasons for choices:\n"
        f"- Equity: {entry['Reason_Equity']}\n"
        f"- Mutual Funds: {entry['Reason_Mutual']}\n"
        f"- Bonds: {entry['Reason_Bonds']}\n"
        f"- Fixed Deposits: {entry['Reason_FD']}\n"
        f"Source of information: {entry['Source']}\n"
    )
    return prompt, response

# Convert each entry into a Document format for use with Chroma
documents = [Document(page_content=f"Prompt: {prepare_prompt_response(entry)[0]}\nResponse: {prepare_prompt_response(entry)[1]}") for entry in data_fin]

# Split documents using Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Load Hugging Face embedding model
hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Directory for storing vector database
persist_directory = 'chroma_db'
vectordb_fin = Chroma.from_documents(
    documents=texts,
    embedding=hg_embeddings,
    persist_directory=persist_directory
)

# Initialize the Hugging Face LLM
HUGGINGFACE_API_KEY = "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"  # Replace with your actual API key

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b", 
    model_kwargs={"temperature": 0.7}, 
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
)

# Set up the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb_fin.as_retriever(),
    return_source_documents=False
)

# User input section in Streamlit
st.header("Get Investment Options Based on Profile")
age = st.number_input("Enter your age:", min_value=18, max_value=100, value=30)
gender = st.selectbox("Select your gender:", ("Male", "Female", "Other"))
avenue = st.text_input("What avenue are you considering? (e.g., Mutual Funds, Equity Market)")
purpose = st.text_input("What is your investment purpose? (e.g., Wealth Creation, Retirement)")
duration = st.text_input("What is the duration of your investment? (e.g., 1-3 years, 5 years)")

if st.button("Get Advice"):
    # Create a query prompt based on the user input
    query = f"I'm a {age}-year-old {gender} looking to invest in {avenue} for {purpose} over the next {duration}. What are my options?"
    
    # Retrieve the result based on the input query using the QA chain
    result = qa({"query": query})
    
    # Display the result in Streamlit
    st.subheader("Suggested Investment Options:")
    st.write(result['result'])