# import streamlit as st
# import pandas as pd
# from langchain.llms import HuggingFacePipeline
# from langchain.agents import create_pandas_dataframe_agent
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # Load GPT-2 locally
# checkpoint = "gpt2"  # You can replace this with a better local model
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
# llm = HuggingFacePipeline(pipeline=pipe)

# # --- Streamlit UI ---
# st.set_page_config(page_title="Local Hardware AI Agent", layout="centered")
# st.title("🧠 Local GPT-2 Hardware Management Agent")
# st.markdown("Upload your hardware Excel and ask questions — runs fully offline!")

# uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

# if uploaded_file:
#     try:
#         df = pd.read_excel(uploaded_file)
#         st.success("✅ File uploaded successfully!")
#         st.dataframe(df.head())

#         # Use LangChain with local GPT-2
#         agent = create_pandas_dataframe_agent(llm, df, verbose=False)

#         query = st.text_input("💬 Ask a question about your hardware data:")
#         if query:
#             with st.spinner("Thinking..."):
#                 response = agent.run(query)
#                 st.write("🤖", response)

#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")


# import streamlit as st
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain_experimental.agents import create_pandas_dataframe_agent

# # Load GPT-2 locally
# #checkpoint = "gpt2"
# checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
# llm = HuggingFacePipeline(pipeline=pipe)

# # Streamlit UI
# st.set_page_config(page_title="Local Hardware AI Agent", layout="centered")
# st.title("🧠 Local GPT-2 Hardware Management Agent")
# st.markdown("Upload your Excel hardware sheet and ask questions.")

# uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])
# if uploaded_file:
#     try:
#         df = pd.read_excel(uploaded_file)
#         st.success("✅ File uploaded successfully!")
#         st.dataframe(df.head())

#         # Use LangChain with GPT-2
#         agent = create_pandas_dataframe_agent(llm, df, verbose=False, allow_dangerous_code=True,
#     handle_parsing_errors=True)

#         query = st.text_input("💬 Ask a question about your hardware data:")
#         if query:
#             with st.spinner("Thinking..."):
#                 response = agent.run(query)
#                 st.write("🤖", response)

#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")


# import streamlit as st
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain_experimental.agents import create_pandas_dataframe_agent

# # Set Streamlit page config
# st.set_page_config(page_title="Local Mistral Hardware AI Agent", layout="centered")
# st.title("🧠 Local Mistral Hardware Management Agent")
# st.markdown("Upload your Excel hardware inventory and ask any question about it.")

# # Load Mistral model (make sure you have enough RAM/GPU)
# @st.cache_resource
# def load_mistral_model():
#     model_id = "mistralai/Mistral-7B-Instruct-v0.2"

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map="auto"  # Auto-detect GPU or CPU
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=256,
#         temperature=0,
#         repetition_penalty=1.1
#     )

#     return HuggingFacePipeline(pipeline=pipe)

# # Load model only once
# llm = load_mistral_model()

# # Upload Excel File
# uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

# if uploaded_file:
#     try:
#         df = pd.read_excel(uploaded_file)
#         st.success("✅ File uploaded successfully!")
#         st.dataframe(df.head())

#         # Create LangChain agent with mistral
#         agent = create_pandas_dataframe_agent(
#             llm,
#             df,
#             verbose=False,
#             allow_dangerous_code=True,
#             handle_parsing_errors=True
#         )

#         query = st.text_input("💬 Ask a question about your hardware data:")
#         if query:
#             with st.spinner("Thinking..."):
#                 response = agent.run(query)
#                 st.write("🤖", response)

#     except Exception as e:
#         st.error(f"❌ Error processing file: {e}")



# import streamlit as st
# import pandas as pd

# from langchain_community.llms import Ollama
# from langchain_experimental.agents import create_pandas_dataframe_agent

# # --- Streamlit UI ---
# st.set_page_config(layout="centered")
# st.title("🤖 Local GPT-2-like Hardware Agent (Mistral via Ollama)")
# st.markdown("Upload your Excel hardware sheet and ask questions about it.")

# # --- File upload ---
# uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])
# if uploaded_file:
#     try:
#         df = pd.read_excel(uploaded_file)
#         st.success("✅ File uploaded successfully!")
#         st.dataframe(df)

#         # Load Mistral model via Ollama
#         llm = Ollama(model="mistral")

#         # Create agent
#         agent = create_pandas_dataframe_agent(
#             llm,
#             df,
#             verbose=True,
#             handle_parsing_errors=True,
#             allow_dangerous_code=True,
#             agent_type="openai-tools",  # robust with tabular data
#         )

#         # User question
#         user_query = st.text_input("💬 Ask a question about your hardware data:")
#         if user_query:
#             with st.spinner("🧠 Thinking..."):
#                 try:
#                     answer = agent.run(user_query)
#                     st.success("📌 Answer:")
#                     st.write(answer)
#                 except Exception as e:
#                     st.error(f"❌ Error processing question: {e}")

#     except Exception as e:
#         st.error(f"❌ Error reading Excel file: {e}")


import streamlit as st
import pandas as pd
from langchain_ollama import Ollama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# -- Streamlit UI Setup --
st.set_page_config(page_title="Local GPT Hardware Agent", layout="centered")
st.title("🧠 Local GPT Hardware Management Agent")
st.markdown("Upload your Excel hardware sheet and ask questions.")

# -- File Upload --
uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

# -- User Question Input --
user_query = st.text_input("💬 Ask a question about your hardware data:")

# -- If File is Uploaded --
if uploaded_file:
    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        # Initialize Ollama with Mistral
        llm = Ollama(model="mistral")

        # Create LangChain Agent
        agent = create_pandas_dataframe_agent(llm, df, verbose=True)

        # Handle query
        if user_query:
            with st.spinner("Thinking..."):
                response = agent.invoke(user_query)
                st.success("✅ Answer:")
                st.write(response)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("Please upload an Excel file to get started.")


