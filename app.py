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


import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.agents import create_pandas_dataframe_agent

# Set Streamlit page config
st.set_page_config(page_title="Local Mistral Hardware AI Agent", layout="centered")
st.title("🧠 Local Mistral Hardware Management Agent")
st.markdown("Upload your Excel hardware inventory and ask any question about it.")

# Load Mistral model (make sure you have enough RAM/GPU)
@st.cache_resource
def load_mistral_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"  # Auto-detect GPU or CPU
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)

# Load model only once
llm = load_mistral_model()

# Upload Excel File
uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        # Create LangChain agent with mistral
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )

        query = st.text_input("💬 Ask a question about your hardware data:")
        if query:
            with st.spinner("Thinking..."):
                response = agent.run(query)
                st.write("🤖", response)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
