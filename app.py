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


import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load GPT-2 locally
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

# Streamlit UI
st.set_page_config(page_title="Local Hardware AI Agent", layout="centered")
st.title("🧠 Local GPT-2 Hardware Management Agent")
st.markdown("Upload your Excel hardware sheet and ask questions.")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        # Use LangChain with GPT-2
        agent = create_pandas_dataframe_agent(llm, df, verbose=False)

        query = st.text_input("💬 Ask a question about your hardware data:")
        if query:
            with st.spinner("Thinking..."):
                response = agent.run(query)
                st.write("🤖", response)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

