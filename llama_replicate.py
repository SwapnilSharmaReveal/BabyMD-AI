import streamlit as st
import replicate
import os
from langchain.chains import LLMChain
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
import boto3
import json
import io
from LineIterator import LineIterator
from dotenv import load_dotenv

load_dotenv()

boto3_session=boto3.session.Session(region_name="us-east-1")
smr = boto3.client('sagemaker-runtime')
endpoint_name = os.getenv("endpoint_name", default=None)
stop_token = '<|endoftext|>'


# App title
st.set_page_config(page_title="BabyMD AI Assistant", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    [data-testid="manage-app-button"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# Replicate Credentials
st.title('👨🏼‍⚕️ BabyMD AI Assistant')
replicate_api = os.getenv('REPLICATE_API_TOKEN')
# with st.sidebar:

#     # st.subheader('Models and parameters')
#     # selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
#     # if selected_model == 'Llama2-7B':
#     #     llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
#     # elif selected_model == 'Llama2-13B':
#     #     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'

#     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        
#     # temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
#     # top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
#     # max_length = st.sidebar.slider('max_length', min_value=32, max_value=4096, value=120, step=8)
#     temperature = 0.1
#     top_p = 0.9
#     max_length = 4096

llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
temperature = 0.1
top_p = 0.9
max_length = 4096

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stToolbar { visibility: hidden; }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Display "Print Action Button" in all environment.
show_print_btn_css = """
<style>
    #MainMenu {visibility: visible;}
    [data-testid="main-menu-popover"] [data-testid="main-menu-list"] ul:not(:nth-of-type(3)) , [data-testid="main-menu-popover"] [data-testid="main-menu-divider"]{
        display: none;
    }
    [data-testid="main-menu-popover"] [data-testid="main-menu-list"]:nth-of-type(2){
        display: none;
    }
</style>
"""
st.markdown(show_print_btn_css, unsafe_allow_html=True)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your health assistant. How can I help you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your health assistant. How can I help you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def generate_llama2_summary(text):
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                            input={"prompt": f"""
                                    Your task is to Summarize the review below in the following factors, each on a separate line:
                                        - Problems
                                        - Age
                                        - Symptoms
                                        - Any medication taken yet,
                                        - Medical history
                                    delimited by triple 
                                    backticks, in atmost 30 words
                                    Review: ```{text}```
                                    """,
                                    "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    
    return output


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "<s>\
        As an [SYS], your role is to conduct a conversation with concern and in a professional manner while collecting information about a child's symptoms and basic information from their parents. Your objective is to gather the necessary information within 9 questions while avoiding any potential diagnosis or cause. Please handle any sensitive information shared by the parents with caution and ensure encryption for privacy and security. Inquire about the specific symptoms the child is experiencing and mandatorily collect the following information within two or three lines each: 1. Age and gender of the child, 2. Basic health information, 3. Chief complaint and duration, 4. Allergies, 5. Feeding history, 6. Existing medical conditions, 7. Medications being taken, 8. Birth history, 9. Recent social activities. Once you have gathered all the necessary information, conclude with the message, 'Thank you for sharing the details. Our pediatrician will respond within the next 10 minutes. Remember not to answer any questions that are unrelated to the medical context.\
        [INT]You need to reply for the users {input}[/INT]\
            "
    conversations = ''
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user" and dict_message["content"]=="bye":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
            conversations += "User: " + dict_message["content"] + "\n\n"
            response = generate_llama2_summary(string_dialogue)
            full_response = ''
            for item in response:
                full_response += item
        elif dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
            conversations += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            conversations += "Assistant: " + dict_message["content"] + "\n\n"
    prompt_template = PromptTemplate(input_variables=["input"],template=string_dialogue)
    chain = LLMChain(llm= Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"), prompt=prompt_template)
    # chain.run(prompt_input)

    body = {"inputs": f"{string_dialogue}s Assistant: ", "parameters": {"max_new_tokens":400, "return_full_text": False}, "stream": True}
    resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name, Body=json.dumps(body), ContentType="application/json")
    event_stream = resp['Body']
    return event_stream

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            event_stream = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for line in LineIterator(event_stream):
                if line != b'':
                    data = json.loads(line[5:].decode('utf-8'))['token']['text']
                    if data != stop_token:
                        full_response += data
                        placeholder.markdown(full_response)

            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)