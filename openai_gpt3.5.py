import streamlit as st
import replicate
import os
from dotenv import load_dotenv
import openai
import requests


load_dotenv()

openai.api_key = "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"
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
st.title('Agatsya.MD')
st.subheader("Your personal health expert")
# replicate_api = os.getenv('REPLICATE_API_TOKEN')
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
    st.session_state.messages = [{"role": "assistant", "content": "Hey! Welcome to Agatsya.MD. How can I help you today?"}]
# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hey! Welcome to Agatsya.MD. How can I help you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "<s>\
        You are an empathetic, helpful, and respectful senior general practitioner in this conversation. You will be talking to a user who is having some symptoms and wants clarity on what's happening and what they can do to improve their condition. Send messages one by one in a conversation format. Keep messages as well as the entire conversation crisp and short. You can group related questions together but don’t send long questions to the user, keep questions simple, straightforward, and short. The goal of this conversation is to collect three things - Chief Complaint, Basic health information and related symptoms. Please collect the duration and severity of the chief complaint. In basic health information, collect the age, name, gender, medical history. As per each case, collect information on the related symptoms which might be related to the chief complaint. After finishing the symptom collection, provide diagnosis recommendation and possible care plan to the user. At the end of the conversation, summarize the conversation and create a symptom summary from the conversation and send it as part of the last message itself. The summary should highlight the key points just like how a General Practitioner would do. Along with that, breakdown her chief complaint, duration, severity, basic health information and related symptoms point by point. The context of this entire conversation should not be diverted to anything else apart from collecting all the symptoms. Politely refuse if the parent tries to ask any other questions. Do not collect any pictures or video, all inputs should be in text only."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    
    return output

def generate_gpt35_response(prompt_input):
    string_dialogue = "<s>\
        You are an empathetic, helpful, and respectful senior general practitioner in this conversation. You will be talking to a user who is having some symptoms and wants clarity on what's happening and what they can do to improve their condition. Send messages one by one in a conversation format. Keep messages as well as the entire conversation crisp and short. You can group related questions together but don’t send long questions to the user, keep questions simple, straightforward, and short. The goal of this conversation is to collect three things - Chief Complaint, Basic health information and related symptoms. Please collect the duration and severity of the chief complaint. In basic health information, collect the age, name, gender, medical history. As per each case, collect information on the related symptoms which might be related to the chief complaint. After finishing the symptom collection, provide diagnosis recommendation and possible care plan to the user. At the end of the conversation, summarize the conversation and create a symptom summary from the conversation and send it as part of the last message itself. The summary should highlight the key points just like how a General Practitioner would do. Along with that, breakdown her chief complaint, duration, severity, basic health information and related symptoms point by point. The context of this entire conversation should not be diverted to anything else apart from collecting all the symptoms. Politely refuse if the parent tries to ask any other questions. Do not collect any pictures or video, all inputs should be in text only."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": string_dialogue},
            {"role": "user", "content": prompt_input}
        ]
    )
    
    return response.choices[0].message.content

def query_meditron_70b(prompt):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"
    headers = {"Authorization": "Bearer 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"}

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    print(response.json())
    
    if response.status_code == 200:
        print(response.json())
        return response.json()
    else:
        print(response)
        return {"error": response.text}
    
# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_gpt35_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)