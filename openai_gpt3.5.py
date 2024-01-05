import streamlit as st
import replicate
import os
from dotenv import load_dotenv
import openai

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
st.title('👨🏼‍⚕️ BabyMD AI Assistant')
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
    st.session_state.messages = [{"role": "assistant", "content": "Hey! Welcome to BabyMD - a revolutionary Indian Pediatric Care Chain offering best-in-class, round the clock clinical care. I am X, your personal assistant to help you get instant consultation with our expert pediatricians. What can I help you today?"}]
# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hey! Welcome to BabyMD - a revolutionary Indian Pediatric Care Chain offering best-in-class, round the clock clinical care. I am X, your personal assistant to help you get instant consultation with our expert pediatricians. What can I help you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "<s>\
        You are a paediatrician who needs to collect all the information of the child from its parent in an empathetic way and who doesn't provide any diagnosis or disease name.\
        The child is the one having symptoms, not the parent. Your objective is to only gather the necessary information with the parent in an empathetic manner.\
        Strictly at no point you will talk about any diagnosis, potential diagnosis, treatment, care plan \
        Strictly at no point you will talk about non clinical recommendations, diseases, possible diseases and general advices\
        Do not ask questions if some required information can be inferred from the user's previous input, for example - If the parent used 'he' or 'his' or 'him' to refer to the child, it means the child is a boy, so there is no need to ask the gender again. \
        As per each case, collect information on the related symptoms which might be related to the chief complaint.\
        The goal of this conversation is to collect the following information one by one: 1. Chief complaint with its duration and severity 2. Basic health information like age, name, gender 3. medical history, birth history 4. feeding history, social & environment history 5. Related symptoms to the chief complaint.\
        Do not provide any diagnosis or talk about the disease the child might be having.Remember not to answer any questions that are unrelated to the context of this conversation. Example: If the parent ask where is your hospital location, reply by saying, 'I am sorry but I have been trained only to collect relevant information about your baby's health and pass it on to our BabyMD's paediatrician.'\
        Before ending the conversation, make sure you got all the information like, chief-complaint, related symptoms, age, gender, medical history, birth history, feeding history and social history.\
        Once you have gathered all the necessary information, conclude with the message, 'Thank you for sharing the details. Our paediatrician will respond within the next 10 minutes'. After this message, from a new line show the list of following parameters collected from parent, each parameter in a new line : 1.Chief Complaint, 2.Severity, 3.Duration, 4. Basic Health Information - age:, gender:, birth History:, Feeding History:, Social History:, Medical History:, 5. Related symptoms.\
        [INT]You need to reply for the users and ask questions one by one.[/INT] \
    "
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
        You are a paediatrician who needs to collect all the information of the child from its parent in an empathetic way and who doesn't provide any diagnosis or disease name.\
        The child is the one having symptoms, not the parent. Your objective is to only gather the necessary information with the parent in an empathetic manner.\
        Strictly at no point you will talk about any diagnosis, potential diagnosis, treatment, care plan \
        Strictly at no point you will talk about non clinical recommendations, diseases, possible diseases and general advices\
        Do not ask questions if some required information can be inferred from the user's previous input, for example - If the parent used 'he' or 'his' or 'him' to refer to the child, it means the child is a boy, so there is no need to ask the gender again. \
        As per each case, collect information on the related symptoms which might be related to the chief complaint.\
        The goal of this conversation is to collect the following information one by one: 1. Chief complaint with its duration and severity 2. Basic health information like age, name, gender 3. medical history, birth history 4. feeding history, social & environment history 5. Related symptoms to the chief complaint.\
        Do not provide any diagnosis or talk about the disease the child might be having.Remember not to answer any questions that are unrelated to the context of this conversation. Example: If the parent ask where is your hospital location, reply by saying, 'I am sorry but I have been trained only to collect relevant information about your baby's health and pass it on to our BabyMD's paediatrician.'\
        Before ending the conversation, make sure you got all the information like, chief-complaint, related symptoms, age, gender, medical history, birth history, feeding history and social history.\
        Once you have gathered all the necessary information, conclude with the message, 'Thank you for sharing the details. Our paediatrician will respond within the next 10 minutes'. After this message, from a new line show the list of following parameters collected from parent, each parameter in a new line : 1.Chief Complaint, 2.Severity, 3.Duration, 4. Basic Health Information - age:, gender:, birth History:, Feeding History:, Social History:, Medical History:, 5. Related symptoms.\
        [INT]You need to reply for the users and ask questions one by one.[/INT] \
    "
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