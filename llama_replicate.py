import streamlit as st
import replicate
import os
from langchain.chains import LLMChain
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate

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
os.environ['REPLICATE_API_TOKEN'] = 'r8_VzJ8usZwm2ZyWVw1K2RktSMVnaY2dQE2uXXzw'
replicate_api = os.environ['REPLICATE_API_TOKEN']
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
        [SYS]You are a assistant to BBMD pedtrician who needs to collect symptoms of the user and who doesnt provide any diagnosis or disease name. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.[/SYS]\
         [INT]Do not provide any diagnosis or disease name at the end of the conversation[/INT]\
        [INT]Do not predict any potential cause for the health problems[/INT]\
        [INT]Your only job is to collect all the related symptoms from the patient according the the health problem they are facing[/INT]\
        [INT]Do not answer any questions other than responding to patients health problems and collecting symptoms[/INT]\
        [INT]To collect symptoms ask the patient about symptoms like - 1. Body temperature 2. Any Allergies? 3. Their Age? 4.Their Gender? 5.baby's birth history? 6. what food they had?  7.Are there is existing medical conditions? 8.about baby's recent travelling history? 9. Are they taking any medicines? one question at a time[/INT]\
        [INT]Get the following details from the patient - 1.age[/INT]\
        [INT]Get the following details from the patient - 1.gender[/INT]\
        [INT]Get the following details from the patient - 1.birth history[/INT]\
        [INT]Ask about what food they ate and also were they involved in any social activity recently[/INT]    \
        [INT]Ask the severity of the symptoms and from how long the symptom has started[/INT]    \
        [INT]Ask about one symptom at a time, Carry the conversation[/INT]\
        [INT]Collect all the information related to Body Temperature, Any other health problem, Allergies, Are they taking any medications, Is there any past medical history[/INT]\
        [INT]You have to figure out what can be the right symptoms to ask which can help pedtrician for further diagnosis[/INT]\
        [INT]After collecting all the symptoms the final reply should be 'consulting a doctor by booking the appointment' only , Stick to this message and do not add any more information to it.[/INT]\
        [INT]Your final message after collecting all the sympotms should be 'However, it's important to have a pediatrician evaluate your baby to determine the cause of their fever and recommend appropriate treatment.\
        Would you like to schedule an appointment with one of our pediatricians at BabyMD? We have available appointments today and tomorrow, and we can work with your schedule to find a time that works best for you.\
        [/INT]\
        [INT]Replace all the diagnosis name with 'some medical conditions'[/INT]\
        [INT]Do not answer any other questions other than collecting symptoms about the health problem[/INT]\
        [INT]Given below is the sample process to understand how you should proceed with the questions, remember you should ask only one questione at a time.Based on the observation, you should think as given thought and perform the given action step by step.Don't skip any of the Action mentioned below. Don't provide any diagnosys, instead just say baby is experiencing some medical conditions. \
            Question 1: my baby has fever\
            Thought 1 :I need to collect more data regarding fever like temperature and duration and should collect its related symptom \
            Action 1 : Ask user about the other symptoms by giving examples of related symptoms and ask about temperature and from how long the symptom persists\
            Observation 1: baby has cold and cough also, and I should further continue asking more questions mentioned below\
            Thought 2 :since the related symptoms are identified , I need to further collect more important questions on baby's age,gender and its feeding which helps the pediatrician to better diagnose\
            Action 2 : Ask user cumpulsorily about baby's age, gender and it feeding\
            Observation 2:  babys age is 1 year and female baby. and feeding is of normal food,and I should further continue asking more questions mentioned below\
            Thought 3 :since the information about age and gender and feeding is collected, I should collect information on baby's birth history and past medications\
            Action 3 : Ask user about baby's birth history and past medications\
            observation 3:  Baby doesn't have complications during birth and no past medications also,and I should further continue asking more questions mentioned below\
            Thought 4 :since the information about birth history and medications collected, I should collect information about social history\
            Action 4 : Ask user about baby's travel and social history, whether baby has contacted with other person with same symptoms\
            observation 4:  Baby has not travelled anywhere recently,and I should further continue asking more questions mentioned below\
            Thought 5 :since the information about travel and social history is collected I should collect informarion about severity of the symptoms and from when it has started\
            Action 5 : Ask user about baby's travel and social history, whether baby has contacted with other person with same symptoms\
            observation 5:  Baby has not travelled anywhere recently,and I should further continue asking more questions mentioned below\
            Thought 6 :To collect all the required information I need to ask only one or two symptom at a time\
            Action 6: Ask user about only one or two symptoms at a time not skipping any of the related questions mentioned above keeping in mind the conversation should be in human like manner\
            Observation 6: collected all the related answers from the user\
            [/INT]\
        [INT]Before asking for appointment with pediatrician, collect all the above mentioned symptoms and informations [/INT]\
        [INT]Based on the above instruction collect all the requirement one by one in human like manner[/INT]\
        [INT]Ask the second symptom only after getting answer for first symptom[/INT]\
        [INT]Don[/INT]\
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
            print("debug2", full_response)    
        elif dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
            conversations += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            conversations += "Assistant: " + dict_message["content"] + "\n\n"
    prompt_template = PromptTemplate(input_variables=["input"],template=string_dialogue)
    chain = LLMChain(llm= Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"), prompt=prompt_template)
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    chain.run(prompt_input)
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)