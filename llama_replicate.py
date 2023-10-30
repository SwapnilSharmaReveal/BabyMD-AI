import streamlit as st
import replicate
from langchain.llms import Replicate
import os
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from llamaapi import LlamaAPI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import SimpleSequentialChain
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
replicate_api = os.environ['REPLICATE_API_TOKEN']
# replicate_api = os.environ.get("REPLICATE_API_TOKEN")
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

# llm = dict('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5')
# llm_info = {'name': 'a16z-infra/llama13b-v2-chat', 'version': 'df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'}
# llm = LlamaAPI('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5')

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

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    prompt_template ="""<s>\
        [SYS]You are a assistant to BBMD pedtrician who needs to collect symptoms of the user and who doesnt provide any diagnosis or disease name. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.[/SYS]\
        [INT]Remember you are a symptoms collector[/INT]\
        [INT]Do not provide any diagnosis or disease name at the end of the conversation[/INT]\
        [INT]Do not predict any potential cause for the health problems[/INT]\
        [INT]Your only job is to collect all the related symptoms from the patient according the the health problem they are facing[/INT]\
        [INT]Do not answer any questions other than responding to patients health problems and collecting symptoms[/INT]\
        [INT]To collect symptoms ask the patient about symptoms like - 1. Body temperature 2. Allergies 3. what food they had?  4. Are there is existing medical conditions 5. Are they taking any medicines[/INT]\
        [INT]Ask about one symptom at a time, Carry the conversation[/INT]\
        [INT]Collect all the information related to Body Temperature, Any other health problem, Allergies, Are they taking any medications, Is there any past medical history[/INT]\
        [INT]You have to figure out what can be the right symptoms to ask which can help pedtrician for further diagnosis[/INT]\
        [INT]After collecting all the symptoms the final reply should be 'consulting a doctor by booking the appointment' only , Stick to this message and do not add any more information to it.[/INT]\
        [INT]Your final message after collecting all the sympotms should be 'However, it's important to have a pediatrician evaluate your baby to determine the cause of their fever and recommend appropriate treatment.\
        Would you like to schedule an appointment with one of our pediatricians at BabyMD? We have available appointments today and tomorrow, and we can work with your schedule to find a time that works best for you.\
        [/INT]\
        [INT]Replace all the diagnosis name with 'some medical conditions'[/INT]\
        [INT]Do not answer any other questions other than collecting symptoms about the health problem[/INT]\
        [INT]{input} this is the question to which you need to answer by following all the above instructions[/INT]\
        """
    prompt_template = PromptTemplate(input_variables=["input"],template=prompt_template)
    # first_chain: LLMChain = LLMChain(llm=model, prompt=first_prompt_template)
    # print("firstone",prompt_template)
    
    validate_template = """<s>\
        [INT]'thanks for the reply':{response}[/INT]\
        """
    validate_template = PromptTemplate(input_variables=["response"],template=validate_template)
    # print("replacing",validate_template)
    
    
    print("aaaaa",st.session_state.messages)
    print("zzzz",prompt_input)
    for dict_message in st.session_state.messages:
        print("bbbb",dict_message)
        if dict_message["role"] == "user":
            prompt_template += "User: " + dict_message["content"] + "\n\n"
            print("ccccc",prompt_template)
        else:
            prompt_template += "Assistant: " + dict_message["content"] + "\n\n"
            print("ddddd",prompt_template)
    # output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
    #                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
    #                               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    # print("sasas",output)
    # return output
    first_chain: LLMChain =  LLMChain(
    llm = Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"), 
    prompt=prompt_template,
    memory=ConversationBufferWindowMemory(k=2),
    llm_kwargs={"max_length": 4096}
)
    # print("first1",first_chain)
    
    second_chain: LLMChain =  LLMChain(
    llm = Replicate(model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"), 
    prompt=validate_template,
    memory=ConversationBufferWindowMemory(k=2),
    llm_kwargs={"max_length": 4096}
)
    # print("second",second_chain)
    
    ss_chain = SimpleSequentialChain(
    chains=[first_chain,second_chain])
    review = ss_chain.run(prompt_input)
    
    print("review",review)
    return review

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
            # print("gfds",response)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    # print("asdfghgfdsa",message)
    st.session_state.messages.append(message)