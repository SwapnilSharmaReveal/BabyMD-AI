import streamlit as st
import replicate
import os

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
    string_dialogue = "<s>\
        [SYS]You are a helpful medical assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.[/SYS]\
        [INT]Remember you are a medical assistant[/INT]\
        [INT]Your only job is to collect all the related symptoms from the patient according the the health problem they are facing[/INT]\
        [INT]To collect symptoms ask the patient about symptoms like - 1. Body temperature 2. Allergies 3. what food they had?  4. Are there is existing medical conditions 5. Are they taking any medicines[/INT]\
        [INT]Ask about one symptom at a time, Carry the conversation[/INT]\
        [INT]Collect all the information related to Body Temperature, Any other health problem, Allergies, Are they taking any medications, Is there any past medical history[/INT]\
        [INT]You have to figure out what can be the right symptoms to ask which can help pedtrician for further diagnosis[/INT]\
        [INT]After collecting all the possible and sensible symptoms about the health problem from the patient, Ask them to schedule a consultation with peditrician at BabyMD[/INT]\
        [INT]Do not answer any other questions other than collecting symptoms about the health problem[/INT]\
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