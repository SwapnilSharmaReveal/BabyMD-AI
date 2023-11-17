import streamlit as st
import replicate
import os
import re

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

def generate_llama2_summary2(texts):
   text = ' '.join(texts.split())
   #text = """User: my baby is not feeling well. Assistant: Sorry to hear that. Can you please tell me a little bit more about your baby's symptoms? For example, what is their age and gender? User: age is 4 year old, gender is male. Assistant: Great, that helps me to narrow down the possible causes. Can you please tell me what their body temperature is right now? User: current temperature is 112f."""
   summ = f"<s>\
        [SYS]Your task is to summarize the {text} to give the health ovierview and all the information given by user.[/SYS]\
        [INT]Remember to provide a descriptive summary having all the information in the paragraph format.[/INT]\
        [INT]After that from a new line, list down the following parameters from the summary : 1.Chief Compliant, 2.Severity, 3.Duration, 4.Age, 5.Birth History, 6.Feeding History, 7.Social History, 8.Medical History, 9.Related symptoms[/INT]\
        </s>"
   output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
                            input={"prompt": f"{summ}",
                                    "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
   return output

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "<s>\
        As an [SYS], your role is to conduct a conversation with concern and in a professional manner while collecting information about a child's symptoms and basic information from their parents. Your objective is to gather the necessary information within 9 questions while avoiding any potential diagnosis or cause. Please handle any sensitive information shared by the parents with caution and ensure encryption for privacy and security. Inquire about the specific symptoms the child is experiencing and mandatorily collect the following information within two or three lines each: 1. Age and gender of the child, 2. Basic health information, 3. Chief complaint and duration, 4. Allergies, 5. Feeding history, 6. Existing medical conditions, 7. Medications being taken, 8. Birth history, 9. Recent social activities. Once you have gathered all the necessary information, conclude with the message, 'Thank you for sharing the details. Our pediatrician will respond within the next 10 minutes. Remember not to answer any questions that are unrelated to the medical context.\
            "
    conversations = ''
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
            conversations += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            conversations += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output,conversations

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response,conversations = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            word_to_search_1 = "10 minutes"
            word_to_search_2 = "10 mins"
            match1 = re.search(r'\b' + re.escape(word_to_search_1) + r'\b', full_response)
            match2 = re.search(r'\b' + re.escape(word_to_search_2) + r'\b', full_response)
            full_summary = ''
            if (match1 or match2):
                st.markdown("Below is the summary:")
                response1 = generate_llama2_summary2(conversations)
                for item in response1:
                    full_summary += item
                st.markdown(full_summary)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
