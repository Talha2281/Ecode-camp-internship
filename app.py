from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
)

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser

import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="AI Text Assistant", page_icon="🤖")

# Title and initial description
st.title('AI Chatbot')
st.write("This chatbot is created by TALHA KHAN")
st.markdown("Hello! I'm your AI assistant. How can I assist you today?")
st.image("data:image/jpeg;base64,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", use_column_width=True)

# Function to get and store the API key securely in session state
def get_api_key():
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key")
    return api_key

# Prompt user for API key input
api_key = get_api_key()

# Ensure the API key is available
if not api_key:
    st.warning("Please enter your API Key to continue.")
else:
    # Create a prompt template
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a helpful AI assistant. Please respond to user queries in English."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Initialize message history
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    # Set up the Google AI model using the provided API key
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    # Combine prompt, model, and output parser
    chain = prompt | model | StrOutputParser()

    # Combine chain with history for maintaining session
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # Get user input
    user_input = st.text_input("Enter your question in English:", "")

    # Check if user input is provided
    if user_input:
        st.chat_message("human").write(user_input)

        # Assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Configuration dictionary
            config = {"configurable": {"session_id": "any"}}

            # Get response from AI model
            response = chain_with_history.stream({"question": user_input}, config)

            # Stream and display response in real-time
            for res in response:
                full_response += res or ""
                message_placeholder.markdown(full_response + "|")
                message_placeholder.markdown(full_response)

    else:
        st.warning("Please enter your question.")
