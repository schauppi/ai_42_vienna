from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import GoogleSerperAPIWrapper
from ai_42_vienna.llm.agent_tool import WriteTool
from langchain.chat_models import ChatOpenAI
import langchain
from dotenv import load_dotenv
import os
import streamlit as st
langchain.debug = True

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def instantiate():
    """
    Instantiate objects

    Args:
        None

    Returns:
        chat_model: the chat model
        tools: the tools
        search: the search
    """

    chat_model = ChatOpenAI(model="gpt-4", temperature=0.5)

    search = GoogleSerperAPIWrapper()
    write = WriteTool()


    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="useful for when you need to ask with search"
        ),
        Tool(
            name="Write to file",
            func=write.run,
            description="useful for when you need to write something to a file"
        ),


    ]

    return chat_model, tools, search, write


def main():
    """
    Run the main function

    Args:
        None

    Returns:
        None
    """

    chat_model, tools, search, write = instantiate()

    st.title("🦜🔗 AI 42 Vienna")
    prompt = st.text_input("Ask a question (query/prompt)")

    if st.button("Submit Query", type="primary"):

        agent = initialize_agent(tools=tools, llm=chat_model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True )
        result = agent.run(prompt)

        st.write("Output: ")
        st.write(result)

if __name__ == '__main__':
    main()

