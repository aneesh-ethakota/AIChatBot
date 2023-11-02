import os
import pickle
from typing import Optional

import streamlit as st

import langchain
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.vectorstores import VectorStore
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.agents.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from flight_search_schema import CustomFlightSearchSchema

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

import prompt_templates as pt

st.set_page_config(page_title="AI Assistant")
st.title("AI Assistant")


def load_retriever():
    with open("vectorstore_all.pkl", "rb") as f:
        vectorstore: Optional[VectorStore] = pickle.load(f)
    return vectorstore.as_retriever()


os.environ["OPENAI_API_KEY"] = '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'
os.environ['OPENWEATHERMAP_API_KEY'] = "f0e91a827c639f496766867f1ba5f145"
os.environ["AMADEUS_CLIENT_ID"] = "4C1Tl1dDicPJehADPyqs6966mkc4flj8"
os.environ["AMADEUS_CLIENT_SECRET"] = "5QkAgl6spFED9zvJ"

# Setup LLM
llm_gpt_4 = OpenAI(model_name="text-davinci-003", temperature=0, streaming=True)

# Setup Chat LLM
chat_llm_gpt_4 = ChatOpenAI(
    model_name="gpt-4", temperature=0, streaming=True
)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, output_key="output",
                                  return_messages=True)


def create_conv_retrieval_chain(model, retriever):
    qa_chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        # callbacks=[PrintRetrievalHandler(st.container())],
        # verbose=True
    )
    return qa_chain


def create_retrieval_chain(model, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=retriever,
        chain_type="stuff",
        # condense_question_llm=question_gen_llm,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        # memory=memory,
        # verbose=True
    )
    return qa_chain


def create_internal_kb_tools(chain_type="conv_retrieval"):
    retriever = load_retriever()

    tool_name = "Internal Knowledge Base"
    tool_description = """
    Only Use this tool when answering general knowledge queries regarding the documents present. Also remember to passing the raw user question to this tool
    as an action input. 
    """

    if chain_type in ['conv_retrieval']:
        chat_llm_gpt_3_5_turbo = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0, streaming=True
        )
        conv_qa_chain = create_conv_retrieval_chain(model=chat_llm_gpt_3_5_turbo, retriever=retriever)
        knowledge_base_tool = Tool(
            name=tool_name,
            func=lambda query: conv_qa_chain({"question": query, "chat_history": memory.chat_memory.messages}),
            description=tool_description
        )
    else:
        qa_chain = create_retrieval_chain(model=chat_llm_gpt_4, retriever=retriever)
        knowledge_base_tool = Tool(
            name=tool_name,
            func=qa_chain.run,
            description=tool_description
        )

    return knowledge_base_tool


def create_weather_tools():
    weather = OpenWeatherMapAPIWrapper()
    weather_tool = Tool(
        name="Check Weather",
        func=weather.run,
        description="""
            A wrapper around OpenWeatherMap API. Useful for fetching current weather information for a specified 
            location. Input should be a location string (e.g. London,GB)., args: 
            {{{{'location': {{{{'title': 'Location', 'type': 'string'}}}}}}}}
            Useful for when you need to check weather for a particular City.
            It only works for single city at a time to get weather details.
            This tool only understand the city name so don't pass any city code etc.
        """
    )
    return [weather_tool]

    # weather_tools = load_tools(['openweathermap-api'])
    # return weather_tools


chat_llm_agent = initialize_agent(
    create_weather_tools() + [create_internal_kb_tools()],
    chat_llm_gpt_4,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, return_intermediate_steps=True, memory=memory
)

destinations = [f"{p['name']}: {p['description']}" for p in pt.prompts_info]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm_gpt_4, router_prompt, verbose=True)

# Show me all available containers and their weight capacity.

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_st_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = router_chain(user_query)
        st.write(response)
        if response['destination'] == "internal_knowledge_base":
            prompt = PromptTemplate(template=pt.internal_knowledge_base, input_variables=["input"])
            response = chat_llm_agent(prompt.format(input=response['input']),
                                      callbacks=[streamlit_st_handler])
            st.write(response['output'])
        else:
            st.write(response['destination'])
