
import os
import asyncio
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

import time
from functools import wraps

# set the temperature
creativity = st.sidebar.slider('Creativity', 0.0, 1.0, 0.7)

# Importing the large language model OpenAI via langchain
model = AzureChatOpenAI(
    openai_api_version="2023-09-01-preview",
    azure_endpoint=os.getenv('AZURE_API_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    azure_deployment=os.getenv('OPENAI_DEPLOYMENT_NAME'),
    model_name=os.getenv('OPENAI_MODEL_NAME'),
    model_version=os.getenv('OPENAI_API_VERSION'),  # Add the desired version identifier
    temperature=creativity
)

## TITLE

# setting up the title prompt templates
title_template = PromptTemplate(
    input_variables = ['concept'], 
    template='Give me a youtube video title about {concept}'
)

# memory buffer
memoryT = ConversationBufferMemory(
    input_key='concept',
      memory_key='chat_history')

# LLM chain
chainT = LLMChain(
    llm=model, 
    prompt=title_template, 
    verbose=True, 
    output_key='title', 
    memory=memoryT)

## SCRIPT
# setting up the script prompt templates
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'web_search'], 
    template='''Give me an attractive youtube video script based on the title {title} 
    while making use of the information and knowledge obtained from the Wikipedia research:{wikipedia_research}
    make use of the additional information from the web search:{web_search}''',
)


# memory buffer
memoryS = ConversationBufferMemory(
    input_key='title', 
    memory_key='chat_history')

# LLM chain
chainS = LLMChain(
    llm=model, 
    prompt=script_template, 
    verbose=True, 
    output_key='script', 
    memory=memoryS)


def retry(max_attempts=3, delay_seconds=2, check_return_value=False):
    """
    A decorator for retrying a function if it raises an exception or returns a falsy value (if check_return_value is True).

    Args:
        max_attempts (int): Maximum number of attempts.
        delay_seconds (int): Delay between attempts in seconds.
        check_return_value (bool): If True, also retry if the return value is falsy.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    if not check_return_value or result:
                        return result
                    else:
                        raise ValueError("Function returned a falsy value.")
                except Exception as e:
                    print(f"Attempt {attempts+1} failed with error: {e}")
                    attempts += 1
                    if attempts < max_attempts:
                        print(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
            raise Exception(f"All {max_attempts} attempts failed.")
        return wrapper
    return decorator


@retry(max_attempts=5, delay_seconds=2, check_return_value=True)
def fetch_wikipedia_data(title):
    wikipedia = WikipediaAPIWrapper()
    return wikipedia.run(title)

@retry(max_attempts=5, delay_seconds=2, check_return_value=True)
def fetch_web_search_results(title):
    search = DuckDuckGoSearchRun()
    return search.run(title)

async def generate_script(input_text):
    gen_title = chainT.run(input_text)
    wikipedia_research = fetch_wikipedia_data(input_text)
    web_search = fetch_web_search_results(input_text)
    script = chainS.run(
        title=gen_title, 
        wikipedia_research=wikipedia_research, 
        web_search=web_search
        )
    
    return gen_title, script, wikipedia_research, web_search


# This function is a wrapper around the async function 'generate_script'
# It allows us to call the async function in a synchronous way
# using 'asyncio.run'
def run_generate_script(input_text):
    """
    Wrapper function to run the async function 'generate_script'
    in a synchronous way
    
    Args:
        input_text (str): The input text passed to the language model

    Returns:
        tuple: A tuple containing the title, script, and wikipedia research
    """
    return asyncio.run(generate_script(input_text))


if __name__ == '__main__':

    # Set the title using StreamLit
    st.title(' Video Script Generator')
    input_text = st.text_input('Enter the video title: ') 

    # Display the output if the the user gives an input
    if input_text:

        # async function call
        title, script, wikipedia_research, search_results = run_generate_script(input_text)

        # writing the title and script
        st.write(title) 
        st.write(script) 

        with st.expander('Wikipedia-based exploration: '): 
            st.info(wikipedia_research)

        with st.expander('Web-based exploration: '):
            st.info(search_results)

