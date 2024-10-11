from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
import json

def lambda_handler(event, context):
    openai_api_key = os.getenv('OPENAI_APIKEY')
    llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
    answer = llm.invoke("""Give me one daily motivational quotes for a new mother suffereing from post partum depression. Don't give any reference or pretext. Lanaguage should be English. Do not reference that the person has postpartum depression.""")
    return {"message":answer}