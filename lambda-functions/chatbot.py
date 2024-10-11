from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
import json

def lambda_handler(event, context):
    openai_api_key = os.getenv('OPENAI_APIKEY')
    if not openai_api_key:
        return {'statusCode': 500, 'body': json.dumps('OpenAI API key not configured.')}

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    prompt_template = PromptTemplate(
        input_variables=['question', 'age'],
        template="""
            You are an expert in the field of postnatal care.
            I am a mom.
            The baby is {age}.
            Can you help me with: {question}.
        """
    )

    if 'question' not in event or 'age' not in event:
        return {'statusCode': 400, 'body': json.dumps('Missing question or age in the input.')}

    chain_input = {
        "question": event['question'],
        "age": event['age']
    }

    chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = chain.invoke(chain_input)
    answer_text = answer['text'].replace("\n", "")

    # Return the answer as a JSON response
    return {
        'statusCode': 200,
        'body': json.dumps(answer_text)
    }