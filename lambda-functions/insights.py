from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
import json

def lambda_handler(event, context):
    openai_api_key = os.getenv('OPENAI_APIKEY')

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    chain_input = {
        "totalpoints": event["totalpoints"],
        "journalentries": event["journalentries"],
        "gratitudeseen": event["gratitudeseen"],
        "accuracy": event["accuracy"],
        "weight": event["weight"],
        "height": event["height"],
        "bmi": event["bmi"],
        "days": event["days"],
        "ppd": event["ppd"],
        "age": event["age"],
        "avgjournal": event["avgjournal"],
        "exercise": event["exercise"],
        "carbs": event["carbs"],
        "proteins": event["proteins"],
        "fats": event["fats"],
        "water": event["water"],
        "calorie": event["calorie"]
    }

    past_template = """
        totalpoints: {totalpoints},
        journalentries: {journalentries},
        gratitudeseen: {gratitudeseen},
        accuracy: {accuracy},
        weight: {weight},
        height: {height},
        bmi: {bmi},
        days: {days},
        ppd: {ppd},
        age: {age},
        avgjournal: {avgjournal},
        exercise: {exercise},
        carbs: {carbs},
        proteins: {proteins},
        fats: {fats},
        water: {water},
        calorie: {calorie}
        
        Legend of data:
        totalpoints: App uses game based approach where every point is a positive activity,
        journalentries: Number of daily journals written,
        gratitudeseen: Number of motivational stories seen,
        accuracy: Accuracy of positive points,
        days: Days since the user started using the app,
        ppd: Status of Post Partum Depression,
        avgjournal: The average sentiment analysis score of the user based on jounal entries,
        exercise: No of times the user exercises in a week,
        carbs: Weight of Carbs a user eats daily,
        proteins: Weight of Proteins a user eats daily,
        fats: Weight of Fats a user eats daily,
        water: Volume of water a user eats daily,
        calorie: ampount of calories a user eats daily

        You are an expert in the field of postnatal care.
        I am a mom. I have been using an app that helps me keep track of different inforamtion.
        Based on the following information give insights summing up my journey up until now.

        Word Limit: 100 words.
        Tone: Empathetic, sensitive, positive, jolly
        No references, pretext. Just give the answer
    """

    present_template = """
            totalpoints: {totalpoints},
            journalentries: {journalentries},
            gratitudeseen: {gratitudeseen},
            accuracy: {accuracy},
            weight: {weight},
            height: {height},
            bmi: {bmi},
            days: {days},
            ppd: {ppd},
            age: {age},
            avgjournal: {avgjournal},
            exercise: {exercise},
            carbs: {carbs},
            proteins: {proteins},
            fats: {fats},
            water: {water},
            calorie: {calorie}
            
            Legend of data:
            totalpoints: App uses game based approach where every point is a positive activity,
            journalentries: Number of daily journals written,
            gratitudeseen: Number of motivational stories seen,
            accuracy: Accuracy of positive points,
            days: Days since the user started using the app,
            ppd: Status of Post Partum Depression,
            avgjournal: The average sentiment analysis score of the user based on jounal entries,
            exercise: No of times the user exercises in a week,
            carbs: Weight of Carbs a user eats daily,
            proteins: Weight of Proteins a user eats daily,
            fats: Weight of Fats a user eats daily,
            water: Volume of water a user eats daily,
            calorie: ampount of calories a user eats daily

            You are an expert in the field of postnatal care.
            I am a mom. I have been using an app that helps me keep track of different inforamtion.
            Based on the following information give insights summarizing my current condition.


            Word Limit: 100 words.
            Tone: Empathetic, sensitive, positive, jolly
            No references, pretext. Just give the answer
        """
    
    future_template = """totalpoints: {totalpoints},
            journalentries: {journalentries},
            gratitudeseen: {gratitudeseen},
            accuracy: {accuracy},
            weight: {weight},
            height: {height},
            bmi: {bmi},
            days: {days},
            ppd: {ppd},
            age: {age},
            avgjournal: {avgjournal},
            exercise: {exercise},
            carbs: {carbs},
            proteins: {proteins},
            fats: {fats},
            water: {water},
            calorie: {calorie}
            
            Legend of data:
            totalpoints: App uses game based approach where every point is a positive activity,
            journalentries: Number of daily journals written,
            gratitudeseen: Number of motivational stories seen,
            accuracy: Accuracy of positive points,
            days: Days since the user started using the app,
            ppd: Status of Post Partum Depression,
            avgjournal: The average sentiment analysis score of the user based on jounal entries,
            exercise: No of times the user exercises in a week,
            carbs: Weight of Carbs a user eats daily,
            proteins: Weight of Proteins a user eats daily,
            fats: Weight of Fats a user eats daily,
            water: Volume of water a user eats daily,
            calorie: ampount of calories a user eats daily

            You are an expert in the field of postnatal care.
            I am a mom. I have been using an app that helps me keep track of different inforamtion.
            Based on the following information give insights as to how i should proceed.

            Word Limit: 100 words.
            Tone: Empathetic, sensitive, positive, jolly
            No references, pretext. Just give the answer
        """

    prompt_template_past = PromptTemplate(
        input_variables=list(chain_input.keys()),
        template = past_template
        )

    prompt_template_present = PromptTemplate(
        input_variables=list(chain_input.keys()),
        template = present_template
    )

    prompt_template_future = PromptTemplate(
        input_variables=list(chain_input.keys()),
        template = future_template
    )

    chain_past = LLMChain(llm=llm, prompt=prompt_template_past)
    answer_past = chain_past.invoke(chain_input)
    answer_text_past = answer_past['text'].replace("\n", "")

    chain_present = LLMChain(llm=llm, prompt=prompt_template_past)
    answer_present = chain_present.invoke(chain_input)
    answer_text_present = answer_present['text'].replace("\n", "")

    chain_future = LLMChain(llm=llm, prompt=prompt_template_past)
    answer_future = chain_future.invoke(chain_input)
    answer_text_future = answer_future['text'].replace("\n", "")

    # Return the answer as a JSON response
    return {
        'statusCode': 200,
        'body': json.dumps({
            'past': answer_text_past,
            'present': answer_text_present,
            'future': answer_text_future
        })
    }