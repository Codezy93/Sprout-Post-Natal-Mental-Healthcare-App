import json
from nltk.sentiment import SentimentIntensityAnalyzer

def lambda_handler(event, context):
    answer = ""
    sia = SentimentIntensityAnalyzer()
    text = event['text']
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        answer = 1
    elif sentiment_scores['compound'] <= -0.05:
        answer = -1
    else:
        answer = 0
    return {
        'statusCode': 200,
        'body': json.dumps(answer)
    }