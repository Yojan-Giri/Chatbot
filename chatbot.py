import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk
import requests
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

# Load and preprocess the conversational dataset
conversations = [
    ["Hello", "Hi there!"],
    ["Hi","Hello"],
    ["What's your name?", "I am a chatbot. You can call me ChatBot."],
    ["How are you?", "I'm doing well, thank you!"],
    ["What can you do?", "I can provide information and have basic conversations."],
    ["Can you give me today's  weather of Kathmandu?", "Yes sure."]
    # Add more dialogues as per your dataset
]

# Separate user queries and bot responses
user_queries = [conversation[0] for conversation in conversations]
bot_responses = [conversation[1] for conversation in conversations]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(user_queries + bot_responses)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences of integers
user_queries_sequences = tokenizer.texts_to_sequences(user_queries)
bot_responses_sequences = tokenizer.texts_to_sequences(bot_responses)

# Find maximum lengths for user queries and bot responses
max_user_query_length = max(len(sequence) for sequence in user_queries_sequences)
max_bot_response_length = max(len(sequence) for sequence in bot_responses_sequences)

# Pad user queries and bot responses separately
user_queries_sequences = pad_sequences(user_queries_sequences, maxlen=max_user_query_length, padding="post")
bot_responses_sequences = pad_sequences(bot_responses_sequences,maxlen=max_bot_response_length, padding="post")

# Function to generate bot responses using the trained model
def generate_bot_response(user_input):
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_sequence = pad_sequences(user_input_sequence, maxlen=max_user_query_length, padding="post")
    bot_response_sequence = model.predict(user_input_sequence)[0]
    bot_response = tokenizer.sequences_to_texts([bot_response_sequence])[0]
    return bot_response

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100, input_length=max_user_query_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(user_queries_sequences, bot_responses_sequences, epochs=10, batch_size=32)

# Save the trained model
model.save("chatbot_model.h5")

# Function to get real-time weather information
def get_weather(city):
    api_key = "77de4aae6c6e4509f65077dc57147205"  
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if data["cod"] == "404":
        return "Weather information not found."
    else:
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        return f"Temperature: {temperature}°C, Humidity: {humidity}%, Description: {description}"

# Main loop for chatting
prev_response = ""
while True:
    user_input = input("User: ")
    preprocessed_input = preprocess_input(user_input)
    entities = extract_entities(preprocessed_input)
    sentiment = analyze_sentiment(user_input)
    response = get_bot_response(user_input, sentiment)
    
    if prev_response:
        feedback = input("Was my previous response helpful? [Positive/Negative/None]: ")
        if feedback.lower() == "positive" or feedback.lower() == "negative":
            feedback_response = generate_bot_response(user_input)
            print("ChatNLTK:", feedback_response)
    
    print("Entities:", entities)
    print("Sentiment:", sentiment)
    
    # Check if the user asked for weather information
    if "weather" in entities:
        city = entities["weather"]
        weather_response = get_weather(city)
        print("Weather:", weather_response)
    
    print("ChatNLTK:", response)
    prev_response = response
    
    if user_input.lower() == "quit":
        break
