from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import time

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
CORS(app)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load training data
try:
    with open('api/intents.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    # Create the directory if it doesn't exist
    os.makedirs('api', exist_ok=True)
    
    # Create a default intents.json file with basic patterns
    intents = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
                "responses": ["Hello!", "Hey there!", "Hi! How can I help you?", "Greetings!"]
            },
            {
                "tag": "goodbye",
                "patterns": ["Bye", "See you later", "Goodbye", "I'm leaving"],
                "responses": ["Goodbye!", "Talk to you later!", "See you soon!"]
            },
            {
                "tag": "thanks",
                "patterns": ["Thanks", "Thank you", "That's helpful"],
                "responses": ["You're welcome!", "Anytime!", "Happy to help!"]
            },
            {
                "tag": "about",
                "patterns": ["Who are you", "What are you", "Tell me about yourself"],
                "responses": ["I'm a simple chatbot created with Python!", "I'm your friendly AI assistant.", "I'm a chatbot here to help you with your questions."]
            },
            {
                "tag": "help",
                "patterns": ["Help", "I need help", "Can you help me", "What can you do"],
                "responses": ["I can answer questions, have a conversation, or just chat!", "I'm here to assist with information and conversation.", "Ask me anything, and I'll try my best to help!"]
            },
            {
                "tag": "fallback",
                "patterns": [],
                "responses": ["I'm not sure I understand.", "Could you please rephrase that?", "I don't have an answer for that yet.", "I'm still learning!"]
            }
        ]
    }
    
    # Save the default intents
    with open('api/intents.json', 'w') as file:
        json.dump(intents, file, indent=2)

# Conversation history
conversation_history = []

def preprocess_sentence(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence.lower())
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def find_intent(message):
    processed_message = preprocess_sentence(message)
    
    # Check each intent
    highest_score = 0
    matched_intent = None
    
    for intent in intents["intents"]:
        score = 0
        for pattern in intent["patterns"]:
            processed_pattern = preprocess_sentence(pattern)
            
            # Count matching words
            matching_words = set(processed_message).intersection(set(processed_pattern))
            if matching_words:
                pattern_score = len(matching_words) / max(len(processed_pattern), len(processed_message))
                score = max(score, pattern_score)
        
        if score > highest_score and score > 0.2:  # Threshold for matching
            highest_score = score
            matched_intent = intent
    
    # Return fallback if no match found
    if matched_intent is None:
        for intent in intents["intents"]:
            if intent["tag"] == "fallback":
                return intent
    
    return matched_intent

@app.route('/api/message', methods=['POST'])
def process_message():
    data = request.json
    message = data.get('message', '')
    
    # Add a slight delay to simulate processing time
    time.sleep(0.5)
    
    # Find the matching intent
    matched_intent = find_intent(message)
    
    # Get a random response from the intent
    if matched_intent and "responses" in matched_intent:
        response = random.choice(matched_intent["responses"])
    else:
        # Fallback response
        fallback_intent = next((intent for intent in intents["intents"] if intent["tag"] == "fallback"), None)
        response = random.choice(fallback_intent["responses"]) if fallback_intent else "I don't understand that yet."
    
    # Add to conversation history
    conversation_history.append({"user": message, "bot": response})
    
    return jsonify({
        "response": response,
        "intent": matched_intent["tag"] if matched_intent else "unknown"
    })

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(conversation_history)

@app.route('/api/train', methods=['POST'])
def train_chatbot():
    data = request.json
    new_pattern = data.get('pattern', '')
    response = data.get('response', '')
    tag = data.get('tag', '')
    
    if not tag:
        return jsonify({"error": "Tag is required"}), 400
    
    if not new_pattern:
        return jsonify({"error": "Pattern is required"}), 400
    
    if not response:
        return jsonify({"error": "Response is required"}), 400
    
    # Find the intent with the given tag or create a new one
    intent_found = False
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            if new_pattern not in intent["patterns"]:
                intent["patterns"].append(new_pattern)
            if response not in intent["responses"]:
                intent["responses"].append(response)
            intent_found = True
            break
    
    if not intent_found:
        intents["intents"].append({
            "tag": tag,
            "patterns": [new_pattern],
            "responses": [response]
        })
    
    # Save the updated intents
    with open('api/intents.json', 'w') as file:
        json.dump(intents, file, indent=2)
    
    return jsonify({"success": True, "message": "Chatbot trained successfully"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)