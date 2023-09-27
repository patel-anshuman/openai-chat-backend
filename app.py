from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

system_message = {
    "role": "system",
    "content": "You are a Healthcare assistant.\nWelcome the user with a message.\nAssess symptoms\nRefer to specific specialists mentioning the department\nIf assessed emergency condition from symptoms, straight away suggest dialling 108 to call an ambulance\nAsk one question at a time. Answer health-related queries only.\nThank you message on completing the assisting."
}
initial_message = {
    "role": "assistant",
    "content": "Welcome to JK Hospital! I'm your health care assistant.\nHow can I assist you with your health today?"
}

messages = [system_message,initial_message]

# Initialize the Flask application
app = Flask(__name__)

# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure CORS settings
from flask_cors import CORS
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Home Page'}), 200

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('user_input', '')
    if not user_input:
        return jsonify({'message': 'User input is required'}), 400

    # Append the user's input to the message history
    messages.append({"role": "user", "content": user_input})

    # Generate a response from the OpenAI model
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=240,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        response_message = response.choices[0].message
        messages.append(response_message)
        return jsonify({'message': response_message}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)