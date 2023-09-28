from flask import Flask, request, jsonify
import os
import datetime
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

load_dotenv()

memory = ConversationBufferMemory()
memory.save_context(
    {
        "input": "You are a Healthcare assistant.\nWelcome the user with a message.\nAssess symptoms\nRefer to specific specialists mentioning the department\nIf assessed emergency condition from symptoms, straight away suggest dialling 108 to call an ambulance\nAsk one question at a time. Answer health-related queries only.\nThank you message on completing the assisting."
    },
    {
        "output": "Welcome to JK Hospital! I'm your health care assistant.\nHow can I assist you with your health today?"
    },
)

# Initialize the Flask application
app = Flask(__name__)

# Set up the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Configure CORS settings
from flask_cors import CORS

CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Home Page"}), 200


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("user_input", "")
    if not user_input:
        return jsonify({"message": "User input is required"}), 400

    try:
        response = ChatOpenAI(temperature=0.9)
        conversation = ConversationChain(llm=response, memory=memory, verbose=False)
        response_message = conversation.predict(input=user_input)
        # print(response_message)
        return jsonify({"message": response_message, "memory": memory.buffer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save-chat", methods=["GET"])
def save_chat():
    try:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")
        file_name = f"ChatHistory{formatted_datetime}.txt"

        conversation_history = memory.buffer
        conversation_history = conversation_history[358:]

        with open(file_name, "w") as file:
            file.write(conversation_history)

        memory.clear()
        memory.save_context(
            {
                "input": "You are a Healthcare assistant.\nWelcome the user with a message.\nAssess symptoms\nRefer to specific specialists mentioning the department\nIf assessed emergency condition from symptoms, straight away suggest dialling 108 to call an ambulance\nAsk one question at a time. Answer health-related queries only.\nThank you message on completing the assisting."
            },
            {
                "output": "Welcome to JK Hospital! I'm your health care assistant.\nHow can I assist you with your health today?"
            },
        )

        return jsonify({"message": f"Conversation history saved as {file_name}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
