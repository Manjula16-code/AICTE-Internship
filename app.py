from flask import Flask, render_template, request
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

generation_config = {
    "temperature": 0.45,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""You are a diet planner chatbot designed to help users create personalized and balanced meal plans tailored to their 
    health goals. Gather necessary information from users, including their age, weight, height, gender, and specific goals (e.g., weight loss, 
    muscle gain, maintenance) in 20 words and provide appropriate meal plans along with nutritional benifits and calories in concise way point-wise.
    Also suggest healthy recipe ideas or alternatives if users ask for variety. Offer motivation, guidance, and tips for sustainable eating habits. 
    Use a friendly, non-judgmental, and supportive tone while interacting. Make sure you ask questions one after the other and answer in fewer 
    sentences so that the user don't feel hectic to read and include emojis if required. Do not provide medical advice. Encourage users to consult
    a registered dietitian or healthcare professional for specific medical concerns."""
)

chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    bot_reply = ""
    if request.method == "POST":
        user_input = request.form["user_input"]

        chat_session = model.start_chat(history=chat_history)
        response = chat_session.send_message(user_input)
        bot_reply = response.text

        chat_history.append({"role": "user", "parts": [user_input]})
        chat_history.append({"role": "model", "parts": [bot_reply]})

    return render_template("index.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
