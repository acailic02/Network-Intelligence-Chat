from flask import Flask, request, jsonify, render_template
from src.llm.client import chat

import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/send', methods=['POST'])
def systemRes():
    data = request.get_json()
    userMSG = data["message"]

    seconds = 2
    time.sleep(seconds)

    res = chat(
        messages=[{"role": "user", "content": userMSG}],
        system="Ti si asistent koji pomaže korisnicima da pretražuju svoju LinkedIn mrežu."
    )["text"] #Here we call LLM

    return jsonify({"systemRes": res})

if __name__ == "__main__":
    app.run(debug=True)