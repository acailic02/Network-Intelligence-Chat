from flask import Flask, request, jsonify, render_template
from sqlalchemy.orm import Session
from src.llm.client import chat
import time
from src.storage.db import engine
from src.storage.repo import get_connections

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

    with Session(engine) as session:
        query_res = get_connections(session, country=userMSG)

    return jsonify({
        "systemRes": res,
        "profiles": [x.first_name + " " + x.last_name for x in query_res]
    })

if __name__ == "__main__":
    app.run(debug=True)