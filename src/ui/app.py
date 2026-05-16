from flask import Flask, request, jsonify, render_template
from sqlalchemy.orm import Session

from src.agents.workflow import build_network_intelligence_workflow
from src.llm.client import chat
from src.storage.db import engine
from src.agents.query_understanding import understand

app = Flask(__name__)
workflow = build_network_intelligence_workflow()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/send', methods=['POST'])
def systemRes():
    data = request.get_json()
    userMSG = data["message"]

    #res = chat(
    #    messages=[{"role": "user", "content": userMSG}],
    #    system="You are assistent that helps in searching users LinkedIn network."
    #)["text"]

    #with Session(engine) as session:
    #    query_res = get_connections(session, country=userMSG)

    #profiles = [x.first_name + " " + x.last_name for x in query_res]

    result = workflow.invoke({"user_input": userMSG})

    return jsonify({
        "systemRes": result["answer"],
        #"profiles": profiles
    })

if __name__ == "__main__":
    app.run(debug=True)
