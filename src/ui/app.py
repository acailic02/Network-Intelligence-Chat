from flask import Flask, request, jsonify, render_template
from sqlalchemy.orm import Session

import json
import os
from src.agents.workflow import build_network_intelligence_workflow
from src.llm.client import chat
from src.storage.db import engine
from src.agents.query_understanding import understand

app = Flask(__name__)
workflow = build_network_intelligence_workflow()

HISTORY_WINDOW = 10
conversation_history = []

query_log_path = os.path.join(os.path.dirname(__file__), "..", ".." , "logs", "final_retrieval_per_query.jsonl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/send', methods=['POST'])
def systemRes():
    data = request.get_json()
    userMSG = data["message"]

    result = workflow.invoke({
        "user_input": userMSG,
        "conversation_history": conversation_history
    })
    
    last_query = None
    with open(query_log_path, "r") as f:
        for line in f:
            last_line = line
    last_query = json.dumps(json.loads(last_line))

    conversation_history.append({
        "user_msg": userMSG,
        "system_res": result["answer"],
        "query_parsed": last_query
    })

    print(json.dumps(conversation_history, indent=2))

    if len(conversation_history) > HISTORY_WINDOW:
        conversation_history.pop(0)

    return jsonify({
        "systemRes": result["answer"],
        "profiles": result["profiles_data"]
    })

if __name__ == "__main__":
    app.run(debug=True)
