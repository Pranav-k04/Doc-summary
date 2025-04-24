from flask import render_template, request, jsonify
from app import app
from app.agents import search_agent, summary_agent
import os

UPLOAD_FOLDER = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q')
    results = search_agent.search_papers(query)
    return jsonify(results)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    structured = summary_agent.generate_structured_summary(filepath)

    return jsonify(structured)
