from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import subprocess
import os
import logging
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import shlex

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the LLM with environment variable
llm = OpenAI(api_key='API_KEY')

# Load static prompts from a JSON file
try:
    with open('static_prompts.json', 'r') as f:
        static_prompts = json.load(f)
except FileNotFoundError:
    logging.error('Static prompts file not found')
    static_prompts = {}

@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    data = request.json
    prompt_text = data.get('prompt_text', '')
    mode = data.get('mode', 'dynamic')

    if mode == 'static':
        prompt = static_prompts.get('prompt', 'No static prompt available')
    else:
        if not prompt_text:
            return jsonify({"error": "Prompt text is required for dynamic mode"}), 400
        template = PromptTemplate(
            input_variables=["prompt_text"],
            template="Generate a security bypass technique for the following scenario: {prompt_text}"
        )
        prompt = template.format(prompt_text=prompt_text)

    return jsonify({"prompt": prompt})

@app.route('/send-prompt', methods=['POST'])
def send_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    
    model_endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {'API_KEY'}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(model_endpoint, json=payload, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Error sending prompt to model: {e}")
        return jsonify({"error": "Failed to get response from model"}), 500

    return jsonify({"response": response.json()})

@app.route('/run-garak-scan', methods=['POST'])
def run_garak_scan():
    data = request.json
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    probe = data.get('probe')
    
    # Sanitize inputs
    command = f"python -m garak --model_type {shlex.quote(model_type)} --model_name {shlex.quote(model_name)} --probes {shlex.quote(probe)}"
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e}")
        return jsonify({"error": str(e), "output": e.output}), 500
    
    return jsonify({"result": output})

if __name__ == '__main__':
    app.run(debug=True)
