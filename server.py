import os
from flask import Flask, request, jsonify, send_from_directory
import subprocess

app = Flask(__name__)

# Directory paths
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), 'scripts')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/mouse', methods=['POST'])
def mouse():
    script_path = os.path.join(SCRIPTS_DIR, 'mouse.py')
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    return jsonify(result.stdout)

@app.route('/volume', methods=['POST'])
def volume():
    script_path = os.path.join(SCRIPTS_DIR, 'volume.py')
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    return jsonify(result.stdout)

@app.route('/texture', methods=['POST'])
def texture():
    script_path = os.path.join(SCRIPTS_DIR, 'texture.py')
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    return jsonify(result.stdout)

if __name__ == '__main__':
    app.run(debug=True)
