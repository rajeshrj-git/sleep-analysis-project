# api/app.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_analysis import run_analysis

from flask import Flask, jsonify
from run_analysis import run_analysis

app = Flask(__name__)


@app.route('/analyze_sleep', methods=['GET'])
def analyze_sleep():
    result = run_analysis()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
