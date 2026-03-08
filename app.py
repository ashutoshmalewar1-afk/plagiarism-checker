# app.py
import os
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64

# You can import other libraries you need
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Example route: Home page
@app.route("/")
def home():
    return "<h1>Welcome to My Flask App on Render!</h1>"

# Example route: Generate a simple plot
@app.route("/plot")
def plot():
    # Create a simple plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure()
    plt.plot(x, y)
    plt.title("Sample Plot")
    
    # Save plot to PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f'<img src="data:image/png;base64,{plot_url}"/>'

# Example route: Use sentence-transformers
@app.route("/encode", methods=["POST"])
def encode():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400
    
    text = data["text"]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text).tolist()
    
    return jsonify({"embedding": embedding})

# Ensure Render detects the correct port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)