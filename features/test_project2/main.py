from flask import Flask, render_template
import os
import json

app = Flask(__name__)

@app.route('/')
def home():
    data_path = os.path.join(app.root_path, "static", "data.json")
    with open(data_path, 'r') as f:
        projects = json.load(f)
    return render_template('index.html', projects=projects["projects"])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
