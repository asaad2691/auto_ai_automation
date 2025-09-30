from flask import Flask, render_template, request, jsonify
import os
import json
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact2():
    return render_template('contact.html')

@app.route('/about')  # New route added for about.html
def about():
    return render_template('about.html')

@app.route('/contact-post', methods=['POST'])
def contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")

    new_data = {"name": name, "email": email, "message": message}

    data_file = os.path.join(app.root_path, 'static', 'data.json')
    
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"messages": []}
    else:
        data = {"messages": []}
    
    if "messages" not in data:
        data["messages"] = []
    data["messages"].append(new_data)
    
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=4)

    return jsonify({"message": "Data added successfully!"}), 201

if __name__ == '__main__':
    app.run(debug=True)
