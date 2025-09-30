from flask import Flask, render_template
import json
with open('data.json') as f:
    data = json.load(f)

def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return render_template('index.html', data=data)
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
