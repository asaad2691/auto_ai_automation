from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
login_manager = LoginManager()
login_manager.init_app(app)

users = {'admin': {'password': 'admin'}}

class User(UserMixin):
    pass

@login_manager.user_loader
def user_loader(email):
    if email not in users:
        return

    user = User()
    user.id = email
    return user

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    emit('reply', data)

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    
    if users[email]['password'] == password:
        user = User()
        user.id = email
        login_user(user)
        return "Logged in successfully"
    else:
        return "Invalid credentials", 401

@app.route('/logout')
def logout():
    if current_user.is_authenticated:
        logout_user()
        return 'Logged out successfully'
    else:
        return 'You are not logged in', 401

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)
