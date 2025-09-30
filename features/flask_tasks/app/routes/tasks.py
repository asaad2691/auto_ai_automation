from flask import render_template, redirect, url_for, request
from app import app

# Initialize a list to store the tasks in memory
tasks = []

@app.route('/')
def index():
    return render_template('index.html', tasks=tasks)

@app.route('/add_task', methods=['POST'])
def add_task():
    task = request.form.get('task')
    if task:
        tasks.append({'name': task, 'completed': False})
    return redirect(url_for('index'))

@app.route('/complete_task', methods=['POST'])
def complete_task():
    idx = int(request.form.get('idx'))
    if 0 <= idx < len(tasks):
        tasks[idx]['completed'] = True
    return redirect(url_for('index'))
