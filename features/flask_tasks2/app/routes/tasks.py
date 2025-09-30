from flask import render_template, request, redirect, url_for
from app import app

tasks = []  # in-memory list for now

@app.route('/')
def index():
    return render_template('index.html', tasks=tasks)

@app.route('/add', methods=['POST'])
def add_task():
    task = request.form.get('new-task')
    if not task:
        return redirect(url_for('index'))  # ignore empty input
    tasks.append({'name': task, 'done': False})
    return redirect(url_for('index'))

@app.route('/complete', methods=['POST'])
def complete_task():
    idx = int(request.form.get('idx'))  # get index of completed task from form data
    tasks[idx]['done'] = True  # mark the task as done
    return redirect(url_for('index'))
