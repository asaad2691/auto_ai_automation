<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\Task;

class TaskController extends Controller
{
    public function index()
    {
        $tasks = Task::all();
        return view('tasks.index', compact('tasks'));
    }
    
    public function create()
    {
        return view('tasks.create');
    }
    
    public function store(Request $request)
    {
        $validated = $request->validate([
            'title' => 'required|max:255',
            'description' => 'required',
        ]);
        
        Task::create($validated);
        
        return redirect('/tasks')->with('message', 'Task created successfully');
    }
    
    public function show(Task $task)
    {
        return view('tasks.show', compact('task'));
    }
    
    public function edit(Task $task)
    {
        return view('tasks.edit', compact('task'));
    }
    
    public function update(Request $request, Task $task)
    {
        $validated = $request->validate([
            'title' => 'required|max:255',
            'description' => 'required',
        ]);
        
        $task->update($validated);
        
        return redirect('/tasks')->with('message', 'Task updated successfully');
    }
    
    public function destroy(Task $task)
    {
        $task->delete();
        
        return back()->with('message', 'Task deleted successfully');
    }
}
