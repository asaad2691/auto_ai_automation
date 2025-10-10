<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <!-- head content -->
</head>
<body>
    <h1>Task Manager</h1>
    @foreach($tasks as $task)
        <div class="card mb-2">
            <div class="card-header">{{ $task->title }}</div>
            <div class="card-body">
                <p>{{ $task->description }}</p>
                <a href="{{ route('tasks.edit', $task) }}" class="btn btn-primary mr-2">Edit</a>
                <form action="{{ route('tasks.destroy', $task) }}" method="POST" style="display: inline;">
                    @csrf
                    @method("DELETE")
                    <button type="submit" class="btn btn-danger">Delete</button>
                </form>
            </div>
        </div>
    @endforeach
    <a href="{{ route('tasks.create') }}" class="btn btn-success mt-2">Create new task</a>
</body>
</html>
