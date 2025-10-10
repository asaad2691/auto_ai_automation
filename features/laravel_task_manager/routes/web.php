<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\TaskController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
*/

Route::get('/', [App\Http\Controllers\HomeController::class, 'index']);
Route::resource('tasks', TaskController::class);
