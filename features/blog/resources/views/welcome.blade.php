<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="UTF-8" name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Laravel</title>
        <!-- Fonts -->
        <link rel="stylesheet" href="{{ asset('css/app.css') }}">
    </head>
    <body class="antialiased text-gray-700 font-serif">
        <div id="app"></div>
        <script src="{{ asset('js/app.js') }}"></script>
    </body>
</html>
