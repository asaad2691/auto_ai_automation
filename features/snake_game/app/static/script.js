// Add your JavaScript logic for the Snake game here
var canvas = document.getElementById('gameCanvas');
var context = canvas.getContext('2d');
var score = 0;
var scoreElement = document.getElementById('score');
scoreElement.textContent = 'Score: ' + score;
