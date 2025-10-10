from tkinter
import *
from tkinter
import messagebox
import random
import time
import os

class SnakeGame:
    def __init__(self, root):
    self.root = root
self.width = 600
self.height = 600
self.canvas = Canvas(root, width = self.width, height = self.height)
self.canvas.pack()

self.snake = Snake()
self.food = {}

def start_game(self):
    if not os.path.exists('data.json'): #Check
if data.json file exists
with open('data.json', 'w') as f:
    f.write('0')# Create a new file with score set to zero
if it doesn 't exist

self.create_buttons()# Display buttons
for start, pause and quit game
self.snake.start_game(self)# Start snake game
self.root.mainloop()

def create_food(self):
    while True:
    x = random.randint(0, 30) * 20
y = random.randint(0, 30) * 20

if (x, y) not in self.snake.coordinates and\((self.food['x'] - x) ** 2 + (self.food['y'] - y) ** 2) >= 400: #Check food is not too close to snake
self.food = { 'x': x, 'y': y }
break

def create_buttons(self):
    button_start = Button(self.root, text = "Start", command = lambda: self.snake.start_game(self))
button_pause = Button(self.root, text = "Pause", command = lambda: self.snake.pause())
button_quit = Button(self.root, text = "Quit", command = self.snake.game_over)

button_start.pack()
button_pause.pack()
button_quit.pack()

def draw_score(self):
    with open('data.json', 'r') as f:
    score = int(f.read())# Read current score from data.json file

text_score = "Score: {}".format(score)
self.canvas.create_text(10, 20, anchor = NW, text = text_score)# Display current score on the top left of screen

def draw_snake(self):
    for i, coordinate in enumerate(self.snake.coordinates):
    color = "white"
if i == 0
else "grey"#
Head of snake is white, rest are grey
self.canvas.create_rectangle(coordinate[0], coordinate[1],
    coordinate[0] + 20, coordinate[1] + 20, fill = color)

def draw_food(self):
    if self.food: #If food exists, draw it on the canvas
self.canvas.create_oval(self.food['x'], self.food['y'],
    self.food['x'] + 20, self.food['y'] + 20, fill = "red")

def game_over(self): #Method to be called when the game is over
messagebox.showinfo("Game Over", "Your score is {}".format(self.snake.score))
self.root.destroy()