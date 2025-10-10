from tkinter import *
from tkinter import messagebox
import random
import os

CELL_SIZE = 20
GRID_SIZE = 30  # 30x30 grid
GAME_SPEED = 100  # milliseconds

class Snake:
    def __init__(self):
        self.snake_size = 1
        self.coordinates = [(0, 0)]
        self.direction = "right"
        self.score = 0
        self.high_score = 0
        self.food = {}

    def move(self):
        x, y = self.coordinates[0]
        if self.direction == "right":
            x += CELL_SIZE
        elif self.direction == "left":
            x -= CELL_SIZE
        elif self.direction == "up":
            y -= CELL_SIZE
        elif self.direction == "down":
            y += CELL_SIZE

        new_head = (x, y)
        self.coordinates.insert(0, new_head)
        if len(self.coordinates) > self.snake_size:
            self.coordinates.pop()

    def check_collision(self):
        x, y = self.coordinates[0]
        if x < 0 or x >= GRID_SIZE * CELL_SIZE or y < 0 or y >= GRID_SIZE * CELL_SIZE:
            return True
        if (x, y) in self.coordinates[1:]:
            return True
        return False

    def check_food(self):
        return self.food and (self.coordinates[0][0] == self.food['x'] and self.coordinates[0][1] == self.food['y'])

    def create_food(self):
        while True:
            x = random.randint(0, GRID_SIZE - 1) * CELL_SIZE
            y = random.randint(0, GRID_SIZE - 1) * CELL_SIZE
            if (x, y) not in self.coordinates:
                self.food = {'x': x, 'y': y}
                break

class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.width = GRID_SIZE * CELL_SIZE
        self.height = GRID_SIZE * CELL_SIZE
        self.canvas = Canvas(root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        self.snake = Snake()
        self.running = False

        self.create_buttons()
        self.root.bind("<Key>", self.change_direction)

    def create_buttons(self):
        button_start = Button(self.root, text="Start", command=self.start_game)
        button_quit = Button(self.root, text="Quit", command=self.root.destroy)
        button_start.pack()
        button_quit.pack()

    def start_game(self):
        self.running = True
        self.snake = Snake()
        self.snake.create_food()
        self.update_game()

    def update_game(self):
        if not self.running:
            return

        self.snake.move()

        if self.snake.check_collision():
            self.game_over()
            return

        if self.snake.check_food():
            self.snake.snake_size += 1
            self.snake.score += 1
            self.snake.create_food()

        self.canvas.delete("all")
        self.draw_snake()
        self.draw_food()
        self.draw_score()

        self.root.after(GAME_SPEED, self.update_game)

    def draw_snake(self):
        for i, (x, y) in enumerate(self.snake.coordinates):
            color = "white" if i == 0 else "gray"
            self.canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE, fill=color)

    def draw_food(self):
        f = self.snake.food
        self.canvas.create_oval(f['x'], f['y'], f['x']+CELL_SIZE, f['y']+CELL_SIZE, fill="red")

    def draw_score(self):
        self.canvas.create_text(10, 10, anchor=NW, fill="white", text=f"Score: {self.snake.score}")

    def change_direction(self, event):
        key = event.keysym
        current = self.snake.direction
        if key == "Up" and current != "down":
            self.snake.direction = "up"
        elif key == "Down" and current != "up":
            self.snake.direction = "down"
        elif key == "Left" and current != "right":
            self.snake.direction = "left"
        elif key == "Right" and current != "left":
            self.snake.direction = "right"

    def game_over(self):
        self.running = False
        messagebox.showinfo("Game Over", f"Your score: {self.snake.score}")

root = Tk()
root.title("Snake Game")
game = SnakeGame(root)
root.mainloop()
