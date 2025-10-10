class Snake:
    # ...other code...

    def move(self):
        x, y = self.coordinates[0]  # get the first coordinate pair (head of snake)
        
        if self.direction == "right":
            new_x = x + 20  # Assuming each 'square' is 20 pixels wide
            new_y = y     # Keep y position constant
        elif self.direction == "left":
            new_x = x - 20
            new_y = y
        elif self.direction == "up":
            new_x = x
            new_y = y - 20  # Subtract from y-coordinate to go upwards
        else:  # direction is down
            new_x = x
            new_y = y + 20  # Add to y-coordinate to go downwards
        
        self.coordinates.insert(0, (new_x, new_y))  # insert the new coordinate pair at start of coordinates list
        
        if len(self.coordinates) > self.snake_size:  # if snake has grown after moving
            del self.coordinates[-1]  # remove last item in coordinates (old tail)
    
    def create_food(self):
        while True:  # loop until we find a location that isn't occupied by the snake
            x = random.randint(0, 30) * 20  # generate x coordinate within grid width
            y = random.randint(0, 30) * 20  # generate y coordinate within grid height
            
            if (x, y) not in self.coordinates:  # check this location isn't already occupied by snake
                self.food = {'x': x, 'y': y}  # create food at new location
                break  # exit loop as we have found a valid location
