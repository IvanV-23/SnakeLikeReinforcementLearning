import random

import pygame


MAX_STEPS = 200
CELL_SIZE = 30

class NavigationEnv:
    """
    A simple navigation environment for a snake-like agent.
    Actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right
    """

    def __init__(self, width=10, height=10, training=True):
        self.width = width
        self.height = height
        self.training = training
        self.body = []
        self.reset()

    def reset(self):
        self.head_x = self.width // 2
        self.head_y = self.height // 2

        self.food_x = random.randint(0, self.width - 1)
        self.food_y = random.randint(0, self.height - 1)

        self.score = 0
        self.steps = 0
        self.body = []  # <-- Add this line
        return self.get_state()

    def get_state(self):
        # 1) 5x5 window centered on head
        radius = 2
        grid_values = []

        for dy in range(-radius, radius + 1):      # -2,-1,0,1,2
            for dx in range(-radius, radius + 1):  # -2,-1,0,1,2
                x = self.head_x + dx
                y = self.head_y + dy

                # Encode cell
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    val = 1  # wall
                elif (x, y) == (self.head_x, self.head_y):
                    val = 4  # head (optional)
                elif (x, y) in self.body:
                    val = 2  # body
                elif (x, y) == (self.food_x, self.food_y):
                    val = 3  # food
                else:
                    val = 0  # empty

                grid_values.append(val)

        # Normalize to [0,1]
        grid_values = [v / 4.0 for v in grid_values]

        # 2) Your existing scalar features
        dx_food = (self.food_x - self.head_x) / self.width
        dy_food = (self.food_y - self.head_y) / self.height

        danger_up = 1 - (self.head_y / self.height)
        danger_down = 1 - ((self.height - self.head_y - 1) / self.height)
        danger_left = 1 - (self.head_x / self.width)
        danger_right = 1 - ((self.width - self.head_x - 1) / self.width)

        body_up = 1 if (self.head_x, self.head_y - 1) in self.body else 0
        body_down = 1 if (self.head_x, self.head_y + 1) in self.body else 0
        body_left = 1 if (self.head_x - 1, self.head_y) in self.body else 0
        body_right = 1 if (self.head_x + 1, self.head_y) in self.body else 0

        scalar_features = [
            dx_food, dy_food,
            danger_up, danger_down, danger_left, danger_right,
            body_up, body_down, body_left, body_right
        ]

        # 3) Final state: grid + scalars
        return grid_values + scalar_features

    def step(self, action):
        info = {"ate_food": False}
        old_distance = abs(self.head_x - self.food_x) + abs(self.head_y - self.food_y)

        # Save current head position for body following
        prev_head = (self.head_x, self.head_y)

        # Move
        if action == 0:
            self.head_y -= 1
        elif action == 1:
            self.head_y += 1
        elif action == 2:
            self.head_x -= 1
        elif action == 3:
            self.head_x += 1
        else:
            raise ValueError("Invalid action")

        # Update body positions
        if self.body:
            self.body = [prev_head] + self.body[:-1]

        self.steps += 1
        done = False

        if self.training and self.steps >= MAX_STEPS:
            return self.get_state(), -1.0, True, info

        if (
            self.head_x < 0 or self.head_x >= self.width or
            self.head_y < 0 or self.head_y >= self.height
        ):
            return self.get_state(), -1.0, True, info
        # --- NEW: self-collision check ---
        
        if (self.head_x, self.head_y) in self.body:
            # head ran into the body
            return self.get_state(), -1.0, True, info
        # ---------------------------------

        new_distance = abs(self.head_x - self.food_x) + abs(self.head_y - self.food_y)
        delta = old_distance - new_distance

        if delta > 0:
            reward = delta
        else:
            reward = 2 * delta

        reward -= 0.01*self.steps

        # Food eaten
        if self.head_x == self.food_x and self.head_y == self.food_y:
            self.score += 1
            reward += 10.0
            info["ate_food"] = True

            # Add new body segment at the last position (or prev_head if first)
            if self.body:
                self.body.append(self.body[-1])
            else:
                self.body.append(prev_head)

            # Spawn new food
            self.food_x = random.randint(0, self.width - 1)
            self.food_y = random.randint(0, self.height - 1)

        return self.get_state(), reward, done, info

    def render(self):
        print(f"Score: {self.score}")
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if x == self.head_x and y == self.head_y:
                    row += "H"
                elif (x, y) in self.body:
                    row += "B"
                elif x == self.food_x and y == self.food_y:
                    row += "F"
                else:
                    row += "."
            print(row)
        print("-" * self.width)
    def init_pygame(self):
        """Call this once before using render_pygame in a loop."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * CELL_SIZE,
                                               self.height * CELL_SIZE))
        pygame.display.set_caption("Snake-like Navigation")
        self.clock = pygame.time.Clock()

    def close_pygame(self):
        """Call this once when you are done."""
        pygame.quit()

    def render_pygame(self, fps=10):
        """Use this instead of print-based render() inside your loop."""
        # Handle window events so it stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Optional: you can set a flag here if you want to stop external loop
                pass

        self.screen.fill((0, 0, 0))

        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = (20, 20, 20)  # empty

                if x == self.food_x and y == self.food_y:
                    color = (0, 200, 0)      # food
                if (x, y) in self.body:
                    color = (0, 120, 255)    # body
                if x == self.head_x and y == self.head_y:
                    color = (255, 50, 50)    # head

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)
        pygame.display.flip()
        self.clock.tick(fps)
