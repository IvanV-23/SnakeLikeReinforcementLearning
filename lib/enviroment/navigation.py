import random

import pygame
import numpy as np

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

    def reset_cnn(self):
        self.head_x = self.width // 2
        self.head_y = self.height // 2

        self.food_x = random.randint(0, self.width - 1)
        self.food_y = random.randint(0, self.height - 1)

        self.score = 0
        self.steps = 0
        self.body = []  # <-- Add this line
        return self.get_image_state()

    def get_state(self):
        # 1) 5x5 window centered on head (KEEP AS IS)
        radius = 2
        grid_values = []

        for dy in range(-radius, radius + 1):     # -2,-1,0,1,2
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

        # 2) Your existing scalar features (KEEP AS IS)
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

        # 3) NEW: Future danger features (8 new dims)
        danger_straight = [0.0] * 4  # 0=up,1=down,2=left,3=right
        danger_right = [0.0] * 4
        
        directions = [(0,-1), (0,1), (-1,0), (1,0)]  # up,down,left,right
        
        for direction in range(4):
            # Straight ahead (3 cells)
            danger_straight[direction] = 1.0 if self._is_danger(direction, distance=3) else 0.0
            # Right turn ahead (2 cells in right direction)
            right_dir = (direction + 1) % 4
            danger_right[direction] = 1.0 if self._is_danger(right_dir, distance=2) else 0.0
        
        danger_features = danger_straight + danger_right

        # 4) Final state: grid + scalars + NEW dangers
        state = grid_values + scalar_features + danger_features
        return np.array(state, dtype=np.float32)

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
            #self.food_x = random.randint(0, self.width - 1)
            #self.food_y = random.randint(0, self.height - 1)

            occupied = {(self.head_x, self.head_y)} | set(self.body)
            while True:
                self.food_x = random.randint(0, self.width - 1)
                self.food_y = random.randint(0, self.height - 1)
                if (self.food_x, self.food_y) not in occupied:
                    break

        return self.get_state(), reward, done, info

    def cnn_step(self, action):
        info = {"ate_food": False}

        # Distance to food BEFORE moving
        old_distance = abs(self.head_x - self.food_x) + abs(self.head_y - self.food_y)

        prev_head = (self.head_x, self.head_y)

        # --- Move head ---
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

        # --- Move body ---
        if self.body:
            self.body = [prev_head] + self.body[:-1]

        self.steps += 1
        done = False

        # --- Time limit terminal ---
        if self.training and self.steps >= MAX_STEPS:
            return self.get_image_state(), -10.0, True, info

        # --- Wall collision ---
        if (
            self.head_x < 0 or self.head_x >= self.width or
            self.head_y < 0 or self.head_y >= self.height
        ):
            return self.get_image_state(), -1.0, True, info

        # --- Self collision ---
        if (self.head_x, self.head_y) in self.body:
            return self.get_image_state(), -1.0, True, info

        # --------- REWARD -----------
        reward = 0.0

        new_distance = abs(self.head_x - self.food_x) + abs(self.head_y - self.food_y)
        delta = old_distance - new_distance

        if delta > 0:
            reward += 0.2   # moved closer
        elif delta < 0:
            reward -= 0.2  # moved away

        # after the delta-based reward
        reward -= 0.01  # small time cost

        # Food eaten
        if self.head_x == self.food_x and self.head_y == self.food_y:
            self.score += 1
            reward = 5.0 + 0.5*len(self.body)        # big positive event
            info["ate_food"] = True

            # Grow body
            if self.body:
                self.body.append(self.body[-1])
            else:
                self.body.append(prev_head)

            # Spawn new food outside snake
            occupied = {(self.head_x, self.head_y)} | set(self.body)
            while True:
                self.food_x = random.randint(0, self.width - 1)
                self.food_y = random.randint(0, self.height - 1)
                if (self.food_x, self.food_y) not in occupied:
                    break
        # ------------------------------------

        return self.get_image_state(), reward, done, info

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

        font = pygame.font.Font(None, 36)  # Use default font, size 36
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(topleft=(10, 10))  # Top-left corner
        self.screen.blit(score_text, score_rect)

        pygame.display.flip()
        self.clock.tick(fps)

    def get_image_state(self, size=84):  # 84x84 = Atari standard
        """Return image instead of vector state for CNN input."""
        surface = pygame.Surface((self.width * CELL_SIZE, self.height * CELL_SIZE))
        
        # Render grid to surface (same logic as render_pygame)
        surface.fill((0, 0, 0))  # black background
        
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = (20, 20, 20)  # empty (dark gray)
                
                if x == self.food_x and y == self.food_y:
                    color = (0, 255, 0)      # food (bright green)
                elif (x, y) in self.body:
                    color = (0, 120, 255)    # body (blue)
                elif x == self.head_x and y == self.head_y:
                    color = (255, 50, 50)    # head (red)
                
                pygame.draw.rect(surface, color, rect)
        
        # Resize to 84x84 (standard CNN input)
        image = pygame.transform.scale(surface, (size, size))
        
        # Convert to numpy array → normalize [0,1]
        img_array = pygame.surfarray.array3d(image).swapaxes(0,1)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW for PyTorch
        
        return img_array  # shape: (3, 84, 84)


    def _is_danger(self, direction, distance):
        dx, dy = [(0,-1),(0,1),(-1,0),(1,0)][direction]
        x, y = self.head_x + dx * distance, self.head_y + dy * distance
        return (x < 0 or x >= self.width or y < 0 or y >= self.height or 
                (x, y) in self.body)
