import numpy as np
import random
import pygame

from lib.enviroment.navigation import NavigationEnv, CELL_SIZE, MAX_STEPS

class MultiSnakeEnv(NavigationEnv):
    """
    Multi-snake cooperative environment on a single grid.

    - n_agents snakes move simultaneously.
    - Each snake has its own head/body, but share:
        * food position
        * global step counter
        * team reward
    - step_multi(actions) operates on all snakes at once
      and returns: obs_dict, shared_reward, done, info.
    """

    def __init__(self, n_agents=2, **kwargs):
        super().__init__(**kwargs)  # sets width, height, etc.
        self.n_agents = n_agents
        self.max_steps = kwargs.get("max_steps", MAX_STEPS)
        # disable single-snake body/head in parent; we will manage per-agent
        self.snakes = None

    # ---------- internal helpers ----------

    def _init_snakes(self):
        """Initialize n_agents snakes at different positions."""
        self.snakes = []
        occupied = set()

        for i in range(self.n_agents):
            while True:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if (x, y) not in occupied:
                    break
            snake = {
                "alive": True,
                "head_x": x,
                "head_y": y,
                "body": [],
                "score": 0,
            }
            self.snakes.append(snake)
            occupied.add((x, y))

        # spawn food not on any snake
        self._spawn_food(occupied)

        self.steps = 0

    def _spawn_food(self, forbidden):
        while True:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            if (fx, fy) not in forbidden:
                self.food_x = fx
                self.food_y = fy
                break

    def _action_to_delta(self, a):
        # same convention as parent: 0=up,1=down,2=left,3=right
        if a == 0:
            return 0, -1
        if a == 1:
            return 0, 1
        if a == 2:
            return -1, 0
        if a == 3:
            return 1, 0
        raise ValueError("Invalid action in MultiSnakeEnv")

    def _build_obs_for_agent(self, idx, size=84):
        """
        Reuse your get_image_state logic but for 'view from agent idx'.
        We render a synthetic frame: one head/body is 'self' color,
        others are 'teammate' color.
        """
        # Create a temporary surface via pygame just like get_image_state
        surface = pygame.Surface((self.width * CELL_SIZE,
                                  self.height * CELL_SIZE))
        surface.fill((0, 0, 0))

        # draw food
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)

                color = (20, 20, 20)  # empty

                if x == self.food_x and y == self.food_y:
                    color = (0, 255, 0)  # food

                # draw bodies / heads
                for j, s in enumerate(self.snakes):
                    if not s["alive"]:
                        continue
                    # choose color depending on self vs teammates
                    if (x, y) == (s["head_x"], s["head_y"]):
                        if j == idx:
                            color = (255, 50, 50)   # self head (red)
                        else:
                            color = (255, 200, 50)  # teammate head (yellow)
                    elif (x, y) in s["body"]:
                        if j == idx:
                            color = (0, 120, 255)   # self body (blue)
                        else:
                            color = (0, 80, 160)    # teammate body (darker blue)

                pygame.draw.rect(surface, color, rect)

        image = pygame.transform.scale(surface, (size, size))
        img_array = pygame.surfarray.array3d(image).swapaxes(0, 1)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # CHW
        return img_array

    # ---------- API ----------

    def reset_multi(self):
        """
        Reset multi-snake env and return dict of observations:
        { 'agent_0': img, 'agent_1': img, ... }
        """
        self._init_snakes()
        obs = {}
        for i in range(self.n_agents):
            obs[f"agent_{i}"] = self._build_obs_for_agent(i)
        return obs

    def step_multi(self, actions):
        """
        Improved rewards for cooperative multi-snake:
        - Sparse: +5 food (team), -5 death (team)
        - Shaping: distance to food per snake, survival bonus
        - Penalty: -0.01 per step (encourages efficiency)
        """
        self.steps += 1
        info = {"ate_food": [False] * self.n_agents}
        shared_reward = 0.0
        done = False

        # 1) Current occupied cells
        occupied_now = set()
        for s in self.snakes:
            if not s["alive"]:
                continue
            occupied_now.add((s["head_x"], s["head_y"]))
            for b in s["body"]:
                occupied_now.add(b)

        # 2) Next head positions
        next_heads = []
        old_distances = []  # for shaping
        for i, s in enumerate(self.snakes):
            if not s["alive"]:
                next_heads.append(None)
                old_distances.append(0.0)
                continue

            if isinstance(actions, dict):
                a = actions.get(f"agent_{i}", 0)
            else:
                a = actions[i]

            dx, dy = self._action_to_delta(a)
            nx = s["head_x"] + dx
            ny = s["head_y"] + dy
            next_heads.append((nx, ny))
            
            # Distance shaping baseline
            old_dist = abs(s["head_x"] - self.food_x) + abs(s["head_y"] - self.food_y)
            old_distances.append(old_dist)

        # 3) Collision detection
        occupied_next = {}
        for i, pos in enumerate(next_heads):
            if pos is None:
                continue
            occupied_next.setdefault(pos, []).append(i)

        for i, s in enumerate(self.snakes):
            if not s["alive"]:
                continue
            pos = next_heads[i]
            x, y = pos

            hit_wall = (x < 0 or x >= self.width or y < 0 or y >= self.height)
            hit_body = (pos in occupied_now)
            head_on = (len(occupied_next.get(pos, [])) > 1)

            if hit_wall or hit_body or head_on:
                s["alive"] = False
                shared_reward -= 1.0  # BIG team penalty for death
                done = True

        # 4) Move surviving snakes + food + SHAPING
        ate_food = False
        for i, s in enumerate(self.snakes):
            if not s["alive"]:
                continue

            old_head = (s["head_x"], s["head_y"])
            new_head = next_heads[i]
            s["head_x"], s["head_y"] = new_head

            # Distance shaping reward
            old_dist = old_distances[i]
            new_dist = abs(s["head_x"] - self.food_x) + abs(s["head_y"] - self.food_y)
            delta = old_dist - new_dist
            if delta > 0:
                shared_reward += 0.05 # small bonus for getting closer
            else:
                shared_reward -= 0.05  #+ self.score*0.1 # small penalty for getting farther
                pass
            # Body update
            s["body"].insert(0, old_head)
            if new_head == (self.food_x, self.food_y):
                ate_food = True
                info["ate_food"][i] = True
                s["score"] += 1
                # grow (don't pop tail)
            else:
                if s["body"]:
                    s["body"].pop()

        if ate_food:
            self.score += 1
            score_reward = 10.0 + self.score
            shared_reward += score_reward  # BIG team bonus (improved by total score)
            occ = set()
            for s in self.snakes:
                if not s["alive"]: continue
                occ.add((s["head_x"], s["head_y"]))
                for b in s["body"]:
                    occ.add(b)
            self._spawn_food(occ)

        # 5) SURVIVAL BONUS (encourages staying alive)
        alive_count = sum(1 for s in self.snakes if s["alive"])
        #shared_reward += 0.02 * alive_count  # +0.02 per alive snake per step

        # 6) STEP PENALTY (encourages efficiency)
        shared_reward -= 0.01 * self.steps  # small cost per step

        # 7) Build observations
        obs = {}
        for i in range(self.n_agents):
            obs[f"agent_{i}"] = self._build_obs_for_agent(i)

        # 8) Termination
        all_dead = not any(s["alive"] for s in self.snakes)
        time_up = self.steps >= self.max_steps
        if time_up:
            shared_reward -= 100.0  # BIG penalty for not finishing
            done = True 
        
        if done:
            self.score = 0

        return obs, shared_reward, done, info

    def render_pygame(self, fps=10):
        """Multi-snake pygame render (one color per agent)."""
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass  # you can set a flag here if desired

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Base grid background
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, (20, 20, 20), rect)         # empty
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)      # grid lines

        # Draw food
        food_rect = pygame.Rect(self.food_x * CELL_SIZE,
                                self.food_y * CELL_SIZE,
                                CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, (0, 200, 0), food_rect)            # food

        # Colors per agent (extend if you add more agents)
        head_colors = [
            (255, 50, 50),   # agent 0 head - red
            (255, 200, 50),  # agent 1 head - yellow
            (50, 255, 50),   # agent 2 head - green
            (50, 50, 255),   # agent 3 head - blue
        ]
        body_colors = [
            (0, 120, 255),   # agent 0 body
            (0, 80, 160),    # agent 1 body
            (0, 160, 80),    # agent 2 body
            (80, 80, 255),   # agent 3 body
        ]

        # Draw each snake
        for i, snake in enumerate(self.snakes):
            if not snake["alive"]:
                continue

            h_color = head_colors[i % len(head_colors)]
            b_color = body_colors[i % len(body_colors)]

            # Head
            hx, hy = snake["head_x"], snake["head_y"]
            rect = pygame.Rect(hx * CELL_SIZE, hy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, h_color, rect)

            # Body
            for bx, by in snake["body"]:
                brect = pygame.Rect(bx * CELL_SIZE, by * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, b_color, brect)

        # Text HUD: show each agent score + status
        font = pygame.font.Font(None, 28)
        y_text = 10
        for i, snake in enumerate(self.snakes):
            score = snake.get("score", 0)
            status = "alive" if snake["alive"] else "dead"
            text = font.render(f"Agent {i} - Score: {score} ({status})",
                            True, (255, 255, 255))
            self.screen.blit(text, (10, y_text))
            y_text += 24

        pygame.display.flip()
        self.clock.tick(fps)