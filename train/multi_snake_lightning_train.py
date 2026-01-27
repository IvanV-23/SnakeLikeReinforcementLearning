import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import random

# Import your classes here
from train.multi_snake_lightning import  MultiSnakeLightning
from lib.enviroment.multy_snake_env import MultiSnakeEnv
from lib.net.multi_agent_conv_nav_net import ConvNavNet

def train_snake():
    # 1. Configuration
    MEM_SIZE = 100_000
    WARM_UP_STEPS = 5000
    TOTAL_STEPS = 1_000_000
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 200_000 # Higher = slower decay
    
    # 2. Initialize Components
    env = MultiSnakeEnv(n_agents=2, width=10, height=10)
    model = ConvNavNet(n_actions=4)
    agent = MultiSnakeLightning(
        model=model, 
        lr=1e-4, 
        batch_size=128, 
        target_update_freq=1000
    )
    
    # Setup Logger
    logger = TensorBoardLogger("logs/", name="snake_ai_industrial")
    
    # 3. Trainer Setup
    # Note: we use limit_train_batches to control how many gradient steps per loop
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_steps=TOTAL_STEPS,
        logger=logger,
        enable_checkpointing=True
    )

    # 4. Main Environment Loop
    obs = env.reset_multi()
    global_step = 0
    
    print("Starting Warm-up Phase...")
    
    while global_step < TOTAL_STEPS:
        # Calculate Epsilon Decay
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1. * global_step / EPS_DECAY)
        agent.epsilon = epsilon # Pass to agent for logging
        
        # Action Selection (Epsilon-Greedy)
        actions = {}
        for i in range(env.n_agents):
            key = f"agent_{i}"
            if key not in obs: continue # Skip dead agents
            
            if random.random() < epsilon:
                actions[key] = random.randrange(4)
            else:
                # Use model for inference
                state_tensor = torch.FloatTensor(obs[key]).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_values = agent.model(state_tensor)
                    actions[key] = q_values.argmax().item()
        
        # Step the Environment
        next_obs, reward, done, info = env.step_multi(actions)
        
        # Store Transitions for each agent
        for i in range(env.n_agents):
            key = f"agent_{i}"
            if key in obs and key in next_obs:
                agent.buffer.push(
                    obs[key], 
                    actions[key], 
                    next_obs[key], 
                    reward, 
                    float(done)
                )
        
        obs = next_obs
        global_step += 1
        
        # Reset if game ends
        if done:
            obs = env.reset_multi()
        
        # 5. Training Phase (Start after warm-up)
        if global_step == WARM_UP_STEPS:
            print("Warm-up complete. Training started...")
            
        if global_step > WARM_UP_STEPS and global_step % 4 == 0:
            # Run one training step (Lightning will pull from DataLoader/Buffer)
            trainer.fit(agent) 
            
        # Logging progress
        if global_step % 1000 == 0:
            print(f"Step: {global_step} | Epsilon: {epsilon:.3f} | Buffer: {len(agent.buffer)}")


        torch.save(agent.model.state_dict(), "final_snake_model.pth")        
        print("Model saved as final_snake_model.pth")

if __name__ == "__main__":
    train_snake()