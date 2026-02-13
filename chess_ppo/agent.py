import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from network import ppo_network
from buffer import buffer
from encoder import board_encoder
from env import chess_env
import chess


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ppo implementation based on openai spinning up
# https://spinningup.openai.com/en/latest/algorithms/ppo.html
# https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
# change: multiprocessing for parallel games - see parallel_self_play() in main.py
class ppo_agent:
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.model = ppo_network().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = buffer()
    
    def get_action(self, board, temperature=1.0):
        state = board_encoder.encode(board)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy_logits, value = self.model(state_t)
        
        policy = torch.softmax(policy_logits, dim=-1)
        
        moves = list(board.legal_moves)
        if not moves:
            return None, 0, 0
        
        probs = policy[0, :len(moves)].cpu().numpy()
        probs = np.clip(probs, 1e-10, 1)
        probs = probs ** (1/temperature)
        probs = probs / probs.sum()
        
        if temperature > 0.5:
            action_idx = np.random.choice(len(moves), p=probs)
        else:
            action_idx = np.argmax(probs)
        
        move = moves[action_idx]
        log_prob = np.log(probs[action_idx] + 1e-10)
        
        return move, value.item(), log_prob
    
    def select_best_move(self, board):
        state = board_encoder.encode(board)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(state_t)
        
        policy = torch.softmax(policy_logits, dim=-1)
        
        moves = list(board.legal_moves)
        if not moves:
            return None
        
        probs = policy[0, :len(moves)].cpu().numpy()
        best_idx = np.argmax(probs)
        
        return moves[best_idx]
    
    def compute_returns(self, rewards, dones):
        returns = []
        running_return = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            running_return = r + self.gamma * running_return * (1 - done)
            returns.insert(0, running_return)
        return torch.FloatTensor(returns).to(device)
    
    def update(self):
        if len(self.buffer.states) == 0:
            return
        
        states = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        actions = torch.LongTensor(self.buffer.actions).to(device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(device)
        returns = self.compute_returns(self.buffer.rewards, self.buffer.dones)
        
        old_values = torch.FloatTensor(self.buffer.values).to(device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.k_epochs):
            policy_logits, values = self.model(states)
            
            dist = torch.softmax(policy_logits, dim=-1)
            log_probs = torch.log(dist + 1e-10)
            
            new_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            # clipped surrogate objective prevents large policy updates
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping for stability
            # https://stackoverflow.com/questions/60018578/what-does-grad-norm-do-in-pytorch
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
        
        self.buffer.clear()


def play_game(agent_white, agent_black, temperature=0.5):
    env = chess_env()
    state = env.reset()
    done = False
    
    while not done:
        if env.board.turn == chess.WHITE:
            move, value, log_prob = agent_white.get_action(env.board, temperature)
        else:
            move, value, log_prob = agent_black.get_action(env.board, temperature)
        
        if move is None:
            break
        
        agent_white.buffer.add(state, 0, 0, value, log_prob, 0)
        
        state, reward, done = env.step(move)
    
    final_reward = 0
    if env.board.is_game_over():
        result = env.board.result()
        if result == "1-0":
            final_reward = 1
        elif result == "0-1":
            final_reward = -1
    
    return final_reward


def self_play(agent, num_games=100):
    results = {"wins": 0, "losses": 0, "draws": 0}
    
    for game in range(num_games):
        result = play_game(agent, agent, temperature=0.7)
        
        for i in range(len(agent.buffer.rewards)):
            if i < len(agent.buffer.rewards) - 1:
                agent.buffer.rewards[i] = 0
            else:
                agent.buffer.rewards[i] = result
        
        agent.update()
        
        if result == 1:
            results["wins"] += 1
        elif result == -1:
            results["losses"] += 1
        else:
            results["draws"] += 1
        
        if (game + 1) % 10 == 0:
            print(f"game {game + 1}/{num_games} | wins {results['wins']} losses {results['losses']} draws {results['draws']}")
    
    return results


def evaluate(agent, num_games=20):
    results = {"wins": 0, "losses": 0, "draws": 0}
    random_agent = ppo_agent()
    
    for game in range(num_games):
        result = play_game(agent, random_agent, temperature=0.0)
        
        if result == 1:
            results["wins"] += 1
        elif result == -1:
            results["losses"] += 1
        else:
            results["draws"] += 1
    
    return results


def save_agent(agent, path):
    torch.save(agent.model.state_dict(), path)
    print(f"saved to {path}")


def load_agent(agent, path):
    agent.model.load_state_dict(torch.load(path))
    print(f"loaded from {path}")
