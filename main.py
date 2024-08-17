import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import TPUSpawnStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TPUStatsMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque, namedtuple

# Ensure TPU is available
assert 'COLAB_TPU_ADDR' in os.environ, 'ERROR: Not connected to a TPU runtime. Please select TPU from Runtime > Change runtime type.'

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class BullsAndCowsGame:
    def __init__(self, code_length=4, num_characters=10):
        self.code_length = code_length
        self.num_characters = num_characters
        self.valid_characters = '0123456789'[:num_characters]
        self.reset()

    def reset(self):
        self.code = ''.join(random.choice(self.valid_characters) for _ in range(self.code_length))
        self.attempts = 0
        return '0' * self.code_length

    def step(self, guess):
        self.attempts += 1
        feedback = self._get_feedback(guess)
        done = feedback == f"{self.code_length}b0c"
        reward = self._calculate_reward(feedback)
        return feedback, reward, done

    def _get_feedback(self, guess):
        bulls = sum(g == c for g, c in zip(guess, self.code))
        cows = sum(g in self.code for g in guess) - bulls
        return f"{bulls}b{cows}c"

    def _calculate_reward(self, feedback):
        bulls, cows = map(int, feedback.replace('b', ' ').replace('c', ' ').split())
        base_reward = bulls * 0.5 + cows * 2
        if bulls == self.code_length:
            reward_scale = [40, 30, 20, 10, -25]
            bonus = next((reward for attempts, reward in zip([5, 10, 15, 25, float('inf')], reward_scale) if self.attempts <= attempts), 0)
            base_reward += bonus
        return base_reward

class BullsAndCowsEnvironment:
    def __init__(self, code_length=4, num_characters=10, max_attempts=100):
        self.code_length = code_length
        self.num_characters = num_characters
        self.max_attempts = max_attempts
        self.game = BullsAndCowsGame(code_length, num_characters)
        self.state_cache = {}  # Cache to store precomputed states

    def encode_state(self, guess, feedback):
        if (guess, feedback) in self.state_cache:
            return self.state_cache[(guess, feedback)]

        state = torch.zeros(self.code_length * self.num_characters + 3, dtype=torch.float32)  # +3 for normalized data
        for i, g in enumerate(guess):
            state[i * self.num_characters + int(g)] = 1
        bulls, cows = map(int, feedback.replace('b', ' ').replace('c', ' ').split())
        state[-3] = bulls / self.code_length
        state[-2] = cows / self.code_length
        state[-1] = self.game.attempts / self.max_attempts

        # Cache the computed state
        self.state_cache[(guess, feedback)] = state
        return state

    def step(self, action, current_guess):
        position = action // self.num_characters
        new_char = self.decode_action(action)
        new_guess = current_guess[:position] + new_char + current_guess[position + 1:]
        feedback, reward, done = self.game.step(new_guess)
        done = done or self.game.attempts >= self.max_attempts
        return new_guess, feedback, reward, done, self.game.attempts

    def decode_action(self, action):
        return str(action % self.num_characters)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        max_priority = np.max(self.priorities) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            probs = self.priorities ** self.alpha
            probs /= probs.sum()
        else:
            probs = self.priorities[:len(self.buffer)] ** self.alpha
            probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()
        self.beta = np.min([1., self.beta + self.beta_increment])

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class DistributionalDQN(nn.Module):
    def __init__(self, state_size, action_size, n_atoms=51, v_min=-10, v_max=10):
        super(DistributionalDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size * n_atoms)
        )

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        value = self.value_stream(x).view(-1, 1, self.n_atoms)
        advantage = self.advantage_stream(x).view(-1, self.action_size, self.n_atoms)

        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(q_dist, dim=-1)


class BullsAndCowsDataset(Dataset):
    def __init__(self, experiences, weights):
        self.states = torch.tensor([e.state for e in experiences], dtype=torch.float32)
        self.actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        self.rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        self.next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32)
        self.dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], self.weights[idx])


class BullsAndCowsAgent(pl.LightningModule):
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.save_hyperparameters()
        self.q_network = DistributionalDQN(state_size, action_size, n_atoms, v_min, v_max)
        self.target_network = DistributionalDQN(state_size, action_size, n_atoms, v_min, v_max)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.replay_buffer = PrioritizedReplayBuffer()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.register_buffer("z", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

    def forward(self, x):
        return self.q_network(x)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.hparams.action_size - 1)
        with torch.no_grad():
            q_dist = self(state.to(self.device))
            q_values = (q_dist * self.z).sum(dim=-1)
            return q_values.argmax().item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones, weights = batch
        states, actions, rewards, next_states, dones, weights = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device),
            weights.to(self.device)
        )

        with torch.no_grad():
            next_q_dist = self.target_network(next_states)
            next_q_values = (next_q_dist * self.z).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=-1)
            next_q_dist = next_q_dist[range(states.shape[0]), next_actions]

        tz = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * self.hparams.gamma * self.z.unsqueeze(0)
        tz = tz.clamp(self.hparams.v_min, self.hparams.v_max)
        b = (tz - self.hparams.v_min) / self.delta_z
        l, u = b.floor().long(), b.ceil().long()

        target_dist = torch.zeros_like(next_q_dist)
        offset = torch.linspace(0, (states.shape[0] - 1) * self.hparams.n_atoms, states.shape[0]).long().unsqueeze(1).expand(states.shape[0], self.hparams.n_atoms).to(self.device)
        target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist * (u.float() - b)).view(-1))
        target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist * (b - l.float())).view(-1))

        current_q_dist = self.q_network(states)[range(states.shape[0]), actions]
        loss = -(target_dist * torch.log(current_q_dist.clamp(min=1e-8))).sum(dim=-1)
        weighted_loss = (loss * weights).mean()

        self.log('train_loss', weighted_loss)
        return weighted_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                'monitor': 'train_loss',
                'frequency': 1
            }
        }

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def on_train_epoch_end(self):
        self.update_target_network()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.log('epsilon', self.epsilon)

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


class SelfPlayTrainer(pl.LightningModule):
    def __init__(self, code_length=4, num_characters=10, batch_size=64, num_episodes=50000, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        state_size = code_length * (num_characters + 3)  # plus three for normalized data
        action_size = num_characters * code_length
        self.agent = BullsAndCowsAgent(state_size, action_size, learning_rate=learning_rate)
        self.env = BullsAndCowsEnvironment(code_length, num_characters)
        self.performance_window = deque(maxlen=100)
        self.episode_count = 0

    def train_dataloader(self):
        if len(self.agent.replay_buffer) < self.hparams.batch_size:
            return None  # Avoid training on an insufficiently filled buffer
        experiences, indices, weights = self.agent.replay_buffer.sample(self.hparams.batch_size)
        return DataLoader(BullsAndCowsDataset(experiences, weights), batch_size=self.hparams.batch_size, num_workers=4)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return
        loss = self.agent.training_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if len(self.agent.replay_buffer) >= self.hparams.batch_size:
            self.run_self_play_episodes()

        return loss

    def run_self_play_episodes(self):
        while len(self.agent.replay_buffer) < self.hparams.batch_size and self.episode_count < self.hparams.num_episodes:
            attempts, total_reward = self.self_play_episode()
            self.episode_count += 1
            self.performance_window.append(attempts)
            if self.episode_count % 100 == 0:
                self.log_metrics()

    def log_metrics(self):
        avg_attempts = sum(self.performance_window) / len(self.performance_window)
        self.log('avg_attempts', avg_attempts)
        self.log('epsilon', self.agent.epsilon)
        print(f"Episode: {self.episode_count}, Avg Attempts: {avg_attempts:.2f}, Epsilon: {self.agent.epsilon:.2f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def self_play_episode(self):
        guess = self.env.game.reset()
        state = self.env.encode_state(guess, '0b0c')
        done = False
        total_reward = 0
        attempts = 0

        while not done:
            action = self.agent.act(state.unsqueeze(0))
            guess, feedback, reward, done, attempts = self.env.step(action, guess)
            next_state = self.env.encode_state(guess, feedback)
            self.agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        return attempts, total_reward

    def test(self, num_games=100):
        total_attempts = 0
        total_rewards = 0
        success_rate = 0

        for _ in range(num_games):
            guess = self.env.game.reset()
            state = self.env.encode_state(guess, '0b0c')
            done = False
            game_reward = 0
            attempts = 0

            while not done:
                action = self.agent.act(state.unsqueeze(0))
                guess, feedback, reward, done, attempts = self.env.step(action, guess)
                state = self.env.encode_state(guess, feedback)
                game_reward += reward

            total_attempts += attempts
            total_rewards += game_reward
            if attempts <= 10:
                success_rate += 1

        avg_attempts = total_attempts / num_games
        avg_rewards = total_rewards / num_games
        success_rate_percent = (success_rate / num_games) * 100

        self.log('test_avg_attempts', avg_attempts)
        self.log('test_avg_rewards', avg_rewards)
        self.log('test_success_rate', success_rate_percent)

        print(f"\nTest Results:")
        print(f"Average attempts: {avg_attempts:.2f}")
        print(f"Average rewards: {avg_rewards:.2f}")
        print(f"Success rate (≤10 attempts): {success_rate_percent:.2f}%")

        return avg_attempts, avg_rewards, success_rate_percent


def main():
    pl.seed_everything(42)

    model = SelfPlayTrainer(batch_size=64, num_episodes=50000, learning_rate=1e-3)
    tpu_stats_monitor = TPUStatsMonitor()

    trainer = pl.Trainer(
        accelerator='tpu',
        devices=8,
        max_epochs=-1,  # We'll stop based on num_episodes, not epochs
        strategy=TPUSpawnStrategy(),  # Use TPUSpawnStrategy for TPU with multiple cores
        callbacks=[
            EarlyStopping(monitor='train_loss', patience=20),
            ModelCheckpoint(monitor='train_loss', save_top_k=3),
            LearningRateMonitor(),
            tpu_stats_monitor
        ],
        logger=CSVLogger(save_dir='logs/'),
        gradient_clip_val=1.0,
    )

    trainer.fit(model)

    # Load best model and run final test
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = SelfPlayTrainer.load_from_checkpoint(best_model_path)
    best_model.test(num_games=100)


if __name__ == "__main__":
    main()
