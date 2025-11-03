import torch
import numpy as np
import random
from env import HangmanEnv
from hmm_model import train_bigram_hmm, hmm_posterior_letter_scores, ALPHABET
from dqn_agent import DQNNet, ReplayBuffer, select_action

# Encode state vector for DQN
def encode_state(obs, trans_logp):
    pattern, guessed, lives = obs["pattern"], obs["guessed"], obs["lives"]
    hmm_scores = hmm_posterior_letter_scores(pattern, guessed, trans_logp)
    hmm_vec = np.array([hmm_scores[a] for a in ALPHABET], dtype=np.float32)
    guessed_vec = np.array([1.0 if a in guessed else 0.0 for a in ALPHABET], dtype=np.float32)
    blanks_left = pattern.count('_') / max(1, len(pattern))
    lives_left = lives / 6.0
    return np.concatenate([hmm_vec, guessed_vec, np.array([blanks_left, lives_left], dtype=np.float32)], axis=0)

# Improved DQN training
def train_dqn(train_words, episodes=4000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trans_logp = train_bigram_hmm(train_words)
    env = HangmanEnv(train_words, lives=6)
    state_dim, action_dim = 26 + 26 + 2, 26

    policy = DQNNet(state_dim, action_dim)
    target = DQNNet(state_dim, action_dim)
    target.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    buf = ReplayBuffer(100000)
    gamma = 0.99

    # Slower epsilon decay
    def epsilon(ep):
        return max(0.1, 1.0 - ep / (episodes * 1.5))

    for ep in range(1, episodes + 1):
        obs = env.reset()
        total_reward = 0
        eps = epsilon(ep)
        steps = 0

        while True:
            s = encode_state(obs, trans_logp)
            a_idx = select_action(policy, s, obs["guessed"], eps, ALPHABET)
            a = ALPHABET[a_idx]
            next_obs, r, done, info = env.step(a)

            # Make correct guesses more rewarding
            if r > 0:
                r += 0.5
            if done and "_" not in next_obs["pattern"]:
                r += 15.0  # big reward for solving the word

            ns = encode_state(next_obs, trans_logp)
            buf.push(s, a_idx, r, ns, done)
            total_reward += r
            steps += 1

            if len(buf) > 512:
                S, A, R, NS, D = buf.sample(128)
                S = torch.from_numpy(S).float()
                A = torch.tensor(A).unsqueeze(1)
                R = torch.tensor(R).float().unsqueeze(1)
                NS = torch.from_numpy(NS).float()
                D = torch.tensor(D).float().unsqueeze(1)

                with torch.no_grad():
                    target_q = target(NS).max(1, keepdim=True)[0]
                    y = R + gamma * (1 - D) * target_q
                q = policy(S).gather(1, A)
                loss = torch.nn.functional.mse_loss(q, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if done:
                break

        if ep % 200 == 0:
            target.load_state_dict(policy.state_dict())
            print(f"[{ep}/{episodes}] Reward={total_reward:.2f} Eps={eps:.3f}")

    torch.save(policy.state_dict(), "dqn_hangman_model.pth")
    print("✅ Training finished — model saved as dqn_hangman_model.pth")

# Larger and slightly more varied training set
train_words = [
    "apple", "banana", "orange", "mango", "pear", "melon", "grape",
    "peach", "berry", "kiwi", "papaya", "plum", "lemon", "lime", "guava"
]

train_dqn(train_words, episodes=2000)

