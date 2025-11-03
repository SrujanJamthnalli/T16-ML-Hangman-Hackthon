import torch
import numpy as np
from env import HangmanEnv
from hmm_model import train_bigram_hmm, hmm_posterior_letter_scores, ALPHABET
from dqn_agent import DQNNet

# --- Helper to encode the game state ---
def encode_state(obs, trans_logp):
    pattern, guessed, lives = obs["pattern"], obs["guessed"], obs["lives"]
    hmm_scores = hmm_posterior_letter_scores(pattern, guessed, trans_logp)
    hmm_vec = np.array([hmm_scores[a] for a in ALPHABET], dtype=np.float32)
    guessed_vec = np.array([1.0 if a in guessed else 0.0 for a in ALPHABET], dtype=np.float32)
    blanks_left = pattern.count('_') / max(1, len(pattern))
    lives_left = lives / 6.0
    return np.concatenate([hmm_vec, guessed_vec, np.array([blanks_left, lives_left], dtype=np.float32)], axis=0)

# --- Simulate one complete game ---
def simulate_game(word="mango", model_path="dqn_hangman_model.pth", vocab=None):
    print("🎯 Starting Hangman Simulation")
    print(f"Target Word: {word}")

    # Prepare environment and model
    vocab = vocab or ["apple", "banana", "orange", "mango", "pear", "melon", "grape", "peach", "berry", "kiwi","python","network"]
    trans_logp = train_bigram_hmm(vocab)
    env = HangmanEnv([word])
    net = DQNNet(54, 26)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    obs = env.reset(word)
    steps = 0
    print(f"Initial Pattern: {obs['pattern']} | Lives: {obs['lives']}")

    while True:
        s = encode_state(obs, trans_logp)
        with torch.no_grad():
            q = net(torch.from_numpy(s).unsqueeze(0).float()).squeeze(0).numpy()
        for i, a in enumerate(ALPHABET):
            if a in obs["guessed"]:
                q[i] = -1e9
        a_idx = int(np.argmax(q))
        a = ALPHABET[a_idx]
        obs, r, done, info = env.step(a)
        steps += 1
        print(f"Step {steps}: Guessed '{a}' | Pattern: {obs['pattern']} | Lives: {obs['lives']} | Reward: {r:.2f}")
        if done:
            if "_" not in obs["pattern"]:
                print(f"✅ The agent WON in {steps} steps! Word: {env.word}")
            else:
                print(f"❌ The agent LOST. Word was: {env.word}")
            break

# --- Run a test simulation ---
simulate_game(word="python")