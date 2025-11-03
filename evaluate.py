import torch, json
from env import HangmanEnv
from hmm_model import train_bigram_hmm, hmm_posterior_letter_scores, ALPHABET
from dqn_agent import DQNNet
import numpy as np

def encode_state(obs, trans_logp):
    pattern, guessed, lives = obs["pattern"], obs["guessed"], obs["lives"]
    hmm_scores = hmm_posterior_letter_scores(pattern, guessed, trans_logp)
    hmm_vec = np.array([hmm_scores[a] for a in ALPHABET], dtype=np.float32)
    guessed_vec = np.array([1.0 if a in guessed else 0.0 for a in ALPHABET], dtype=np.float32)
    blanks_left = pattern.count('_') / max(1, len(pattern))
    lives_left = lives / 6.0
    return np.concatenate([hmm_vec, guessed_vec, np.array([blanks_left, lives_left], dtype=np.float32)], axis=0)

def evaluate_model(words, model_path="dqn_hangman_model.pth", games=2000):
    trans_logp = train_bigram_hmm(words)
    env = HangmanEnv(words)
    net = DQNNet(54, 26)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    wins, total_wrong, total_repeat = 0, 0, 0
    for _ in range(games):
        obs = env.reset()
        while True:
            s = encode_state(obs, trans_logp)
            with torch.no_grad():
                q = net(torch.from_numpy(s).unsqueeze(0).float()).squeeze(0).numpy()
            for i,a in enumerate(ALPHABET):
                if a in obs["guessed"]:
                    q[i] = -1e9
            a_idx = int(np.argmax(q))
            a = ALPHABET[a_idx]
            obs, r, done, info = env.step(a)
            if done:
                if env.pattern == env.word: wins += 1
                total_wrong += info["wrong"]
                total_repeat += info["repeated"]
                break

    success_rate = wins / games
    final_score = (success_rate * games * 10) - (total_wrong * 1) - (total_repeat * 0.5)
    print("📊 FINAL RESULTS 📊")
    print(f"Total Games: {games}")
    print(f"Success Rate: {success_rate*100:.2f}%")
    print(f"Wrong Guesses: {total_wrong}")
    print(f"Repeated Guesses: {total_repeat}")
    print(f"🏆 Final Score: {final_score:.2f}")

test_words = ["apple", "banana", "grape", "orange", "melon", "mango", "pear", "kiwi", "peach", "berry"]
evaluate_model(test_words, games=2000)
