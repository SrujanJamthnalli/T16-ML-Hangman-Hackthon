import random

class HangmanEnv:
    def __init__(self, words, lives=6, seed=42):
        self.words = words
        self.lives0 = lives
        self.rng = random.Random(seed)
        self.reset()

    def reset(self, word=None):
        self.word = word if word else self.rng.choice(self.words)
        self.guessed = set()
        self.lives = self.lives0
        self.pattern = "_" * len(self.word)
        self.done = False
        self.info = {}
        return self._obs()

    def _obs(self):
        return {"pattern": self.pattern, "guessed": set(self.guessed), "lives": self.lives}

    def step(self, action: str):
        if self.done:
            raise RuntimeError("Game already finished. Call reset().")
        reward = 0.0
        wrong = 0
        repeated = 0

        if action in self.guessed:
            repeated = 1
            reward -= 2.0
        else:
            self.guessed.add(action)
            if action in self.word:
                new_pat = list(self.pattern)
                for i, ch in enumerate(self.word):
                    if ch == action:
                        new_pat[i] = action
                self.pattern = "".join(new_pat)
                reward += 1.0
            else:
                wrong = 1
                self.lives -= 1
                reward -= 1.0

        reward -= 0.01

        if "_" not in self.pattern:
            self.done = True
            reward += 10.0
        elif self.lives <= 0:
            self.done = True
            reward -= 10.0

        self.info = {"wrong": wrong, "repeated": repeated}
        return self._obs(), reward, self.done, self.info
