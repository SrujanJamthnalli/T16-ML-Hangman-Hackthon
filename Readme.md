Hangman Reinforcement Learning - Project README
👥 Authors

Developed as part of ML Hackathon coursework

Team Members:
- PES1UG23AM917 - CHANDAN R
-PES1UG24AM814 – SRUJAN J
-PES1UG23AM347 - CHAKRESH

📋 Project Overview

This project implements an intelligent Hangman game solver using a combination of a Hidden Markov Model (HMM) for probabilistic letter predictions and a Deep Q-Network (DQN) for decision-making. 
The goal is to create an agent that plays Hangman efficiently, maximizing success rate while minimizing wrong and repeated guesses.

🎯 Objective

Build an RL agent that can learn optimal guessing behavior through interaction with the Hangman environment, leveraging letter transition probabilities derived from the HMM model.

📁 Project Structure

├── hmm_model.py       # Builds and trains the HMM for letter probability estimation
├── train.py           # DQN training loop with replay buffer and target network
├── stimulate.py       # Simulates a single Hangman game using trained model
├── evaluate.py        # Evaluates the trained DQN agent on test words
├── requirements.txt   # All required Python dependencies
└── dqn_hangman_model.pth  # Saved trained model weights

⚙️ Installation

1. Clone or download the repository containing all source files.
2. Install dependencies using:

•	pip install -r requirements.txt

3. Ensure Python ≥ 3.9 and PyTorch ≥ 2.0 are installed.

🚀 Usage

To train the model from scratch:

•	python train.py

To test the trained agent on evaluation words:

•	python evaluate.py

To visualize step-by-step predictions for a specific word:

•	python stimulate.py
🧠 Technical Details

• **State Representation (54D)**: Combination of 26 HMM posterior probabilities, 26 binary guessed indicators, and 2 normalized scalars (blanks_left, lives_left).

• **HMM (hmm_model.py)**: 
  - Trains a bigram transition model using Laplace smoothing.
  - Computes posterior probabilities for letter positions given masked patterns.

• **DQN (train.py)**: 
  - Uses replay buffer and target network for stability.
  - Employs ε-greedy policy with decaying ε for exploration-exploitation tradeoff.
  - Reward shaping encourages correct predictions and penalizes repetition.

• **Evaluation (evaluate.py)**: 
  - Runs multiple games, logging success rate, wrong/repeated guesses, and final score.

📊 Expected Results

• Success Rate: Target 45-65% after training ~2000 episodes.
• Wrong Guesses: < 3 per game on average.
• Repeated Guesses: Approaching 0.
• Training Time: ~30–40 minutes depending on system performance.

🔮 Future Enhancements

• Use Double or Dueling DQN to improve value stability.
• Add word embeddings to encode semantic similarity.
• Train on a larger and more diverse corpus for generalization.
• Introduce LSTM-based sequential prediction for better letter context understanding.




