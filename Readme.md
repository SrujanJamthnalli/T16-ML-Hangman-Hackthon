# Hangman ML Hackathon - UE23CS352A

## 📋 Project Overview

This project implements an intelligent Hangman game solver using:
1. **Hidden Markov Model (HMM)** - For letter probability estimation
2. **Reinforcement Learning (RL)** - For optimal letter guessing strategy

## 🎯 Objective

Build an agent that plays Hangman efficiently:
- Maximizes win rate (success rate)
- Minimizes wrong guesses
- Minimizes repeated guesses

## 📊 Evaluation

The agent will be evaluated on **2000 test words** with **6 lives per game**.

**Scoring Formula:**
```
Final Score = (Success Rate × 2000) - (Total Wrong Guesses × 5) - (Total Repeated Guesses × 2)
```

## 📁 Project Structure

```
ml-hackathon/
├── Data/
│   ├── corpus.txt          # 50,000 training words
│   └── test.txt            # 2,000 test words
├── notebooks/              # Jupyter notebooks
│   ├── 01_HMM.ipynb
│   ├── 02_RL_Agent.ipynb
│   ├── 03_Training.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_Complete_Solution.ipynb
├── src/                    # Python source files
│   ├── environment.py      # Hangman game environment
│   ├── hmm_model.py        # HMM implementation
│   ├── rl_agent.py         # RL agent implementation
│   └── utils.py            # Helper functions
├── TASK_BREAKDOWN.md       # Detailed task breakdown
├── QUICK_START.md          # Quick start guide
├── Analysis_Report.pdf     # Final analysis report
└── README.md               # This file
```

## 🚀 Quick Start

See `QUICK_START.md` for a step-by-step guide.

### Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn hmmlearn
# Optional: for Deep Q-Networks
pip install torch
```

### Basic Usage

```python
from src.environment import HangmanEnv
from src.hmm_model import HangmanHMM
from src.rl_agent import QLearningAgent

# Train HMM
hmm = HangmanHMM()
hmm.train('Data/corpus.txt')

# Train RL Agent
agent = QLearningAgent()
# ... training loop ...

# Evaluate
results = evaluate_agent(agent, hmm, 'Data/test.txt')
print(f"Success Rate: {results['success_rate']}")
print(f"Final Score: {results['final_score']}")
```

## 📝 Deliverables

1. **Jupyter Notebooks** - Complete implementation with:
   - HMM construction and training
   - RL environment and agent design
   - Training loops and hyperparameters
   - Evaluation results and plots

2. **Analysis_Report.pdf** - Analysis covering:
   - Key observations and insights
   - HMM and RL design choices
   - Exploration strategies
   - Future improvements

3. **Demo & Viva** - Live demonstration and presentation

## 🎓 Key Concepts

### Hidden Markov Model (HMM)
- Estimates probability of each letter appearing in masked positions
- Trained on corpus.txt to learn letter patterns and context

### Reinforcement Learning (RL)
- Agent learns optimal guessing strategy
- Uses HMM probabilities + game state to make decisions
- Balances exploration vs exploitation

### Hangman Environment
- Game state: masked word, guessed letters, lives remaining
- Actions: Guess a letter (A-Z)
- Rewards: Positive for correct, negative for wrong, bonus for win

## 📈 Expected Results

Track these metrics during development:
- Success Rate: Target > 80%
- Average Wrong Guesses: Target < 2 per game
- Average Repeated Guesses: Target = 0
- Final Score: Maximize!

## 🔧 Development Tips

1. **Start Simple**: Basic Q-learning + simple HMM first
2. **Iterate Fast**: Test on small subsets before full corpus
3. **Visualize**: Plot learning curves, letter frequencies, etc.
4. **Evaluate Early**: Check test performance periodically
5. **Document**: Comment code, track experiments

## 📚 Resources

- See `TASK_BREAKDOWN.md` for detailed implementation guide
- See `QUICK_START.md` for quick start instructions
- Problem statement: `Problem_Statement.pdf`

## 🏆 Success Criteria

- High success rate on test set
- Low number of wrong guesses
- Zero repeated guesses
- Well-documented code and analysis
- Clear presentation in demo/viva

Good luck! 🎯
