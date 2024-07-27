import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BernoulliBandit:
    """Bernoulli multi-armed bandit with K arms."""
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # Generate random probabilities for each arm
        self.best_idx = np.argmax(self.probs)  # Index of the arm with the highest probability
        self.best_prob = self.probs[self.best_idx]  # Highest probability value
        self.K = K

    def step(self, k):
        """Simulate pulling the k-th arm."""
        return 1 if np.random.rand() < self.probs[k] else 0

np.random.seed(1)  # Set random seed for reproducibility
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"Generated a {K}-armed Bernoulli bandit")
print(f"The best arm is {bandit_10_arm.best_idx} with a probability of {bandit_10_arm.best_prob:.4f}")

class Solver:
    """Base class for multi-armed bandit algorithms."""
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # Number of times each arm has been pulled
        self.regret = 0.  # Cumulative regret
        self.actions = []  # List of actions taken
        self.regrets = []  # List of cumulative regrets

    def update_regret(self, k):
        """Update cumulative regret."""
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        """Determine which arm to pull (to be implemented by subclasses)."""
        raise NotImplementedError

    def run(self, num_steps):
        """Run the algorithm for a specified number of steps."""
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    """Epsilon-greedy algorithm."""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # Initial reward estimates for each arm

    def run_one_step(self):
        """Select an arm to pull using the epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # Explore: choose a random arm
        else:
            k = np.argmax(self.estimates)  # Exploit: choose the best arm based on estimates
        r = self.bandit.step(k)  # Get reward for the chosen arm
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class UCB(Solver):
    """Upper Confidence Bound (UCB) algorithm."""
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        """Select an arm to pull using the UCB strategy."""
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)  # Choose the arm with the highest UCB value
        r = self.bandit.step(k)  # Get reward for the chosen arm
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    """Thompson Sampling algorithm."""
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # List to store the number of successes for each arm
        self._b = np.ones(self.bandit.K)  # List to store the number of failures for each arm

    def run_one_step(self):
        """Select an arm to pull using the Thompson Sampling strategy."""
        samples = np.random.beta(self._a, self._b)  # Sample from the Beta distribution
        k = np.argmax(samples)  # Choose the arm with the highest sample value
        r = self.bandit.step(k)
        self._a[k] += r  # Update the success count
        self._b[k] += (1 - r)  # Update the failure count
        return k

def plot_results(solvers, solver_names):
    """Plot cumulative regret over time for different solvers."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx], linewidth=2)

    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.title(f'{solvers[0].bandit.K}-Armed Bandit: Cumulative Regret', fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(50000)
print(f'Cumulative regret of the epsilon-greedy algorithm: {epsilon_greedy_solver.regret}')

np.random.seed(1)
coef = 1  # Coefficient to control the weight of uncertainty
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(50000)
print(f'Cumulative regret of the UCB algorithm: {UCB_solver.regret}')

np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(50000)
print(f'Cumulative regret of the Thompson Sampling algorithm: {thompson_sampling_solver.regret}')

plot_results([epsilon_greedy_solver, UCB_solver, thompson_sampling_solver], ["EpsilonGreedy", "UCB", "ThompsonSampling"])
