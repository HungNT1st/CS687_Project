import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def featurize_state_action(s, a, obs_dim, n_actions):
    # Create a feature vector for Q(s,a) in R^(obs_dim * n_actions)
    x_sa = np.zeros(obs_dim * n_actions, dtype=np.float32)
    start = a * obs_dim
    end = start + obs_dim
    x_sa[start:end] = s
    return x_sa

def epsilon_greedy(Q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q_values))
    else:
        return np.argmax(Q_values)

def run_episode_true_online_sarsa_lambda(env, gamma, alpha, lam, epsilon, w):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    s, info = env.reset()
    s = s.astype(np.float32)
    
    # Choose A by epsilon-greedy
    Q_values = [w @ featurize_state_action(s, a, obs_dim, n_actions) for a in range(n_actions)]
    A = epsilon_greedy(Q_values, epsilon)
    
    x = featurize_state_action(s, A, obs_dim, n_actions)
    z = np.zeros_like(w, dtype=np.float32)
    Q_old = 0.0
    total_reward = 0.0
    done = False

    while not done:
        s_next, R, terminated, truncated, info = env.step(A)
        s_next = s_next.astype(np.float32)
        done = terminated or truncated
        total_reward += R

        if not done:
            # compute next action A' epsilon-greedy
            Q_values_next = [w @ featurize_state_action(s_next, a, obs_dim, n_actions) for a in range(n_actions)]
            A_next = epsilon_greedy(Q_values_next, epsilon)
            x_next = featurize_state_action(s_next, A_next, obs_dim, n_actions)
            Q_prime = Q_values_next[A_next]
        else:
            # terminal state
            x_next = np.zeros_like(x)
            Q_prime = 0.0

        Q = w @ x
        delta = R + gamma * Q_prime - Q
        z = gamma * lam * z + (1 - alpha * gamma * lam * (z @ x)) * x
        w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
        Q_old = Q_prime
        x = x_next
        A = A_next if not done else None
    return total_reward, w

def run_single_experiment(env_name, algorithm, gamma, alpha_theta, alpha_w, n_episodes, n_steps, seed, run_id, 
                          lam=0.9, epsilon=0.1, alpha=0.001):
    seed += run_id
    print(f'Running {algorithm} on {env_name} with seed {seed}')
    np.random.seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if algorithm == 'true_online_sarsa_lambda':
        w = np.zeros(obs_dim * n_actions, dtype=np.float32)
    else:
        w = np.zeros(obs_dim, dtype=np.float32)

    theta = np.zeros((n_actions, obs_dim), dtype=np.float32) 

    epsilon_decay = 0.95
    epsilon_min = 0.001
    run_returns = []
    for ep in range(n_episodes):
        if algorithm == 'reinforce_baseline':
            G = run_episode_reinforce_with_baseline(env, gamma, theta, w, alpha_theta, alpha_w)
        elif algorithm == 'actor_critic':
            G = run_episode_actor_critic(env, gamma, theta, w, alpha_theta, alpha_w)
        elif algorithm == 'n_step_sarsa':
            G = run_episode_n_step_sarsa(env, gamma, theta, w, alpha_theta, alpha_w, n_steps)
        elif algorithm == 'true_online_sarsa_lambda':
            # Epsilon decay
            if ep % 5 == 0:
                epsilon = max(epsilon * epsilon_decay, epsilon_min)
            G, w = run_episode_true_online_sarsa_lambda(env, gamma, alpha, lam, epsilon, w)
        else:
            raise ValueError("Unknown algorithm!")
        if ep % 500 == 0:
            print(f'Run {run_id+1} - Episode {ep}/{n_episodes} - Return: {G}')
        run_returns.append(G)

    env.close()
    return run_returns

def run_experiments(env_name, algorithm='reinforce_baseline', gamma=0.99,
                    alpha_theta=0.001, alpha_w=0.01, n_episodes=2000, n_runs=5, n_steps=4, seed=0,
                    lam=0.9, epsilon=0.1, alpha=0.001):
    args = [(env_name, algorithm, gamma, alpha_theta, alpha_w, n_episodes, n_steps, seed + run, run, lam, epsilon, alpha) 
            for run in range(n_runs)]

    with Pool() as pool:
        results_all_runs = pool.starmap(run_single_experiment, args)

    returns_all_runs = np.array(results_all_runs)
    mean_returns = np.mean(returns_all_runs, axis=0)
    std_returns = np.std(returns_all_runs, axis=0)
    return mean_returns, std_returns


if __name__ == "__main__":
    gamma = 0.99
    n_steps = 5
    n_episodes = 2000
    n_runs = 5

    # envs = ['CartPole-v1']
    envs = ['Acrobot-v1']
    algorithms = ['true_online_sarsa_lambda']

    lam = 0.9
    epsilon = 0.1
    alpha = 0.0003
    
    # Grid search for best parameters
    # lam_values = np.linspace(0.1, 0.95, 9)
    # alpha_values = np.linspace(0.0001, 0.001, 10)

    # best_mean_return = -np.inf
    # best_params = None

    # seed = np.random.randint(0, 2**32 - 1)
    # for lam in lam_values:
    #     for alpha in alpha_values:
    #         mean_ret, std_ret = run_experiments(envs[0], algorithm=algorithms[0], gamma=gamma,
    #                                             alpha_theta=0.0, alpha_w=0.0,
    #                                             n_episodes=n_episodes, n_runs=n_runs, n_steps=n_steps, seed=seed,
    #                                             lam=lam, epsilon=epsilon, alpha=alpha)
    #         if np.mean(mean_ret) > best_mean_return:
    #             best_mean_return = np.mean(mean_ret)
    #             best_params = (lam, alpha)

    # print(f'Best parameters: lambda={best_params[0]}, alpha={best_params[1]} with mean return {best_mean_return}')

    seed = np.random.randint(0, 2**32 - 1)
    results = {}
    for env_name in envs:
        results[env_name] = {}
        for algo in algorithms:
            if algo == 'reinforce_baseline':
                alpha_theta_used = baseline_alpha_theta
                alpha_w_used = baseline_alpha_w
            elif algo == 'actor_critic':
                alpha_theta_used = ac_alpha_theta
                alpha_w_used = ac_alpha_w
            elif algo == 'n_step_sarsa':
                alpha_theta_used = nss_alpha_theta
                alpha_w_used = nss_alpha_w
            elif algo == 'true_online_sarsa_lambda':
                alpha_theta_used = 0.0
                alpha_w_used = 0.0
            mean_ret, std_ret = run_experiments(env_name, algorithm=algo, gamma=gamma,
                                                alpha_theta=alpha_theta_used, alpha_w=alpha_w_used,
                                                n_episodes=n_episodes, n_runs=n_runs, n_steps=n_steps, seed=seed,
                                                lam=lam, epsilon=epsilon, alpha=alpha)
            results[env_name][algo] = (mean_ret, std_ret)

    for env_name in envs:
        plt.figure(figsize=(10,6))
        for algo in algorithms:
            mean_ret, std_ret = results[env_name][algo]
            episodes = np.arange(len(mean_ret))
            plt.plot(episodes, mean_ret, label=algo)
            plt.fill_between(episodes, mean_ret - std_ret, mean_ret + std_ret, alpha=0.2)
        plt.title(f'{env_name} Learning Curves')
        plt.xlabel('Episode')
        plt.ylabel('Average Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figure/{env_name}_{algorithms}_learning_curves.png')
        plt.close()
