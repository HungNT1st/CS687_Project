import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

def demo_random_actions():
    env = gym.make('CartPole-v1', render_mode='human')
    obs, info = env.reset(seed=42)
    done = False
    while not done:
        action = env.action_space.sample()  
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def policy_probs(theta, s):
    logits = theta @ s
    return softmax(logits)

def value(w, s):
    return w @ s

def grad_log_policy(theta, s, a):
    pi_s = policy_probs(theta, s)
    grad = -np.outer(pi_s, s)
    grad[a, :] += s
    return grad

def grad_value_func(s):
    return s

def run_episode_reinforce_with_baseline(env, gamma, theta, w, alpha_theta, alpha_w):
    # Generate an episode
    states, actions, rewards = [], [], []
    s, info = env.reset()
    s = s.astype(np.float32)
    done = False
    while not done:
        pi_s = policy_probs(theta, s)
        a = np.random.choice(len(pi_s), p=pi_s)
        states.append(s)
        actions.append(a)
        s_next, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        rewards.append(r)
        s = s_next.astype(np.float32)

    # Compute returns and update weights
    G = 0
    for t in reversed(range(len(states))):
        G = rewards[t] + gamma * G
        s_t = states[t]
        a_t = actions[t]
        
        # The baseline
        v_s_t = value(w, s_t)
        delta = G - v_s_t
        
        # Update weights
        w += alpha_w * delta * grad_value_func(s_t)
        
        # Update theta
        theta += alpha_theta * delta * grad_log_policy(theta, s_t, a_t)

    return sum(rewards)

def run_episode_actor_critic(env, gamma, theta, w, alpha_theta, alpha_w):
    s, info = env.reset()
    s = s.astype(np.float32)
    done = False
    total_reward = 0
    I = 1
    while not done:
        pi_s = policy_probs(theta, s)
        # This sometimes cause ValueError: probabilities contain NaN
        # a = np.random.choice(len(pi_s), p=pi_s)
        
        if np.isnan(pi_s).any():
            pi_s = np.nan_to_num(pi_s, nan=1.0/len(pi_s))
            pi_s /= np.sum(pi_s)
        a = np.random.choice(len(pi_s), p=pi_s)
        
        s_next, r, terminated, truncated, info = env.step(a)
        s_next = s_next.astype(np.float32)
        done = terminated or truncated
        total_reward += r

        
        v_s = value(w, s)
        v_s_next = value(w, s_next) if not done else 0.0
        delta = r + gamma * v_s_next - v_s

        w += alpha_w * delta * grad_value_func(s)
        theta += alpha_theta * I * delta * grad_log_policy(theta, s, a)
        I *= gamma

        s = s_next

    return total_reward

def run_episode_n_step_sarsa(env, gamma, theta, w, alpha_theta, alpha_w, n_steps):
    s, info = env.reset()
    s = s.astype(np.float32)
    done = False
    total_reward = 0
    
    #pick first action
    pi_s = policy_probs(theta, s)
    a = np.random.choice(len(pi_s), p=pi_s)
    
    # Init n-step buffers
    states, actions, rewards = [s], [a], []
    
    T = float('inf')
    
    for t in itertools.count():
        if t < T:
            s_next, r, terminated, truncated, info = env.step(a)
            s_next = s_next.astype(np.float32)
            done = terminated or truncated
            total_reward += r

            states.append(s_next)
            rewards.append(r)
            
            # Choose next action
            if not done:
                pi_s_next = policy_probs(theta, s_next)
                a_next = np.random.choice(len(pi_s_next), p=pi_s_next)
                actions.append(a_next)
            else:
                T = t + 1
        
        tau = t + 1 - n_steps
        
        
        # Update after n steps or at the end of the episode
        if tau >= 0:
            # compute n-step returns
            G = sum(np.power(gamma, i) * rewards[tau + i] for i in range(min(n_steps, T - tau)))
            if tau + n_steps < T:
                G += np.power(gamma, n_steps) * value(w, states[tau + n_steps])
        
            #update weights
            s_tau = states[tau]
            a_tau = actions[tau]
            v_s_tau = value(w, s_tau)
            delta = G - v_s_tau
            
            w += alpha_w * delta * grad_value_func(s_tau)
            theta += alpha_theta * delta * grad_log_policy(theta, s_tau, a_tau)
        
        if tau == T - 1:
            break
        
        s, a = s_next, a_next
        
    return total_reward
            
def run_experiments(env_name, algorithm='reinforce_baseline', gamma=0.99,
                    alpha_theta=0.001, alpha_w=0.01, n_episodes=2000, n_runs=5, n_steps=4, seed=0):
    np.random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    returns_all_runs = []
    for run in range(n_runs):
        theta = np.zeros((n_actions, obs_dim))
        w = np.zeros(obs_dim)

        run_returns = []
        for ep in tqdm(range(n_episodes), desc=f'Run {run+1}/{n_runs}'):
            if algorithm == 'reinforce_baseline':
                G = run_episode_reinforce_with_baseline(env, gamma, theta, w, alpha_theta, alpha_w)
            elif algorithm == 'actor_critic':
                G = run_episode_actor_critic(env, gamma, theta, w, alpha_theta, alpha_w)
            elif algorithm == 'n_step_sarsa':
                G = run_episode_n_step_sarsa(env, gamma, theta, w, alpha_theta, alpha_w, n_steps)
            else:
                raise ValueError("Unknown algorithm!")

            run_returns.append(G)
        returns_all_runs.append(run_returns)

    env.close()
    returns_all_runs = np.array(returns_all_runs)
    mean_returns = np.mean(returns_all_runs, axis=0)
    std_returns = np.std(returns_all_runs, axis=0)
    return mean_returns, std_returns

if __name__ == "__main__":
    # Parameters
    gamma = 0.99
    baseline_alpha_theta = 0.00006
    baseline_alpha_w = 0.006
    ac_alpha_theta = 0.02
    ac_alpha_w = 0.1
    nss_alpha_theta = 0.001
    nss_alpha_w = 0.01
    n_steps = 5
    n_episodes = 5000
    n_runs = 5

    # Run experiments
    # envs = ['CartPole-v1', 'MountainCar-v0']
    # envs = ['CartPole-v1']
    envs = ['MountainCar-v0']
    # algorithms = ['reinforce_baseline', 'actor_critic']
    # algorithms = ['actor_critic']
    # algorithms = ['n_step_sarsa']
    algorithms = ['reinforce_baseline']

    seed = np.random.randint(0, 2**32 - 1)
    results = {}
    for env_name in envs:
        results[env_name] = {}
        for algo in algorithms:
            if algo == 'reinforce_baseline':
                alpha_theta = baseline_alpha_theta
                alpha_w = baseline_alpha_w
            elif algo == 'actor_critic':
                alpha_theta = ac_alpha_theta
                alpha_w = ac_alpha_w
            elif algo == 'n_step_sarsa':
                alpha_theta = nss_alpha_theta
                alpha_w = nss_alpha_w
            mean_ret, std_ret = run_experiments(env_name, algorithm=algo, gamma=gamma,
                                                alpha_theta=alpha_theta, alpha_w=alpha_w,
                                                n_episodes=n_episodes, n_runs=n_runs, n_steps=n_steps, seed=seed)
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
        # plt.show()
        plt.savefig(f'figure/{env_name}_{algorithms}_learning_curves.png')
        plt.close()