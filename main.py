import torch
import pennylane as qml
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# -------------------------- Hyperparameters (tune if needed) --------------------------
n_wires = 3
depth = 1
action_dim = 3
scale = torch.pi
gamma = 0.99
tau = 0.001
batch_size = 32
replay_buffer_size = 10000
lr_actor = 1e-4
lr_critic = 1e-3
num_episodes = 2500        # ~2–3k episodes is enough to reach p > 0.99
T = 50                     # trajectory length
expl_noise = 0.2           # exploration noise on action
reward_noise_std = 0.01    # noise on reward (as in the paper)
reward_scale = 10.0

# Target state |1⟩
sd = torch.tensor([0.0 + 0j, 1.0 + 0j], dtype=torch.complex128)

# Devices
dev = qml.device("default.qubit", wires=n_wires)
env_dev = qml.device("default.qubit", wires=1)

# -------------------------- Ansatz helpers --------------------------
def layer(params, wires):
    for w in wires:
        qml.RX(params[w, 0], wires=w)
        qml.RZ(params[w, 1], wires=w)
        qml.RX(params[w, 2], wires=w)

def entangle(wires):
    for w in wires[:-1]:
        qml.CNOT(wires=[w, w+1])

# -------------------------- QNodes --------------------------
@qml.qnode(dev, interface="torch", diff_method="backprop")
def policy_qnode(params, state_vec):
    qml.StatePrep(state_vec, wires=0)
    for l in range(depth):
        layer(params[l], range(n_wires))
        entangle(range(n_wires))
    # Three observables give three action components
    return (qml.expval(qml.PauliX(1)),
            qml.expval(qml.PauliY(1)),
            qml.expval(qml.PauliZ(2)))

@qml.qnode(dev, interface="torch", diff_method="backprop")
def critic_qnode(params, state_vec, action):
    qml.StatePrep(state_vec, wires=0)
    # Simple angle encoding of the action
    qml.RX(action[0], wires=1)
    qml.RY(action[1], wires=2)
    qml.RZ(action[2], wires=1)
    for l in range(depth):
        layer(params[l], range(n_wires))
        entangle(range(n_wires))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(env_dev, interface="torch")
def apply_action(state_vec, action):
    qml.StatePrep(state_vec, wires=0)
    qml.RX(action[0], wires=0)
    qml.RZ(action[1], wires=0)
    qml.RX(action[2], wires=0)
    return qml.state()

# -------------------------- Networks & optimizers --------------------------
params_actor = torch.tensor(np.random.uniform(0, 2*np.pi, (depth, n_wires, 3)),
                            dtype=torch.float64, requires_grad=True)
params_critic = torch.tensor(np.random.uniform(0, 2*np.pi, (depth, n_wires, 3)),
                             dtype=torch.float64, requires_grad=True)

params_target_actor = params_actor.clone().detach()
params_target_critic = params_critic.clone().detach()

opt_actor = torch.optim.Adam([params_actor], lr=lr_actor)
opt_critic = torch.optim.Adam([params_critic], lr=lr_critic)

replay_buffer = deque(maxlen=replay_buffer_size)

def get_action(state_vec, params, noise=True):
    raw = torch.stack(policy_qnode(params, state_vec))
    action = scale * torch.tanh(raw)
    if noise:
        action = action + torch.randn(action_dim) * expl_noise
        action = torch.clamp(action, -scale, scale)
    return action

# -------------------------- Training loop --------------------------
print("Starting training...")
for ep in range(num_episodes):
    state_vec = torch.tensor(qml.math.random_state(2), dtype=torch.complex128)

    for t in range(T):
        action = get_action(state_vec, params_actor)

        next_state_vec = apply_action(state_vec, action)

        overlap = torch.vdot(next_state_vec, sd)
        p = torch.real(overlap * overlap.conj())
        reward = reward_scale * p + torch.randn(1) * reward_noise_std

        replay_buffer.append((state_vec.clone().detach(),
                              action.clone().detach(),
                              reward.clone().detach(),
                              next_state_vec.clone().detach()))

        state_vec = next_state_vec

        # Update networks if buffer is large enough
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states = torch.stack([b[0] for b in batch])
            actions = torch.stack([b[1] for b in batch])
            rewards = torch.stack([b[2] for b in batch]).squeeze()
            next_states = torch.stack([b[3] for b in batch])

            # Target Q-values
            next_actions = torch.stack([get_action(s, params_target_actor, noise=False) for s in next_states])
            target_q = rewards + gamma * torch.tensor(
                [critic_qnode(params_target_critic, s, a).item() for s, a in zip(next_states, next_actions)]
            )

            # Critic loss
            current_q = torch.tensor(
                [critic_qnode(params_critic, s, a).item() for s, a in zip(states, actions)]
            )
            critic_loss = torch.mean((current_q - target_q)**2)

            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

            # Actor loss
            policy_actions = torch.stack([get_action(s, params_actor, noise=False) for s in states])
            actor_loss = -torch.mean(torch.tensor(
                [critic_qnode(params_critic, s, a).item() for s, a in zip(states, policy_actions)]
            ))

            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            # Soft update targets
            with torch.no_grad():
                params_target_actor = tau * params_actor + (1 - tau) * params_target_actor
                params_target_critic = tau * params_critic + (1 - tau) * params_target_critic

    if (ep + 1) % 200 == 0:
        print(f"Episode {ep+1}/{num_episodes} completed.")

print("Training finished.")

# -------------------------- Test on 1000 random initial states (Figure 4 style) --------------------------
num_test = 1000
means = [[] for _ in range(T+1)]
for _ in range(num_test):
    state_vec = torch.tensor(qml.math.random_state(2), dtype=torch.complex128)
    overlap = torch.vdot(state_vec, sd)
    means[0].append(torch.real(overlap * overlap.conj()).item())

    for t in range(T):
        action = get_action(state_vec, params_actor, noise=False)
        state_vec = apply_action(state_vec, action)
        overlap = torch.vdot(state_vec, sd)
        means[t+1].append(torch.real(overlap * overlap.conj()).item())

avg_p = [np.mean(step) for step in means]
var_p = [np.var(step) for step in means]

plt.figure(figsize=(8,5))
plt.plot(range(T+1), avg_p, label="Average $|\\langle \\psi_t | 1 \\rangle|^2$")
plt.fill_between(range(T+1), np.array(avg_p) - np.array(var_p), np.array(avg_p) + np.array(var_p), alpha=0.3)
plt.xlabel("Step t")
plt.ylabel("Overlap probability")
plt.title("1-qubit state generation with quantum DDPG (1000 random initials)")
plt.legend()
plt.grid(True)
plt.show()
