# QUADRAPAD — Quadruped Locomotion Using Reinforcement Learning

QUADRAPAD is a quadruped robotics project focused on training and visualizing the locomotion of a four-legged robot using **Proximal Policy Optimization (PPO)** and the **MuJoCo physics simulator**.

The project includes the robot model, training environment, trained model checkpoints, training scripts, and evaluation tools used to observe the robot's movement.

## Project Overview

The project is organized into five main components:

- **Checkpoints:** Stores trained PPO model checkpoints.
- **Environments:** Contains the environment used to train the robot using reinforcement learning.
- **Evaluation:** Contains scripts for viewing, testing, and inspecting robot behavior.
- **Robot Model:** Contains the Unitree Go1 robot model and its MuJoCo simulation files.
- **Training:** Contains the training script and saved training results.

## Repository Structure

```text
QUADRAPAD-ROBOT-USING-MUJUCO/
│
├── checkpoints/
│   └── Saved model checkpoints
│
├── envs/
│   ├── go1_env.py
│   └── go1_env.py.backup
│
├── evaluation/
│   ├── evaluate_stand.py
│   ├── go1_env.py
│   ├── inspect1_go1.py
│   ├── inspect_go1.py
│   ├── stand_view.py
│   ├── test_env.py
│   ├── trait_movement.py
│   └── view.py
│
├── models/
│   └── unitree_go1/
│       ├── assets/
│       ├── CHANGELOG.md
│       ├── LICENSE
│       ├── README.md
│       ├── go1.png
│       ├── go1.xml
│       └── scene.xml
│
├── training/
│   ├── train.py
│   ├── training_stg1.py
│   ├── go1_ppo.zip
│   ├── go1_ppo_v1.zip
│   ├── go1_ppo_v2.zip
│   ├── go1_ppo_v3.zip
│   ├── go1_ppo_v5.zip
│   ├── go1_ppo_v5_vecnormalize.pkl
│   ├── go1_ppo_v7_gpu.zip
│   ├── go1_trot_ppo.zip
│   ├── go1_trot_pure_rl.zip
│   └── Training logs
│
└── .gitignore
```

*Note: The tree highlights the files visible in the repository screenshots. Some directories contain additional files and logs.*

## 1. Robot Model — `models/unitree_go1/`

This directory contains the Unitree Go1 quadruped model used in the MuJoCo simulation.

- **`go1.xml`** — Robot model description.
- **`scene.xml`** — Simulation scene used to load the robot into MuJoCo.
- **`assets/`** — Supporting model assets.
- **`go1.png`** — Robot image.
- **`README.md` and `LICENSE`** — Documentation and licensing information associated with the model.

The robot model provides the simulated body, joints, and physical properties needed to interact with the environment.

## 2. Training Environment — `envs/`

The `envs/` directory contains the environment used to train the quadruped.

- **`go1_env.py`** — Defines the robot's training environment.
- **`go1_env.py.backup`** — Backup copy of the environment file.

The environment connects the robot simulation to the reinforcement learning algorithm. It defines what the agent observes, which actions it can take, and how its performance is rewarded.

During training, the agent repeatedly interacts with the environment and improves its policy based on the rewards it receives.

## 3. Training — `training/`

The `training/` directory contains the main training scripts and saved PPO models.

- **`train.py`** — Main training script used to train the PPO policy.
- **`training_stg1.py`** — Script associated with the first training stage.
- **PPO checkpoint files** — Saved policies from different training runs and experiments.
- **Training logs** — Records generated during training.

### Reinforcement Learning Workflow

1. The environment provides observations about the robot's state.
2. The PPO policy receives these observations.
3. The policy outputs actions for the robot's joints.
4. MuJoCo simulates the resulting movement.
5. The environment calculates a reward.
6. PPO uses the collected experience to update the policy.

This process repeats to improve the robot's behavior.

## 4. Checkpoints — `checkpoints/`

The `checkpoints/` directory stores saved training checkpoints.

These checkpoints preserve model states from training runs, allowing previous results to be retained for evaluation or comparison.

Keeping checkpoints from different experiments also makes it easier to track the development of the project over time.

## 5. Evaluation — `evaluation/`

The `evaluation/` directory contains scripts for inspecting and testing the robot and its movement.

| File | Purpose |
|---|---|
| `view.py` | Used to load and visualize a trained model in the MuJoCo simulation. |
| `trait_movement.py` | Contains the handmade trot movement implementation. |
| `evaluate_stand.py` | Script for evaluating standing behavior. |
| `stand_view.py` | Script associated with viewing standing behavior. |
| `inspect_go1.py` | Script for inspecting the Go1 robot or environment. |
| `inspect1_go1.py` | Additional robot inspection script. |
| `test_env.py` | Script for testing the environment. |
| `go1_env.py` | Environment-related file present in the evaluation directory. |

The main purpose of this directory is to inspect how the robot behaves in simulation and evaluate the results of training.

### Learned Movement vs. Handmade Movement

The project includes a handmade trot implementation in `trait_movement.py` and PPO-trained policies saved as model checkpoints.

These represent two different approaches to producing robot movement:

- **Handmade movement:** Movement is specified through programmed joint trajectories or rules.
- **PPO-based movement:** The policy learns actions through interaction with the environment and reward feedback.

The PPO-based approach is intended to develop locomotion through learning rather than relying entirely on a predefined movement trajectory.

## Technologies Used

- Python
- MuJoCo
- Gymnasium
- Stable-Baselines3
- Proximal Policy Optimization (PPO)
- NumPy
- PyTorch

*The list reflects the technologies associated with the project; the exact dependencies should be confirmed from the training code and installed environment.*

## Project Goals

- Simulate a quadruped robot using MuJoCo.
- Develop a reinforcement learning environment for quadruped locomotion.
- Train a locomotion policy using PPO.
- Save checkpoints from different training experiments.
- Visualize and inspect trained robot behavior.
- Explore the difference between learned locomotion and handmade movement.

## Current Development

QUADRAPAD brings together the simulation model, training environment, PPO training scripts, model checkpoints, and evaluation tools in one repository.

The saved checkpoints provide a record of different experiments. Further work can focus on improving gait stability, foot-ground contact, forward movement, and the consistency of the learned locomotion.

## Future Improvements

- Improve the stability and consistency of learned trotting.
- Reduce unnecessary leg movement and shaking.
- Improve ground contact and balance.
- Evaluate locomotion across different training checkpoints.
- Compare learned locomotion with the handmade trot implementation.
- Explore robustness under different simulation conditions.

## Author

**Shaik Dada Khalandar**  
Undergraduate Student — Artificial Intelligence and Robotics  
Dayananda Sagar University
