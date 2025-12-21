# 🚗 CARLA Reinforcement Learning Training

A reinforcement learning codebase for training self-driving cars in the CARLA simulator using Deep Q-Network (DQN) with asynchronous multi-agent architecture (ARTDQN). This implementation uses a 5-layer residual CNN model to learn autonomous driving policies from camera inputs.

---

## 📁 Directory Structure

```
rl_training_carla/
├── train.py                 # Main training script - orchestrates training process
├── play.py                  # Inference script - test trained models in CARLA
├── settings.py              # Configuration file - all hyperparameters and settings
├── control.py               # Interactive command interface for runtime control
├── list_gpus.py             # Utility to list available GPUs
├── requirements.txt         # Python dependencies
│
├── sources/                 # Core implementation modules
│   ├── __init__.py         # Module initialization and exports
│   │
│   ├── models.py           # Neural network model architectures
│   │                        # - Defines model_base_5_residual_CNN (5-layer residual CNN)
│   │                        # - Model heads for action prediction
│   │                        # - Custom models can be added here
│   │
│   ├── agent.py            # ARTDQNAgent class - individual RL agents
│   │                        # - Executes actions in CARLA environment
│   │                        # - Collects experiences (transitions)
│   │                        # - Loads and applies updated model weights
│   │
│   ├── trainer.py          # ARTDQNTrainer class - central training process
│   │                        # - Manages replay memory (experience buffer)
│   │                        # - Updates neural network weights
│   │                        # - Manages target network updates
│   │                        # - Handles checkpoint saving
│   │
│   ├── carla.py            # CARLA environment wrapper
│   │                        # - CarlaEnv class - OpenAI Gym-like interface
│   │                        # - Environment state management
│   │                        # - Action execution and reward calculation
│   │                        # - CARLA simulator lifecycle (start/restart/kill)
│   │
│   ├── common.py           # Shared utilities and constants
│   │                        # - STOP states and messages
│   │                        # - Hyperparameter loading/saving
│   │                        # - OS detection utilities
│   │
│   ├── console.py          # ConsoleStats class - real-time stats display
│   │                        # - Prints training metrics to terminal
│   │                        # - Episode statistics, epsilon, rewards, etc.
│   │
│   ├── commands.py         # Commands class - runtime command processing
│   │                        # - Handles user commands during training
│   │                        # - Epsilon/discount adjustments, stopping, etc.
│   │
│   └── tensorboard.py      # TensorBoard logging integration
│                            # - Logs training metrics for visualization
│
├── models/                  # Saved model checkpoints (created during training)
├── checkpoint/              # Training checkpoints and hyperparameters (created during training)
├── logs/                    # TensorBoard log files (created during training)
└── tmp/                     # Temporary files for inter-process communication
```

---

## 🔧 Setup

### Prerequisites

**Python 3.7** (required - older TensorFlow/Keras versions are used)

**CARLA 0.9.6** - Download from the [CARLA website](https://carla.org/releases/):
1. Download CARLA 0.9.6 for your platform from https://carla.org/releases/
2. Extract the CARLA package to a location of your choice
3. Note the path to the CARLA root folder (contains `CarlaUE4.sh` or `CarlaUE4.exe`)

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: This installs TensorFlow 1.13.1 with GPU support and Keras 2.2.4. Ensure you have compatible CUDA/cuDNN versions.

2. **Configure CARLA path in `settings.py`:**
   ```python
   CARLA_PATH = '../CARLA_0.9.6_Python_3.7'  # Update this to your CARLA installation path
   ```

3. **Verify GPU setup (optional but recommended):**
   ```bash
   python list_gpus.py
   ```

4. **Create necessary directories** (they will be created automatically, but you can create them manually):
   ```bash
   mkdir -p models checkpoint logs tmp
   ```

---

## 🔄 Training Workflow

The training system uses an **Asynchronous Real-Time DQN (ARTDQN)** architecture with multiple agents and a centralized trainer:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CARLA Simulator                          │
│  (Multiple agents can run in parallel instances)            │
└────────────┬────────────────────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
  ┌───▼───┐    ┌───▼───┐      ┌───▼───┐
  │Agent 1│    │Agent 2│ ...  │Agent N│  (Multiple agents)
  └───┬───┘    └───┬───┘      └───┬───┘
      │            │               │
      │            │ Collect Experiences │
      │            │  (transitions)      │
      └────────────┴──────────┬──────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Replay Memory   │
                    │  (Experience Buffer)│
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   ARTDQNTrainer   │
                    │                   │
                    │  • Train DQN      │
                    │  • Update weights │
                    │  • Target network │
                    │  • Checkpoints    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Updated Weights │
                    │  (Shared Memory)  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │    All Agents     │
                    │  (Load new weights)│
                    └───────────────────┘
```

### Detailed Training Process

1. **Initialization (`train.py`):**
   - Starts CARLA simulator(s)
   - Creates/loads model architecture (5-layer residual CNN from `sources/models.py`)
   - Initializes replay memory buffer
   - Sets up shared memory for weight synchronization
   - Loads previous checkpoint if available (resume training)

2. **Agent Processes (`sources/agent.py`):**
   - Each agent runs in a separate process
   - Continuously interacts with CARLA environment:
     - Observes current state (camera image + speed)
     - Selects action using epsilon-greedy policy
     - Executes action in CARLA
     - Receives reward and new state
     - Stores transition `(state, action, reward, next_state, done)` in queue
   - Periodically loads updated model weights from shared memory
   - Sends statistics to trainer for logging

3. **Trainer Process (`sources/trainer.py`):**
   - Runs in a separate process
   - Continuously:
     - Collects transitions from agent queues into replay memory
     - Samples mini-batches from replay memory when buffer is large enough
     - Trains the neural network using DQN loss:
       ```
       Q(s, a) = r + γ * max_a' Q_target(s', a')
       loss = (Q(s, a) - Q_predicted(s, a))²
       ```
     - Updates model weights
     - Periodically copies weights to target network (for stable Q-learning)
     - Serializes and shares updated weights with all agents
     - Logs metrics to TensorBoard
     - Saves checkpoints at configured intervals

4. **Environment (`sources/carla.py`):**
   - Manages CARLA simulator connection
   - Handles car spawning, weather, NPCs
   - Processes camera images (resize, grayscale/RGB)
   - Executes actions (steering, throttle, brake)
   - Calculates rewards based on:
     - Speed (encourages movement)
     - Collisions (penalizes crashes)
     - Episode progress

5. **Checkpointing:**
   - Models are saved periodically (`SAVE_CHECKPOINT_EVERY` episodes)
   - Saves model weights and hyperparameters to `checkpoint/`
   - Allows resuming training from last checkpoint
   - Best models (above `MIN_REWARD` threshold) are saved to `models/`

6. **Monitoring:**
   - Console stats display: real-time metrics in terminal
   - TensorBoard: detailed training curves and visualizations
   - Runtime commands: adjust hyperparameters during training via `control.py`

---

## 🚀 Usage

### Training

**Start training:**
```bash
python train.py
```

The system will:
1. Start CARLA simulator (or connect to existing instances)
2. Initialize model and agents
3. Begin training loop
4. Display real-time statistics in the terminal

**Training Configuration:**

Edit `settings.py` to customize:
- `AGENTS`: Number of parallel agents (more = faster data collection)
- `MODEL_BASE`: Model architecture (`'5_residual_CNN'` for 5-layer residual)
- `START_EPSILON`, `EPSILON_DECAY`, `MIN_EPSILON`: Exploration strategy
- `DISCOUNT`: Q-learning discount factor (gamma)
- `UPDATE_TARGET_EVERY`: Frequency of target network updates
- `REPLAY_MEMORY_SIZE`: Experience buffer size
- `MINIBATCH_SIZE`: Training batch size
- `OPTIMIZER_LEARNING_RATE`: Learning rate
- And many more...

**Runtime Control:**

Use `control.py` in a separate terminal to send commands during training:
```bash
python control.py
```

Available commands:
- `epsilon current|decay|min value` - Adjust exploration parameters
- `discount value` - Change discount factor
- `target update_every value` - Change target network update frequency
- `stop now|checkpoint` - Stop training immediately or at next checkpoint
- `help` - See all available commands

### Playing (Testing Trained Models)

**Test a trained model:**
```bash
python play.py
```

This will:
1. List all available saved models
2. Prompt you to select a model
3. Start CARLA simulator
4. Run the agent using the selected model weights
5. Display the agent's behavior in real-time

**Using Pre-trained Model:**

A pre-trained 5-layer residual CNN model checkpoint (trained for 94,200 episodes) is available:

🔗 **[Download Model Checkpoint](https://drive.google.com/drive/folders/1Wqhbu2s8gxeoAfeKDT74_Vjg5bgoW3bO?usp=sharing)**

The checkpoint folder contains:
- `5_residual_64-64-128-256-512_94200.model` - The trained model weights (93.9 MB)
- `hparams.json` - Hyperparameters and training configuration (54 KB)

**Setup Instructions:**
1. Download both files from the Google Drive link above
2. Place them in the `checkpoint/` directory
3. Run `play.py` to test the model:
   ```bash
   python play.py
   ```

---

## 📊 Model Architecture

The default model is a **5-layer residual CNN** (`model_base_5_residual_CNN` in `sources/models.py`):

- **Input**: Grayscale or RGB camera images
- **Architecture**:
  - 5 convolutional layers with residual connections
  - Residual connections concatenate layer outputs with inputs
  - Progressive feature extraction: 64 → 64 → 128 → 256 → 512 filters
  - Average pooling between layers for downsampling
- **Head**: Dense layers for Q-value prediction
- **Output**: Q-values for each action (forward, forward_left, forward_right, brake, brake_left, brake_right)

You can add custom model architectures in `sources/models.py` and reference them in `settings.py` via `MODEL_BASE`.

---

## 📈 Monitoring Training

### Console Output

The terminal displays real-time statistics:
- Episode number
- Epsilon (exploration rate)
- Training statistics (loss, learning rate)
- Agent statistics (FPS, state)
- Episode statistics (rewards, duration)
- CARLA simulator status

### TensorBoard

View detailed training curves:
```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser to see:
- Episode rewards (raw and weighted)
- Episode duration
- Q-values
- Training loss
- And more metrics

---

## 🔍 Key Components Explained

### ARTDQN (Asynchronous Real-Time DQN)

- **Asynchronous**: Multiple agents collect experiences in parallel
- **Real-Time**: Agents continuously interact with environment while training happens
- **DQN**: Deep Q-Network learns Q-function for action-value estimation

### Experience Replay

- Agents store transitions `(s, a, r, s', done)` in a shared replay buffer
- Trainer samples random batches from this buffer
- Breaks correlation between consecutive experiences
- Enables stable learning from diverse experiences

### Target Network

- Separate network for stable Q-value targets
- Updated periodically from main network
- Prevents instability from chasing moving targets
- Critical for DQN convergence

### Epsilon-Greedy Exploration

- Balances exploration vs exploitation
- Starts with high epsilon (mostly random actions)
- Gradually decays to minimum epsilon (mostly learned policy)
- Ensures agent explores environment while learning

---

## ⚙️ Important Settings

Key configuration values in `settings.py`:

- **`CARLA_PATH`**: Path to CARLA 0.9.6 installation (must be set correctly)
- **`CARLA_HOSTS_NO`**: Number of CARLA instances to run (can parallelize)
- **`AGENTS`**: Number of agent processes (typically 1-4)
- **`SECONDS_PER_EPISODE`**: Episode duration in seconds
- **`IMG_WIDTH`, `IMG_HEIGHT`**: Input image dimensions
- **`ACTIONS`**: Available actions (steering + throttle/brake combinations)
- **`MODEL_BASE`**: Model architecture name
- **`REPLAY_MEMORY_SIZE`**: Size of experience buffer
- **`MIN_REPLAY_MEMORY_SIZE`**: Minimum experiences before training starts

---

## 📝 Notes

- Training is computationally intensive - GPU recommended
- CARLA simulator requires significant resources (CPU and RAM)
- Training can take many hours/days depending on hardware
- Checkpoints allow resuming interrupted training sessions
- Model performance improves gradually - monitor TensorBoard for progress
- Adjust hyperparameters in `settings.py` to tune training behavior

---

## 🔗 Resources

- **CARLA Simulator**: https://carla.org/
- **CARLA 0.9.6 Download**: https://carla.org/releases/
- **CARLA Documentation**: https://carla.readthedocs.io/

---

## 📄 License

See `LICENSE` file for details.
