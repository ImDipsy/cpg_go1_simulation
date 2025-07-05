# CPG-Go1 Simulation

[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2406.13419-blue)](https://doi.org/10.48550/arXiv.2406.13419)

A comprehensive simulation framework for quadruped robot locomotion based on Central Pattern Generator (CPG) networks and neural execution control for the Unitree Go1 robot.

## Overview

This project implements a biologically-inspired locomotion control system for quadruped robots, combining:

- **Central Pattern Generator (CPG)**: 8-neuron Stein oscillator network for rhythmic gait generation
- **Neural Execution Network**: Multi-layer perceptron (MLP) for translating CPG states to joint commands
- **Sensory Feedback System**: Real-time sensor integration for adaptive locomotion control
- **Physics Simulation**: High-fidelity PyBullet-based simulation environment

### Key Features

- 🦾 **Multi-Gait Support**: Walk, trot, pace, bound, and pronk gaits
- 🔄 **Smooth Gait Transitions**: Four transition strategies (Switch, Power Pair, Wait&Switch, Wait&Power Pair)
- 🎯 **Sensory Feedback**: IMU, camera, and contact sensor integration
- 🏃‍♂️ **Adaptive Control**: Real-time CPG parameter adjustment based on environmental feedback
- 🎮 **Multiple Environments**: Flat terrain, slopes, and figure-8 path following

## Project Structure

```
cpg_go1_simulation/
├── src/cpg_go1_simulation/                            # Core package
│   ├── config.py                                      # Configuration files and constants
│   ├── execution_neural_network/                      # Execution neural network
│   │   └── mlp.py                                     # Mlp network
│   ├── gait_and_transition/                           # Gait and gait transition
│   │   ├── export_cpg.py                              # CPG data generation
│   │   └── process_network.py                         # CPG-to-joint mapping
│   ├── sensor/                                        # Sensor-based control
│   │   ├── Quadruped_model/                           # Go1 robot model files and ground model
│   │   ├── quadruped_robot.py                         # Robot simulation environment
│   │   ├── realtime_controller.py                     # Real-time controller
│   │   ├── robot_controller.py                        # Basic controller
│   │   ├── sensor.py                                  # IMU and RGB Carame sensor
│   │   ├── reflexion_processor.py                     # Reflexion processor
│   │   └── visual_processor_rgb.py                    # Vision-based processor
│   └── stein/                                         # CPG network implementation
│       ├── base.py                                    # Abstract CPG base class
│       └── implementations.py                         # 8-neuron CPG implementation
├── examples/                                          # Demo scripts
│   ├── demo_video3_gait.py                            # Basic gait demonstration
│   ├── demo_video4_gait_transition.py                 # Gait transition demo
│   ├── demo_video5_sensory_feedback_path_following.py # Path following
│   ├── demo_video6_sensory_feedback_reflex.py         # Reflex loop
│   └── demo_video7_backward_control.py                # Backward locomotion
├── data/                                              # Generated data storage
├── resources/                                         # Pre-trained models
│   └── best_model/                                    # Neural network model
└── pyproject.toml                                     # Project configuration
```

## Installation

### Prerequisites

- Python 3.12 or higher
- Git

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/imdipsy/cpg_go1_simulation.git
   cd cpg_go1_simulation
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

3. **Verify installation** by running a demo:
   ```bash
   python examples/demo_video3_gait.py
   ```


## Quick Start Examples

### 1. Basic Gait Simulation

Run a simple trot gait simulation:

```bash
python examples/demo_video3_gait.py
```

### 2. Gait Transitions

Demonstrate smooth transitions between gaits:

```bash
python examples/demo_video4_gait_transition.py
```


### 3. Sensory Feedback Control

####  Path following using camera
Run adaptive control with sensor feedback:

```bash
python examples/demo_video5_sensory_feedback_path_following.py
```

#### Gait transition via reflex loop
```bash
python examples/demo_video6_sensory_feedback_reflex.py
```
### 4. Backward Locomotion
```bash
python examples/demo_video7_backward_control.py
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{liu2024eight,
  title={An eight-neuron network for quadruped locomotion with hip-knee joint control},
  author={Liu, Yide and Liu, Xiyan and Wang, Dongqi and Yang, Wei and others},
  journal={arXiv preprint arXiv:2406.13419},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email**: dipsy@zju.edu.cn(Xiyan Liu), yide_liu@zju.edu.cn(Yide Liu)
- 🐛 **Issues**: [GitHub Issues](https://github.com/imdipsy/cpg_go1_simulation/issues)


