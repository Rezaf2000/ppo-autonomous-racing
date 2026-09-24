# PPO Autonomous Racing

A compact PyTorch implementation of Proximal Policy Optimization for the Gym CarRacing environment. The upstream repository includes a demo, report, and saved policy weights.

## Task and policy

A convolutional network processes four stacked image frames and approximates the policy and value function. PPO updates the policy; generalized advantage estimation estimates advantages. The policy uses a Beta distribution for bounded continuous controls.

## Repository map

| Path | Purpose |
| --- | --- |
| `ppo.py`, `memory.py` | PPO implementation and rollout storage |
| `main.py`, `config.toml` | Training entry point and configuration |
| `demo.py` | Playback using a saved policy |
| `games/`, `scripts/` | Environment and supporting scripts |
| `extra/` | Upstream report, weights, and demo GIF |

## Upstream demo and results

![Upstream racing demonstration](extra/demo.gif)

The original project reports **909.48 ± 10.30** mean score over 100 episodes after 5,000 training steps. This is an upstream claim, not a new evaluation. Its `extra/final_weights.pt` can be used with `python demo.py --ckpt extra/final_weights.pt --delay_ms 0` where dependencies are available.

## Source and license

Based on and adapted from [JinayJain/deep-racing](https://github.com/JinayJain/deep-racing). See the [original README](UPSTREAM_README.md), [technical report](extra/report.pdf), and retained [MIT license](LICENSE). No training or demo playback was performed for this fork.