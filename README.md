# No-Limit-Texas-Hold-em

In this project, we conduct an ablation study comparing two variants of the Proximal Policy Optimization (PPO) algorithm—PPO-Clip and PPO-KL—within the context of No-Limit Texas Hold'em using the RLCard environment. We develop a rich state representation, implement domain-specific reward shaping, and introduce a self-play training framework to expose agents to a curriculum of increasingly sophisticated opponents. Our results show that PPO-KL converges to a stable, exploitative strategy with a high win rate but lower average rewards, while PPO-Clip continues exploring diverse strategies and achieves higher reward peaks at the cost of consistency. This study illustrates how different constraint mechanisms in PPO shape learning dynamics and strategic behavior in complex, imperfect-information games like poker.

---

## Training

Set up training hyperparameters in the `/configs` directory:

- `default_config.py`  – Common hyperparameters for the training process  
- `ppo_clip_config.py` – Parameters specific for PPO-Clip agent  
- `ppo_kl_config.py`   – Parameters specific for the PPO-KL agent  

---

### Run ablation study

Run `main.py` with arguments `--mode ablation --episodes <training_episodes>`

#### Example:

```bash
python main.py --mode ablation --episodes 25000
```

This runs and evaluates training for both PPO-Clip and PPO-KL agents.
The checkpoints of models during training are saved under /checkpoints

### Run RL agent

To run a trained agent:

```bash
python play_poker.py
```

By default this loads the model with the best win rate and uses the PPO-Clip agent to play 3 games of Poker. To change this edit the argument in the play_poker.py file.
