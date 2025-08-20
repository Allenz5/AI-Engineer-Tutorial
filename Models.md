# Models
### DeepSeek R1

DeepSeek R1 exclusively uses questions with verifiable answers, such as coding and math challenges. They initially verified that the reasoning process (Chain of Thought) becomes more sophisticated, reasoning time increases, and accuracy improves when a large language model is post-trained solely on questions with known correct answers. This was the *“Aha” moment* for DeepSeek R1. They created a rule-based reward system for reinforcement learning. 
  
For each prompt, a group of responses is generated and scored using a reward model. The average score across these responses is used as the baseline (Monte Carlo Method), and the advantage of each response is computed relative to this baseline. The advantage is then used to calculate the loss for each response, which guides the model’s gradient update. In this way, the average score reflects the model’s current capability and serves as a baseline. The model is then updated to move toward behaviors we prefer and away from those we do not.  
  
Unlike DPO and PPO, Group Relative Policy Optimization (GRPO) also uses KL-Divergence with clipping to prevent the model from deviating too far from the initial model and experiencing catastrophic forgetting. KL-Divergence measures the difference between two models.  

---