# AI-Engineer-Tutorial
## Pre-Training
### Attention

### Encoder & Decoder
The encoder-decoder architecture has been widely used in machine learning even before the introduction of the attention mechanism—for example, in sequence-to-sequence models based on recurrent neural networks. The Transformer model enhanced this architecture by introducing self-attention and cross-attention mechanisms, significantly improving its effectiveness for sequence modeling tasks.  
  
In the encoder, we apply self-attention to allow each token to attend to all other tokens in the input sequence. This enables the model to capture global context and semantic relationships across the entire input. In contrast, the decoder uses masked self-attention, which restricts each token to attend only to previous tokens in the output sequence. This masking is essential to prevent information leakage during autoregressive generation. Additionally, the decoder incorporates cross-attention to access the encoder’s output representations, effectively allowing the decoder to condition its output on the input sequence.  
  
This encoder-decoder setup is particularly effective for tasks like machine translation, where the encoder is responsible for understanding the source sentence, and the decoder generates a grammatically and semantically correct target sentence in another language.  
  
Notably, GPT adopts a decoder-only architecture. It is trained in an autoregressive manner, meaning it generates text one token at a time, conditioning only on previously generated tokens. As such, it employs only masked self-attention and does not have access to future tokens during training or inference. This design ensures that the model avoids information leakage and is optimized for generation tasks rather than bidirectional understanding.  
  
### FFN
### MoE
## Post-Training
### Supervised Fine-Tuning
### PPO
### DPO
### DeepSeek R1 && GRPO
DeepSeek R1 exclusively uses questions with verifiable answers, such as coding and math challenges. They initially verified that the reasoning process (Chain of Thought) becomes more sophisticated, reasoning time increases, and accuracy improves when a large language model is post-trained solely on questions with known correct answers. This was the *“Aha” moment* for DeepSeek R1. They created a rule-based reward system for reinforcement learning. 
  
For each prompt, a group of responses is generated and scored using a reward model. The average score across these responses is used as the baseline (Monte Carlo Method), and the advantage of each response is computed relative to this baseline. The advantage is then used to calculate the loss for each response, which guides the model’s gradient update. In this way, the average score reflects the model’s current capability and serves as a baseline. The model is then updated to move toward behaviors we prefer and away from those we do not.  
  
Unlike DPO and PPO, Group Relative Policy Optimization (GRPO) also uses KL-Divergence with clipping to prevent the model from deviating too far from the initial model and experiencing catastrophic forgetting. KL-Divergence measures the difference between two models.  

### Offline
## Evaluations
## RAG
## Infra
