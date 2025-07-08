# AI-Engineer-Tutorial
## Pre-Training
### Self-Attention
## Understanding Self-Attention in the Transformer Architecture

The **self-attention mechanism** is at the heart of the Transformer architecture. It was introduced to address a long-standing challenge in sequence modeling: determining which parts of the input sequence are most relevant to a given token.  
  
In self-attention, the input sequence is first tokenized and embedded into vectors. These embeddings are enriched with **positional encodings** to preserve the order of tokens, forming a matrix of shape *(sequence length, embedding dimension)* — for example, *(n, 512)*. This matrix is then linearly projected into three distinct matrices:  
  
- **Q (Query)**  
- **K (Key)**  
- **V (Value)**  
  
using learned weight matrices **W<sub>Q</sub>**, **W<sub>K</sub>**, and **W<sub>V</sub>**.  
  
- The **Query vector** represents what the current token is looking for.  
- The **Key vector** indicates what information each token contains.  
- The **Value vector** holds the actual content to be aggregated.  
  
The core attention function computes how much each token should attend to others by comparing its Query with all Keys, typically using **scaled dot-product attention**:  
  
```math
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) \times V
```
This operation results in a new representation for each token that integrates not only its own meaning but also contextual information from other tokens in the sequence.  
  
To further enhance the model's capacity, **multi-head attention** is applied. Instead of performing a single attention operation, the model splits the input into multiple subspaces (or "heads"), performs attention in parallel, and concatenates the results. This allows the model to capture different types of relationships and interactions simultaneously.  
  
Finally, the output passes through a **feed-forward neural network (FFN)**, which applies non-linear transformations to each token independently, boosting the model's expressiveness. This combination of mechanisms allows Transformers to model complex dependencies in sequences without relying on recurrence or convolution.  
  
### Encoder & Decoder
The encoder-decoder architecture has been widely used in machine learning even before the introduction of the attention mechanism—for example, in sequence-to-sequence models based on recurrent neural networks. The Transformer model enhanced this architecture by introducing self-attention and cross-attention mechanisms, significantly improving its effectiveness for sequence modeling tasks.  
  
In the encoder, we apply self-attention to allow each token to attend to all other tokens in the input sequence. This enables the model to capture global context and semantic relationships across the entire input. In contrast, the decoder uses masked self-attention, which restricts each token to attend only to previous tokens in the output sequence. This masking is essential to prevent information leakage during autoregressive generation. Additionally, the decoder incorporates cross-attention to access the encoder’s output representations, effectively allowing the decoder to condition its output on the input sequence.  
  
This encoder-decoder setup is particularly effective for tasks like machine translation, where the encoder is responsible for understanding the source sentence, and the decoder generates a grammatically and semantically correct target sentence in another language.  
  
Notably, GPT adopts a decoder-only architecture. It is trained in an autoregressive manner, meaning it generates text one token at a time, conditioning only on previously generated tokens. As such, it employs only masked self-attention and does not have access to future tokens during training or inference. This design ensures that the model avoids information leakage and is optimized for generation tasks rather than bidirectional understanding.  
  
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
