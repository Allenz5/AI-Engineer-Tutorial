# AI-Engineer-Tutorial
## Post-Training
### Supervised Fine-Tuning
### LoRA and QLoRA
All matrices can be decomposed using Singular Value Decomposition (SVD). A low-rank matrix has fewer non-zero singular values than the number of rows, which means its information can be represented using fewer “directions” in space. In LLMs, the model parameters are typically high-rank matrices, as they need to encode rich and diverse information across many dimensions. However, during fine-tuning, the delta matrix which represents updates to the original model parameters is often low-rank because the new information is task-specific and relatively narrow in scope.  
  
As a result, we can focus on only the top singular directions, which capture most of the meaningful variation in the fine-tuning signal. Since the delta matrix is low-rank, we don’t need to store or compute the full SVD (U, Σ, Vᵀ). Instead, we can approximate the update using the product of two smaller matrices, A and B, which is sufficient to represent the essential information efficiently.  
  
We still use backpropagation to train the two matrices A and B. The main hyperparameters include the scaling factor and the rank r. A larger r means we want to capture more directions in the parameter space, which slows down training. The scaling factor controls how much influence the delta matrix has on the original parameters. LoRA can be applied to multiple layers in a Transformer model, most commonly to the query and value projection matrices.  
  
In LLMs, parameters are typically stored as floating-point numbers. Mathematically, there's a technique called quantization, which allows us to represent a float matrix using integers while preserving most of its information. In QLoRA, we apply quantization to reduce the memory footprint during fine-tuning by storing the model in an integer format (e.g., int4 or int8). When performing inference, the model is converted back to float as needed. This approach allows efficient fine-tuning of large models with significantly reduced hardware requirements.  

### Catastrophic Forgetting
Catastrophic forgetting refers to the phenomenon where a large language model (LLM) loses previously learned knowledge and suffers performance degradation after being fine-tuned. This typically happens because the updated parameters drift too far from the original pre-trained parameters. To mitigate this, many techniques are designed to constrain the parameter updates and preserve the model’s original knowledge. Mathematically, this can be achieved by reducing the learning rate, applying regularization methods such as KL-divergence (as in GRPO) or L2 regularization, and using Elastic Weight Consolidation (EWC). Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA also help prevent catastrophic forgetting by updating only a small subset of parameters. In LoRA, the scaling factor further controls the extent of parameter modification. In addition to model-level techniques, careful design of the training pipeline can also reduce the risk of catastrophic forgetting from a systems perspective.  
### PPO
### DPO
### DeepSeek R1 && GRPO
DeepSeek R1 exclusively uses questions with verifiable answers, such as coding and math challenges. They initially verified that the reasoning process (Chain of Thought) becomes more sophisticated, reasoning time increases, and accuracy improves when a large language model is post-trained solely on questions with known correct answers. This was the *“Aha” moment* for DeepSeek R1. They created a rule-based reward system for reinforcement learning. 
  
For each prompt, a group of responses is generated and scored using a reward model. The average score across these responses is used as the baseline (Monte Carlo Method), and the advantage of each response is computed relative to this baseline. The advantage is then used to calculate the loss for each response, which guides the model’s gradient update. In this way, the average score reflects the model’s current capability and serves as a baseline. The model is then updated to move toward behaviors we prefer and away from those we do not.  
  
Unlike DPO and PPO, Group Relative Policy Optimization (GRPO) also uses KL-Divergence with clipping to prevent the model from deviating too far from the initial model and experiencing catastrophic forgetting. KL-Divergence measures the difference between two models.  

### Offline

## Multimodal
### Intro
There are currently three mainstream approaches to multimodality:  
1. vision-language connector  
2. cross-attention  
3. native multimodal model.  
### CLIP and ViT
### Vision-language connector
vision-language connector的主要模型包括LLaVA和BLIP，他们的架构是在图片理解模型如CLIP-ViT和LLM之间建立一个projector，使LLM可以理解图片内容，并执行与图片相关的任务。
### Cross-attention
### Native Multimodal Model
## LLM Foundation
### Self-Attention
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
In Transformer architectures, the Mixture of Experts (MoE) module typically replaces the Feed-Forward Network (FFN). An MoE consists of a router and multiple experts. The router selects one or two experts per token and forwards the token to those experts for FFN processing.  
  
This approach increases model capacity without significantly raising computational cost. Each expert may specialize in certain patterns, though the specialization is abstract and not as clearly defined as subject areas.  
  
During training, the MoE participates in backpropagation. The router gradually learns to route tokens to the most appropriate experts. Both LLaMA 4 and DeepSeek adopt MoE architectures, with a shared expert included to handle general-purpose tasks.  
  
## Evaluations
## RAG
## Infra
## Papers and Tech reports
### Gemini Tech Report
### Ernie4.5 Tech Report
### DeepSeek R1 Tech Report
### Multimodality GRPO
