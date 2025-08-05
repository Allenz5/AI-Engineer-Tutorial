# LLM-Learning-Notes
## Post-training
## Do you really need post-training?

| Use Cases                                                                 | Methods                                              | Characteristics                                                                                   |
|---------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Follow a few instructions (do not discuss XXX)                            | Prompting                                            | Simple yet brittle: models may not always follow all instructions                                 |
| Query real-time database or knowledgebase                                 | Retrieval-Augmented Generation (RAG) or Search       | Adapt to rapidly-changing knowledgebase                                                           |
| Create a medical LLM / Cybersecurity LLM                                  | Continual Pre-training + Post-training               | Inject large-scale domain knowledge (>1B tokens) not seen during pre-training                     |
| Follow 20+ instructions tightly; Improve targeted capabilities            | Post-training                                        | Reliably change model behavior & improve targeted capabilities; May degrade other capabilities if not done right |

| Method | Data Format | Objective | Learning Focus | Use Cases |
|--------|-------------|-----------|----------------|-----------|
| **SFT** | `Input + Output` | Mimic responses | Teaches the model *what* to output | Structured tasks like function calls, learning stop tokens, constrained outputs |
| **DPO** | `Input + (Good Output, Bad Output)` | Learn to prefer better responses | Teaches the model *which output is better* among alternatives | Preference tuning: helpfulness, tone, safety |
| **RFT** | `Input + Reward Scoring Model` | Optimize response quality via rewards | Teaches the model *how to optimize* its thinking process to earn high reward | Complex tasks like coding, multi-turn dialogue |
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
### DeepSeek R1 and GRPO
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
### ViT and CLIP
- https://www.youtube.com/watch?v=-TdDZ6C9rdg  
- https://www.youtube.com/watch?v=BxQep0qdeWA
  
ViT first splits the input image into fixed-size patches. Each patch is flattened into a 1D vector and linearly projected into a patch token. A learnable `[CLS]` token is prepended to the sequence of patch tokens, which is used to aggregate global information from the entire image. Additionally, a learnable position embedding is added to each token (including the `[CLS]` token) to retain spatial information. ViT uses an encoder-only architecture, meaning all tokens attend to each other via self-attention. After several layers of self-attention and feed-forward networks (FFNs), the `[CLS]` token encodes the overall semantic representation of the image. This token is then passed through an MLP head to produce the final output, typically for classification. It’s important to note that the original ViT was designed for image classification tasks, trained in a supervised manner using fixed category tags, and needed to be retrained for different tasks.  

The CLIP model consists of two main components: **an image encoder (typically ViT or ResNet)** and a **text encoder (a Transformer)**. It is trained using contrastive learning across batches. Specifically, for each batch, the model computes similarity scores between all image-text pairs and optimizes the embeddings so that matched pairs are pulled closer together in the embedding space, while mismatched pairs are pushed apart. The goal is to align images and their corresponding textual descriptions within a shared semantic embedding space. One of CLIP’s key advantages is that it only requires image-caption pairs, which are easy to collect at scale, and doesn’t rely on hard-coded task-specific labels or supervised fine-tuning for each downstream task. Since the training objective is to compare semantic similarity rather than generate captions or perform classification directly, the learning task becomes significantly easier. Thanks to this architecture, CLIP enables zero-shot image classification. We can construct prompts based on tags and compare the similarity between each prompt and the image to be classified. CLIP also has limitations. It struggles with abstract or complex instructions, and its performance can degrade on out-of-distribution images.   
### Vision-language connector
- https://www.youtube.com/watch?v=jjdKfk89yAM&t=1522s
- https://www.youtube.com/watch?v=bK9ns4DkxQg&t=2593s
- https://www.youtube.com/watch?v=k0DAtZCCl1w&t=2273s
  
Vision-language connector models, such as **LLaVA** and **BLIP**, aim to bridge image understanding models like **CLIP-ViT** with large language models (LLMs). They introduce a lightweight **projector** module that maps image embeddings into the LLM's embedding space, enabling the LLM to understand and respond to visual content. This architecture is efficient and easy to train, but the projector can become a bottleneck that limits overall performance.

**LLaVA** (Large Language and Vision Assistant) notably proposed a method for generating multimodal instruction-following datasets using GPT-4. Its core components include: a **CLIP-ViT image encoder** for visual feature extraction, a **projector** that transforms image embeddings into the LLM space, and a **language model** (e.g., Vicuna or LLaMA).  
#### Phase 1: Projector Alignment  
- **Input**: Image-caption pairs  
- **Process**: Images - CLIP encoder - projector - concatenation with an instruction prompt - LLM - Compare with ground truth  
- **Objective**: Match the generated caption with the ground truth caption using loss computation  
- **Optimization**: Only the **projector** is trained in this phase via backpropagation  
#### Phase 2: Instruction Tuning  
- **Input**: More complex GPT-4-generated instruction-following datasets containing images and tasks (e.g., question answering, dialogue)
- **Process**: Similar to Phase 1  
- **Objective**: Fine-tune the model so the LLM can perform vision-language tasks  
- **Optimization**: **Both the projector and the LLM** are trained. Parameter-efficient tuning methods like **LoRA** can be applied to accelerate and stabilize training.  
  
BLIP-2 follows a similar three-module structure: an image encoder, a vision-language connector (Q-Former), and a language model. Q-Former acts as a lightweight transformer module that bridges the gap between the image encoder the language model. Its core design involves **learnable queries** that extract high-level semantic information from noisy image embeddings. The process includes: **Cross-attention**: The learnable query tokens attend to the frozen image embeddings output by the CLIP image encoder, **Self-attention**: The query tokens interact with each other to refine the representation, **Feed-forward network (FFN)**: The processed queries are passed through an FFN, **Projection layer**: A linear layer projects the output into the embedding space used by the LLM. The intuition is that each query learns to focus on specific aspects of the image, such as "What objects are present?"  

#### Phase 1: Projector Alignment  
In the first training stage, BLIP-2 proposes three objectives to train Q-Former while keeping the image encoder and language model frozen:
- **Image-Text Contrastive Learning**  
   - Similar to CLIP.
   - A text encoder converts captions into embeddings.
   - The Q-Former is trained so that its image embeddings align with the corresponding text embeddings.
- **Image-Text Matching**  
   - A text encoder embeds captions.
   - A binary classifier predicts whether an image-text pair matches based on the alignment between image and text embeddings.
- **Image Caption Generation Loss**  
   - A frozen lightweight text decoder is used to generate captions from the image embeddings output by Q-Former.
   - The loss is computed against ground-truth captions.
   - Only the Q-Former (and possibly the image encoder) is trained, so that the generated embeddings are more compatible with the decoder.

#### Phase 2: Instruction Tuning 
- **Input**:  Instruction-following datasets generated by GPT-4, containing images paired with complex tasks such as visual question answering, captioning, reasoning, and dialogue.
- **Process**:  The image is passed through the frozen CLIP image encoder and Q-Former to produce query embeddings, which are then prepended to the language model input along with the textual instruction prompt.
- **Objective**:  Fine-tune the language model so it can understand and respond to instructions grounded in visual content. The model learns to generate appropriate textual outputs based on both the image and prompt.
- **Optimization**:  Only the **LLM is trained** in this phase. The CLIP encoder, Q-Former, and projector remain frozen. Parameter-efficient fine-tuning methods like **LoRA** can be used to reduce memory usage and stabilize training.  
### Gemini Multimodality
### Ernie4.5 Multimodality
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
