### Do you really need post-training?
| Use Cases                                                                 | Methods                                              | Characteristics                                                                                   |
|---------------------------------------------------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Follow a few instructions (do not discuss XXX)                            | Prompting                                            | Simple yet brittle: models may not always follow all instructions                                 |
| Query real-time database or knowledgebase                                 | Retrieval-Augmented Generation (RAG) or Search       | Adapt to rapidly-changing knowledgebase                                                           |
| Create a medical LLM / Cybersecurity LLM                                  | Continual Pre-training + Post-training               | Inject large-scale domain knowledge (>1B tokens) not seen during pre-training                     |
| Follow 20+ instructions tightly; Improve targeted capabilities            | Post-training                                        | Reliably change model behavior & improve targeted capabilities; May degrade other capabilities if not done right |

---

### SFT, DPO and RFT
| Method | Data Format | Objective | Learning Focus | Use Cases |
|--------|-------------|-----------|----------------|-----------|
| **SFT** | `Input + Output` | Mimic responses | Teaches the model *what* to output | Structured tasks like function calls, learning stop tokens, constrained outputs |
| **DPO** | `Input + (Good Output, Bad Output)` | Learn to prefer better responses | Teaches the model *which output is better* among alternatives | Preference tuning: helpfulness, tone, safety |
| **RFT** | `Input + Reward Scoring Model` | Optimize response quality via rewards | Teaches the model *how to optimize* its thinking process to earn high reward | Complex tasks like coding, multi-turn dialogue |

---

### Supervised Fine-Tuning
Supervised Fine-Tuning (SFT) trains a language model to generate ideal responses to prompts by **imitating human-annotated examples**. It does this by minimizing the **negative log-likelihood** of the correct responses using a cross-entropy loss.  
```math
\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log \left(p_\theta(\text{Response}(i) \mid \text{Prompt}(i))\right)
```

Where:
- N: Number of training examples  
- p_theta: Model with parameters θ  
- Prompt(i): The input for the i-th training example  
- Response(i): The ideal response for the i-th training example  

This equation encourages the model to assign **high probability to the correct full response**. Each response is a sequence of tokens:  
```math
\text{Response}^{(i)} = (y_1^{(i)}, y_2^{(i)}, \dots, y_{T_i}^{(i)})
```
  
Then the SFT loss becomes:  

```math
\mathcal{L}_{\text{SFT}} = - \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log p_\theta\left(y_t^{(i)} \mid \text{Prompt}^{(i)}, y_1^{(i)}, \dots, y_{t-1}^{(i)}\right)
```
  
This means:  
- For each sample i, split the response into tokens y₁, y₂, ..., y_T
- At each time step t, compute the probability of the correct token y_t
- Condition on both the prompt and the previous tokens y₁ to y_{t-1}
- The total loss is the negative log-likelihood summed over all tokens in all samples  

SFT teaches a language model *what to say* by showing it high-quality human responses. During training, the model learns to mimic these responses token by token. The training objective is typically cross-entropy loss, which decreases as the model better predicts the target output. This is a form of imitation learning, and because the model replicates what it sees, using high-quality data is crucial (1,000 high quality data > 1,000,000 mixed quality data).
 

SFT training typically uses between 1K and 1B examples. When training, multiple prompts are grouped into batches, allowing parallel computation and gradient updates. Gradient accumulation is used for delaying updates until several smaller batches have been processed. Large batch sizes accelerate training, while small batch sizes reduce memory usage. Delayed gradient updates can also help stabilize training by smoothing optimization steps.

---

### LoRA and QLoRA
All matrices can be decomposed using Singular Value Decomposition (SVD). A low-rank matrix has fewer non-zero singular values than the number of rows, which means its information can be represented using fewer “directions” in space. In LLMs, the model parameters are typically high-rank matrices, as they need to encode rich and diverse information across many dimensions. However, during fine-tuning, the delta matrix which represents updates to the original model parameters is often low-rank because the new information is task-specific and relatively narrow in scope.  
  
As a result, we can focus on only the top singular directions, which capture most of the meaningful variation in the fine-tuning signal. Since the delta matrix is low-rank, we don’t need to store or compute the full SVD (U, Σ, Vᵀ). Instead, we can approximate the update using the product of two smaller matrices, A and B, which is sufficient to represent the essential information efficiently.  
  
We still use backpropagation to train the two matrices A and B. The main hyperparameters include the scaling factor and the rank r. A larger r means we want to capture more directions in the parameter space, which slows down training. The scaling factor controls how much influence the delta matrix has on the original parameters. LoRA can be applied to multiple layers in a Transformer model, most commonly to the query and value projection matrices.  
  
In LLMs, parameters are typically stored as floating-point numbers. Mathematically, there's a technique called quantization, which allows us to represent a float matrix using integers while preserving most of its information. In QLoRA, we apply quantization to reduce the memory footprint during fine-tuning by storing the model in an integer format (e.g., int4 or int8). When performing inference, the model is converted back to float as needed. This approach allows efficient fine-tuning of large models with significantly reduced hardware requirements.  

---

### Catastrophic Forgetting

Catastrophic forgetting refers to the phenomenon where a large language model (LLM) loses previously learned knowledge and suffers performance degradation after being fine-tuned. This typically happens because the updated parameters drift too far from the original pre-trained parameters. To mitigate this, many techniques are designed to constrain the parameter updates and preserve the model’s original knowledge. Mathematically, this can be achieved by reducing the learning rate, applying regularization methods such as KL-divergence (as in GRPO) or L2 regularization, and using Elastic Weight Consolidation (EWC). Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA also help prevent catastrophic forgetting by updating only a small subset of parameters. In LoRA, the scaling factor further controls the extent of parameter modification. In addition to model-level techniques, careful design of the training pipeline can also reduce the risk of catastrophic forgetting from a systems perspective.  

---

### DPO

DPO minimizes the contrastive loss which penalize negative response and encourages positive response.  
```math
\mathcal{L}_{\text{DPO}} = - \log \sigma \Bigg( 
\beta \Big( 
\log \frac{\pi_\theta(y_{\text{pos}} \mid x)}{\pi_{\text{ref}}(y_{\text{pos}} \mid x)} 
- 
\log \frac{\pi_\theta(y_{\text{neg}} \mid x)}{\pi_{\text{ref}}(y_{\text{neg}} \mid x)} 
\Big) 
\Bigg)
```
This means:
- If the model prefers to generate the positive response, the difference becomes larger.  
- Beta is a hyperparameter to adjust the effect.  
- Sigmoid converts the difference into a probability between 0 and 1 to guide the loss function.  
- Larger difference means the sigmoid approaches 1, so −log approaches 0. Zero or negative difference makes the sigmoid less than 0.5, so −log becomes large.
  
DPO is best used for changing model behavior with small adjustments like identity, multilingual ability, instruction following, and safety. It is also effective for improving model capabilities, performing better than SFT due to its contrastive nature, and online DPO works better than offline for capability improvement.  

High-quality DPO data can be curated through correction, where the original model’s response is treated as negative and an improved version as positive, or through online/on-policy methods, where multiple responses are generated and the best is chosen as positive and the worst as negative using reward functions or human judgment. To avoid overfitting, ensure positive samples do not rely on shortcuts, such as always containing a few special words.  

---

### Policy Gradient Update

```math
\theta \leftarrow \theta + \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \cdot G_t
```
- $a_t$: action at timestep $t$ 
- $s_t$: state at timestep $t$  
- $\pi_{\theta}$: policy parameterized by $theta$  
- $G_t$: return from timestep $t$  

This is the standard **policy gradient update rule** in reinforcement learning. It updates the policy by weighting the log-probability of actions with their corresponding returns.  

The key component is $G_t$:  
- If $G_t > 0$, the action is reinforced (the policy is rewarded).  
- If $G_t < 0$, the action is discouraged (the policy is penalized).  

In **PPO** and **GRPO**, $G_t$ is computed using a reward together with a baseline. The baseline is needed because sometimes the state itself already has a high expected return regardless of the action. Without a baseline, we might overestimate the contribution of a specific action.  

- **PPO**: uses a **trainable value model** as the baseline, estimating the expected return for each state.  
- **GRPO**: uses the **average reward of a group of generated answers** as the baseline, which is shared across states within that group.  
  
---

### PPO

The **reward model**’s goal is to output a scalar score given a prompt and an answer.

```math
\mathcal{L}_{\text{RM}}(\phi)
= - \mathbb{E}_{(x, y^{+}, y^{-})}
\left[
\log \sigma\!\big( r_\phi(x, y^{+}) - r_\phi(x, y^{-}) \big)
\right]
```

- The goal is maximize the difference between positive answer and negative answer  
- Sigmoid Converts the score difference into a probability of preference. 
- E means averaging the training batch to decrease variance (mini-batch SGD)
- Larger difference means the sigmoid approaches 1, so −log approaches 0. Zero or negative difference makes the sigmoid less than 0.5, so −log becomes large.
  
The **value model** estimates the **expected reward** (baseline) given a state or prompt \(s\). It helps reduce variance in policy gradient updates by providing a reference value for computing the **advantage**. Without this baseline, it would be unclear whether a reward is actually good or bad, making it hard to determine the correct update direction. For example, generating “The” at the beginning already has a high baseline, so we don’t want to encourage it any further

```math
\mathcal{L}_{\text{value}}(\theta)
= \mathbb{E}_{t}\left[ \big( V_\theta(s_t) - R_t \big)^2 \right]
```


- **$V_\theta(s_t)$:** The predicted value (expected reward) for state \(s_t\).  
- **$R_t$:** The scalar reward from the reward model. We cannot get exact $R_t$ for each state in trajectory from final reward. We typically use Monte Carlo return, GAE or Temporal Difference (TD) Learning to approximate $R_t$.
- **Goal:** Minimize the difference between predicted value and actual return, so the value model becomes a good baseline.  
- If the value model predicts too low but the return is high → it updates upward.  
- If it predicts too high but the return is low → it updates downward.  
- Over time, $V_\theta(s)$ converges to the **average reward** for each state.  


Once the reward model and value model are available, PPO updates the policy using Advantage while preventing large policy updates.  

Define the probability ratio:  
```math
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
```

The clipped surrogate loss is:  
```math
\mathcal{L}_{\text{PPO}}(\theta) =
\mathbb{E}_t\Big[
\min\big(
r_t(\theta) \cdot A_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t
\big)
\Big]
```

The advantage is:  
```math

A_t = R_t - V_\theta(s_t)

```
- $\pi_\theta(a_t | s_t)$: current policy probability of action \(a_t\) at state \(s_t\)  
- $\pi_{\theta_{\text{old}}}(a_t | s_t)$: reference (previous) policy probability  
- **Ratio:** This is an **importance weight** that re-weights actions sampled from the old policy so they can be correctly evaluated under the new policy.  
  - If $r_t > 1$: the new policy assigns higher probability to this action than before.  
  - If $r_t < 1$: the new policy assigns lower probability.  
  - Importance weighting ensures the policy gradient remains unbiased even though samples were drawn from the old policy, making PPO a form of **off-policy correction**
- **Clipping:**  
  - Prevents excessively large updates when \(r_t\) moves too far from 1.  
  - Keeps training stable by limiting the change to within \([1-\epsilon, 1+\epsilon]\).  
- **$R_t$:** The scalar reward from the reward model. We cannot get exact $R_t$ for each state in trajectory from final reward. We typically use Monte Carlo return, GAE or Temporal Difference (TD) Learning to approximate $R_t$.
- If $A_t > 0$: action was **better than expected** → increase $\pi_\theta(a_t | s_t)$ probability.  
- If $A_t < 0$: action was **worse than expected** → decrease $\pi_\theta(a_t | s_t)$ probability.  

---

### GRPO
The **Group Relative Policy Optimization (GRPO)** objective extends PPO by using group-based advantages and a KL penalty.  

```math
\mathcal{L}_{\text{GRPO}}(\theta) 
= \frac{1}{G} \sum_{i=1}^{G} 
\Big( 
\min \Big( 
\frac{\pi_{\theta}(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} A_i,\;
\text{clip}\!\left(\frac{\pi_{\theta}(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\epsilon, 1+\epsilon \right) A_i 
\Big) 
- \beta D_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{ref}}) 
\Big)
```

```math
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}
```

Where:  
- **Group Output**: Sample outputs in a group of size $G$. The policy is updated using the average loss across all sampled outputs in the group, which is why the term $\frac{1}{G} \sum_{i=1}^{G}$ appears.  
- **Ratio**: The importance weight is same as in PPO.  
- **KL Divergence**: KL Divergence term measures the distance between the current policy and the reference policy (often the policy before reinforcement learning). This term penalizes large deviations, ensuring the new policy does not drift too far from the initial one and helping to prevent catastrophic forgetting.  
- **Group-normalized Advantage**: The advantage is computed as the reward minus the average reward of the group, normalized by the group’s standard deviation.  


---

### Gradient Descent

- **Define the model function** with parameters $\theta$ (e.g., $f_\theta(x)$).  
- **Choose a loss function** $L(\theta)$ that measures the difference between predictions and true values.  
   - Example: Mean Squared Error (MSE):  
     $$
     L(\theta) = \frac{1}{N} \sum_{i=1}^N (f_\theta(x_i) - y_i)^2
     $$  
- **Compute the gradient** of the loss function with respect to $\theta$:  
   $$
   \nabla_\theta L(\theta)
   $$  
   In this step, we need to substitute data points to calculate the gradient. There are multiple ways to do this:
    - **Batch Gradient Descent (GD):** Uses the entire dataset to compute the gradient.  
    - **Stochastic Gradient Descent (SGD):** Uses a single sample per update.  
    - **Mini-batch SGD:** Uses a small batch of samples (most common in practice).  
- **Update parameters** in the opposite direction of the gradient:  
   $$
   \theta \leftarrow \theta - \eta \cdot \nabla_\theta L(\theta)
   $$  
   where $\eta$ is the learning rate.  
- **Repeat** until convergence (loss becomes small or parameters stabilize).  

- **Average vs. Sum:**  
  - Loss is usually written as an **average** over data points (not sum).  
  - This keeps the loss scale independent of dataset size and makes learning rate tuning easier.  

---

### Offline vs. Online
