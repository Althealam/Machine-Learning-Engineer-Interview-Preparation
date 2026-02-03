# ✅ GenAI Knowledge Checklist

---

## 1️⃣ Architecture
- [☑️] Pre-Norm, Post-Norm, Double Norm
- [☑️] LayerNorm vs RMSNorm
- [☑️] Activation Function: ReLU, GeLU, Gated Activations
- [☑️] Position Embeddings Variants: RoPE

---

## 2️⃣ Mixture of Experts (MoE)
- [☑️] Conceptual
* What is a MoE model?
* How does and MoE differ from a standard dense Transformer?
* What is an "export" in a MoE architecture?
* What does it mean that MoE models are sparse?
* Why can MoEs increase parameter count without increasing FLOPs?
* Where are MoE layers typically placed in Transformers?
- [☑️] Architecture Design
* What is router in MoE?
* What is token-choice routing vs expert-choice routing?
* What does "top-k routing" mean?
* Why is k usually small?
* What are shared experts and why are they used?
- [☑️] Forward Pass Mechanics
* How does a token decide which experts to use?
* How are export outputs combined?
* What are gating weights and how are they normalized?
* What happends to experts that are not selected?
* How does MoE inference differ from dense inference?
* How does expert capacity limit affect the forward pass?

Pretraining
- [ ] Language modeling objective (next-token prediction)
- [ ] Cross-entropy loss for LLMs
- [ ] Teacher forcing
- [ ] Curriculum & data mixture

## 2️⃣ Alignment
- [ ] Instruction tuning (SFT)
- [ ] Preference learning (RLHF / RFT / DPO)
- [ ] Reward model basics
- [ ] Why alignment ≠ accuracy