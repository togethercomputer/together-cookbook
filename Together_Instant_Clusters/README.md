# Together Instant Clusters

Scale your AI workloads with Kubernetes-based distributed training on Together's infrastructure.

## 🚀 Examples

### [GRPO BlackJack Training](GRPO_BlackJack/)
Full PyTorch stack (OpenEnv + TorchForge + Monarch) on Together Instant Clusters to train Qwen 1.5B with GRPO to play BlackJack.

**What's Included:**
- Kubernetes manifests for PVC, workspace, and BlackJack server
- GRPO training pipeline with vLLM policy serving
- Distributed training with TitanTrainer
- Rollout generation and replay buffer management
- Comprehensive logging and monitoring

**Infrastructure:**
- Requires 1x H100 node minimum
- Shared volume support
- kubectl access with KUBECONFIG

**Learn More:** [GRPO BlackJack README](GRPO_BlackJack/README.md)

## 🎯 Key Concepts

**Together Instant Clusters**
- Kubernetes-based GPU clusters
- Elastic scaling
- Pre-configured with ML frameworks
- Shared persistent volumes

**Training Stack**
- **OpenEnv**: Environment interface (e.g., BlackJack, Chess, Code Interpreter)
- **TorchForge**: Distributed training orchestration
- **vLLM**: Fast policy model serving
- **TitanTrainer**: RL training loop

**GRPO (Group Relative Policy Optimization)**
- Reinforcement learning algorithm
- On-policy method for LLM training
- Reward-based fine-tuning
- Applicable to games, tool use, reasoning

## 🔧 Getting Started

1. **Create Cluster**: [api.together.ai/clusters](https://api.together.ai/clusters)
2. **Configure kubectl**: Set `KUBECONFIG=/path/to/your/kubeconfig`
3. **Deploy Example**: Follow [GRPO BlackJack](GRPO_BlackJack/README.md)

## 📊 Use Cases

- **RL Training**: Train models on games, simulations, or tool use
- **Distributed Fine-tuning**: Scale fine-tuning across multiple GPUs
- **Large-scale Inference**: Serve models with high throughput
- **Custom Training Loops**: Deploy proprietary training algorithms

## 🌐 Resources

- [Together Instant Clusters Docs](https://docs.together.ai/docs/clusters)
- [OpenEnv Repository](https://github.com/meta-pytorch/OpenEnv)
- [TorchForge Documentation](https://github.com/pytorch/torchforge)

## Prerequisites

- Together Instant Cluster access
- Kubernetes knowledge
- Understanding of distributed training
- Familiarity with RL concepts (for GRPO example)
