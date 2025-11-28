# GRPO BlackJack on Together Instant Cluster

This recipe shows how to run the full PyTorch stack (OpenEnv + TorchForge + Monarch) on a Together Instant Cluster and train Qwen with GRPO to play BlackJack. It is inspired by the OpenEnv example from this [repo](https://github.com/meta-pytorch/OpenEnv/tree/main/examples/grpo_blackjack).

## Prerequisites

- Together Instant Cluster (Kubernetes) with at least **1x H100 node**
- Shared volume (e.g. `together-openenv-integration`)
- `kubectl` configured (`KUBECONFIG=/path/to/your/kubeconfig`)
- Local clone of this repository

## Contents

```
OpenEnv_GRPO_BlackJack/
├── blackjack.yaml          # Forge training config
├── cluster_setup.sh        # Installs OpenEnv & TorchForge on the cluster
├── grpo_training.py        # Main training script
├── grpo_utils.py           # Helper utilities
└── k8s-manifests.yaml      # PVC + workspace pod + BlackJack server
```

## Step 1 – Apply Kubernetes manifests

```bash
cd together-cookbook/OpenEnv_GRPO_BlackJack
kubectl apply -f k8s-manifests.yaml
```

Before applying, update `k8s-manifests.yaml` if your cluster uses a different PersistentVolume or storage class. The sample manifest references `volumeName: together-openenv-integration`; change that to match the PV/PVC available in your cluster.

This creates:
- `openenv-storage` PVC (bound to whichever PV you set in the manifest; the sample uses `together-openenv-integration`)
- `grpo-workspace` pod (idle pod with PVC mounted at `/workspace`)
- `blackjack-server` deployment + service (OpenSpiel server on port 8004)


## Step 2 – Copy recipe files into the workspace PV

```bash
kubectl cp . grpo-workspace:/workspace/grpo-demo
```


## Step 3 – Install dependencies inside the cluster

```bash
kubectl exec -it grpo-workspace -- bash -c "
  cd /workspace/grpo-demo &&
  bash cluster_setup.sh
"
```

This clones OpenEnv & TorchForge under `/workspace`, installs `open_spiel`, vLLM, TorchForge, and all CUDA 12.8 wheels. Re-run this step any time you recreate the `grpo-workspace` pod.

## Step 4 – Run GRPO training

```bash
kubectl exec -it grpo-workspace -- bash -c "
  cd /workspace/grpo-demo &&
  export GRPO_TRAINING_STEPS=20 &&
  export GRPO_LOG_DIR=/workspace/grpo-demo/logs &&
  export OPENENV_PATH=/workspace/OpenEnv/src &&
  export PYTHONPATH=/workspace/OpenEnv/src:\$PYTHONPATH &&
  python grpo_training.py
"
```

What happens:
- Forge services (policy, trainer, ref model, reward, replay buffer) boot inside the workspace pod.
- The vLLM policy hits the OpenEnv blackjack server, produces rollouts, and streams them into the replay buffer.
- TitanTrainer consumes batches and runs GRPO for `GRPO_TRAINING_STEPS`.
- After every step, the script flushes aggregated metrics (look for `=== [global_reduce] - METRICS STEP N ===`) and pushes weights.

Logging:
- `stdout`: `/workspace/grpo-demo/logs/<timestamp>_grpo_run.log`.
- Structured logs: `/workspace/grpo-demo/logs/grpo_training_<timestamp>.log`.
- Game transcripts: `/workspace/grpo-demo/game_logs/games_<timestamp>.log`.


## Monitoring

```bash
kubectl logs -f deployment/blackjack-server

# Tail the most recent GRPO log inside the workspace
kubectl exec -it grpo-workspace -- bash -c "
  ls -t /workspace/grpo-demo/logs | head -n 1 | xargs -I{} tail -f /workspace/grpo-demo/logs/{}
"

# Watch aggregated Forge metrics (global_reduce blocks)
kubectl exec -it grpo-workspace -- bash -c "
  tail -f /workspace/grpo-demo/logs/$(ls -t /workspace/grpo-demo/logs | grep grpo_run | head -n1) | grep -n 'METRICS STEP'
"
```

## Cleanup

```bash
kubectl delete -f k8s-manifests.yaml
```


