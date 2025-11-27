"""
GRPO BlackJack Training Script for Together Instant Cluster

Converted from grpo_blackjack_tutorial.ipynb
This script trains a Qwen model to play BlackJack using GRPO.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path


def add_openenv_to_path():
    """Ensure OpenEnv source is on PYTHONPATH."""
    env_path = os.environ.get("OPENENV_PATH")
    candidates = []

    if env_path:
        candidates.append(Path(env_path))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "OpenEnv" / "src")

    for path in candidates:
        if path and path.exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            os.environ["OPENENV_PATH"] = str(path)
            return path

    raise FileNotFoundError(
        "Could not locate OpenEnv sources. "
        "Set OPENENV_PATH to the directory containing OpenEnv/src."
    )

# Import from grpo_utils
OPENENV_SRC = add_openenv_to_path()


def configure_logging() -> Path:
    """Configure dual stream/file logging and return the log file path."""
    default_dir = Path(__file__).parent / "logs"
    log_dir = Path(os.environ.get("GRPO_LOG_DIR", default_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"grpo_training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


LOG_PATH = configure_logging()
LOGGER = logging.getLogger("grpo_training")

# Import after logging setup so Forge components inherit configuration
from grpo_utils import setup_forge_training


async def main():
    """Run GRPO training."""
    
    LOGGER.info("=" * 70)
    LOGGER.info("GRPO BlackJack Training on Together Instant Cluster")
    LOGGER.info("=" * 70)
    
    # Environment setup for Monarch/Torchstore
    conda_prefix = os.environ.get('CONDA_PREFIX', sys.prefix)
    lib_path = f"{conda_prefix}/lib"
    
    if 'LD_LIBRARY_PATH' in os.environ:
        if lib_path not in os.environ['LD_LIBRARY_PATH']:
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = lib_path
    
    LOGGER.info("✅ Environment configured")
    
    # Setup Forge training infrastructure
    LOGGER.info("🏗️ Initializing Forge infrastructure...")
    LOGGER.info("This will:")
    LOGGER.info("  • Load the Qwen model")
    LOGGER.info("  • Initialize vLLM inference servers")
    LOGGER.info("  • Setup distributed training (TorchTitan)")
    LOGGER.info("  • Create replay buffer and reference model")
    LOGGER.info("⏳ This may take 1-2 minutes...")
    
    config_path = str(Path(__file__).parent / "blackjack.yaml")
    trainer = await setup_forge_training(config_path)
    
    LOGGER.info("✅ Ready to train!")
    
    # Run training
    train_steps = int(os.environ.get("GRPO_TRAINING_STEPS", "20"))
    LOGGER.info("🚀 Starting GRPO training!")
    LOGGER.info("=" * 70)
    LOGGER.info("Training for %s steps (set GRPO_TRAINING_STEPS to override)", train_steps)
    LOGGER.info("=" * 70)

    results = await trainer.run(steps=train_steps)
    
    LOGGER.info("=" * 70)
    LOGGER.info("🎉 Training complete!")
    LOGGER.info("=" * 70)
    
    # Cleanup
    await trainer.shutdown()
    LOGGER.info("✅ Shutdown complete")
    LOGGER.info("Logs written to %s", LOG_PATH)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user")
    except Exception as exc:
        message = str(exc)
        benign_shutdown = (
            "actor mesh is stopped due to proc mesh shutdown" in message
            or "MeshFailure" in message
        )
        if benign_shutdown:
            LOGGER.warning("Ignoring benignshutdown exception: %s", message)
            LOGGER.info("Exit cleanly despite benign shutdown error")
        else:
            raise

