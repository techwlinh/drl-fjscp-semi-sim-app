import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.meta.ga import (
    GAConfig,
    Chromosome,
    ScheduledTask,
    FJSPDecoder,
    GAOptimizer,
    export_schedule_results,
    main,
)

__all__ = [
    "GAConfig",
    "Chromosome",
    "ScheduledTask",
    "FJSPDecoder",
    "GAOptimizer",
    "export_schedule_results",
    "main",
]

if __name__ == "__main__":
    main()
