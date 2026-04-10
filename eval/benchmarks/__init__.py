from .base import EvalSample, BaseBenchmark
from .nextqa import NextQABenchmark
from .egoschema import EgoSchemaBenchmark
from .ovobench import OVOBenchmark
from .ego4d_nlq import Ego4DNLQBenchmark
from .liveqa import LiveQABenchmark

BENCHMARKS = {
    "nextqa": NextQABenchmark,
    "egoschema": EgoSchemaBenchmark,
    "ovobench": OVOBenchmark,
    "ego4d_nlq": Ego4DNLQBenchmark,
    "liveqa": LiveQABenchmark,
}
