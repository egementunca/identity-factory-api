"""
Generator Registry.
Central registry for available circuit generators.
"""

import logging
from typing import Dict, List, Optional, Type

from .base import CircuitGenerator, GeneratorInfo

logger = logging.getLogger(__name__)


class GeneratorRegistry:
    """
    Registry of available circuit generators.
    Provides discovery and instantiation of generators.
    """

    _instance: Optional["GeneratorRegistry"] = None

    def __init__(self):
        self._generators: Dict[str, CircuitGenerator] = {}
        self._generator_classes: Dict[str, Type[CircuitGenerator]] = {}

    @classmethod
    def get_instance(cls) -> "GeneratorRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtin_generators()
        return cls._instance

    def _register_builtin_generators(self):
        """Register built-in generators."""
        try:
            from .sat_forward_inverse import SATForwardInverseGenerator

            self.register(SATForwardInverseGenerator)
        except ImportError as e:
            logger.warning(f"Could not load SAT Forward/Inverse generator: {e}")

        try:
            from .eca57_identity_miner import ECA57IdentityMinerGenerator

            self.register(ECA57IdentityMinerGenerator)
        except ImportError as e:
            logger.warning(f"Could not load ECA57 Identity Miner generator: {e}")

        try:
            from .go_enumerator import GoEnumeratorGenerator

            self.register(GoEnumeratorGenerator)
        except ImportError as e:
            logger.warning(f"Could not load Go Enumerator generator: {e}")

        try:
            from .skeleton_generator import SkeletonGenerator

            self.register(SkeletonGenerator)
        except ImportError as e:
            logger.warning(f"Could not load Skeleton generator: {e}")

    def register(self, generator_class: Type[CircuitGenerator]):
        """Register a generator class."""
        instance = generator_class()
        info = instance.get_info()
        self._generator_classes[info.name] = generator_class
        self._generators[info.name] = instance
        logger.info(f"Registered generator: {info.name} ({info.display_name})")

    def get_generator(self, name: str) -> Optional[CircuitGenerator]:
        """Get a generator by name."""
        return self._generators.get(name)

    def list_generators(self) -> List[GeneratorInfo]:
        """List all available generators."""
        return [g.get_info() for g in self._generators.values()]

    def get_generators_for_gate_set(self, gate_set: str) -> List[GeneratorInfo]:
        """Get generators that support a specific gate set."""
        return [
            g.get_info()
            for g in self._generators.values()
            if g.supports_gate_set(gate_set)
        ]

    def get_all_gate_sets(self) -> List[str]:
        """Get all supported gate sets across all generators."""
        gate_sets = set()
        for g in self._generators.values():
            info = g.get_info()
            gate_sets.update(info.gate_sets)
        return sorted(gate_sets)


def get_registry() -> GeneratorRegistry:
    """Get the generator registry singleton."""
    return GeneratorRegistry.get_instance()
