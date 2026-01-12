"""
ML feature extraction system for identity circuit factory.
Extracts features from circuits for complexity prediction and optimization suggestions.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional import for SAT synthesis
try:
    from circuit.circuit import Circuit

    CIRCUIT_AVAILABLE = True
except ImportError:
    Circuit = None
    CIRCUIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CircuitFeatures:
    """Represents extracted features from a circuit."""

    # Basic circuit properties
    width: int
    length: int
    gate_count: int

    # Gate composition features
    not_gate_count: int
    cnot_gate_count: int
    ccnot_gate_count: int
    multi_control_gate_count: int

    # Gate distribution features
    gate_density: float  # gates per qubit
    control_density: float  # controls per gate
    target_diversity: float  # unique targets / total gates

    # Structural features
    max_control_count: int
    avg_control_count: float
    control_entropy: float  # Shannon entropy of control distributions

    # Connectivity features
    qubit_connectivity: float  # fraction of qubit pairs that interact
    gate_interaction_graph_density: float

    # Complexity indicators
    non_triviality_score: float
    cancellation_difficulty: float

    # Gate pattern features
    consecutive_same_gate_count: int
    gate_repetition_ratio: float
    control_target_overlap: float

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary for ML models."""
        return {
            "width": float(self.width),
            "length": float(self.length),
            "gate_count": float(self.gate_count),
            "not_gate_count": float(self.not_gate_count),
            "cnot_gate_count": float(self.cnot_gate_count),
            "ccnot_gate_count": float(self.ccnot_gate_count),
            "multi_control_gate_count": float(self.multi_control_gate_count),
            "gate_density": self.gate_density,
            "control_density": self.control_density,
            "target_diversity": self.target_diversity,
            "max_control_count": float(self.max_control_count),
            "avg_control_count": self.avg_control_count,
            "control_entropy": self.control_entropy,
            "qubit_connectivity": self.qubit_connectivity,
            "gate_interaction_graph_density": self.gate_interaction_graph_density,
            "non_triviality_score": self.non_triviality_score,
            "cancellation_difficulty": self.cancellation_difficulty,
            "consecutive_same_gate_count": float(self.consecutive_same_gate_count),
            "gate_repetition_ratio": self.gate_repetition_ratio,
            "control_target_overlap": self.control_target_overlap,
        }


class MLFeatureExtractor:
    """Extracts ML features from circuits."""

    def __init__(self):
        self.feature_cache = {}  # Simple cache for extracted features

    def extract_features(self, circuit_id: int, circuit: Circuit) -> "CircuitFeatures":
        """
        Extract a comprehensive set of features from a circuit.

        Args:
            circuit_id: The ID of the circuit.
            circuit: The Circuit object to analyze.

        Returns:
            CircuitFeatures object with all extracted features
        """
        width = circuit.width()
        gates = circuit.gates()
        gate_count = 0
        depth = 0

        if gates:
            gate_count = len(gates)
            depth = self._calculate_depth(gates, width)

        # Basic properties
        length = len(circuit)

        # Gate composition analysis
        gate_composition = self._analyze_gate_composition(gates)

        # Gate distribution features
        gate_density = gate_count / width if width > 0 else 0
        control_density = (
            sum(len(controls) for controls, _ in gates) / gate_count
            if gate_count > 0
            else 0
        )
        # Ensure target is hashable (convert to tuple if list)
        target_diversity = (
            len(
                set(
                    tuple(target) if isinstance(target, list) else target
                    for _, target in gates
                )
            )
            / gate_count
            if gate_count > 0
            else 0
        )

        # Structural features
        control_counts = [len(controls) for controls, _ in gates]
        max_control_count = max(control_counts) if control_counts else 0
        avg_control_count = np.mean(control_counts) if control_counts else 0
        control_entropy = self._compute_entropy(control_counts)

        # Connectivity features
        qubit_connectivity = self._compute_qubit_connectivity(gates, width)
        gate_interaction_density = self._compute_gate_interaction_density(gates, width)

        # Complexity indicators
        non_triviality_score = self._estimate_non_triviality(gates, width)

        cancellation_difficulty = self._estimate_cancellation_difficulty(gates)

        # Gate pattern features
        consecutive_same_gate = self._count_consecutive_same_gates(gates)
        gate_repetition = self._compute_gate_repetition_ratio(gates)
        control_target_overlap = self._compute_control_target_overlap(gates)

        # Feature dictionary
        features = {
            "width": float(width),
            "gate_count": float(gate_count),
            "depth": float(depth),
            "length": float(length),
            "not_count": float(gate_composition["not_count"]),
            "cnot_count": float(gate_composition["cnot_count"]),
            "ccnot_count": float(gate_composition["ccnot_count"]),
            "multi_control_count": float(gate_composition["multi_control_count"]),
            "gate_density": gate_density,
            "control_density": control_density,
            "target_diversity": target_diversity,
            "max_control_count": float(max_control_count),
            "avg_control_count": avg_control_count,
            "control_entropy": control_entropy,
            "qubit_connectivity": qubit_connectivity,
            "gate_interaction_graph_density": gate_interaction_density,
            "non_triviality_score": non_triviality_score,
            "cancellation_difficulty": cancellation_difficulty,
            "consecutive_same_gate_count": float(consecutive_same_gate),
            "gate_repetition_ratio": gate_repetition,
            "control_target_overlap": control_target_overlap,
        }

        return CircuitFeatures(
            width=width,
            length=length,
            gate_count=gate_count,
            not_gate_count=gate_composition["not_count"],
            cnot_gate_count=gate_composition["cnot_count"],
            ccnot_gate_count=gate_composition["ccnot_count"],
            multi_control_gate_count=gate_composition["multi_control_count"],
            gate_density=gate_density,
            control_density=control_density,
            target_diversity=target_diversity,
            max_control_count=max_control_count,
            avg_control_count=avg_control_count,
            control_entropy=control_entropy,
            qubit_connectivity=qubit_connectivity,
            gate_interaction_graph_density=gate_interaction_density,
            non_triviality_score=non_triviality_score,
            cancellation_difficulty=cancellation_difficulty,
            consecutive_same_gate_count=consecutive_same_gate,
            gate_repetition_ratio=gate_repetition,
            control_target_overlap=control_target_overlap,
        )

    def _analyze_gate_composition(self, gates: List[Tuple]) -> Dict[str, int]:
        """Analyze the composition of gates in the circuit."""
        composition = {
            "not_count": 0,
            "cnot_count": 0,
            "ccnot_count": 0,
            "multi_control_count": 0,
        }

        for controls, _ in gates:
            control_count = len(controls)
            if control_count == 0:
                composition["not_count"] += 1
            elif control_count == 1:
                composition["cnot_count"] += 1
            elif control_count == 2:
                composition["ccnot_count"] += 1
            else:
                composition["multi_control_count"] += 1

        return composition

    def _calculate_depth(self, gates: List[Tuple], width: int) -> int:
        """Calculate the depth of the circuit (maximum number of gates in parallel)."""
        if not gates:
            return 0

        # Simple depth calculation: count gates that can't be executed in parallel
        # This is a simplified approach - a more accurate calculation would require
        # detailed dependency analysis

        # For now, we'll use a heuristic based on gate count and width
        # More gates and fewer qubits generally mean higher depth
        if width == 0:
            return len(gates)

        # Estimate depth based on gate count and circuit width
        # This is a rough approximation
        estimated_depth = max(1, len(gates) // max(1, width // 2))

        return estimated_depth

    def _compute_entropy(self, values: List[int]) -> float:
        """Compute Shannon entropy of a list of values."""
        if not values:
            return 0.0

        counter = Counter(values)
        total = len(values)
        entropy = 0.0

        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _compute_qubit_connectivity(self, gates: List[Tuple], width: int) -> float:
        """Compute the fraction of qubit pairs that interact."""
        if width < 2:
            return 0.0

        # Track which qubit pairs interact
        interacting_pairs = set()

        for controls, target in gates:
            # Add all control-target pairs
            for control in controls:
                pair = tuple(sorted([control, target]))
                interacting_pairs.add(pair)

            # Add control-control pairs for multi-control gates
            if len(controls) > 1:
                for i in range(len(controls)):
                    for j in range(i + 1, len(controls)):
                        pair = tuple(sorted([controls[i], controls[j]]))
                        interacting_pairs.add(pair)

        # Total possible pairs
        total_pairs = width * (width - 1) // 2

        return len(interacting_pairs) / total_pairs if total_pairs > 0 else 0.0

    def _compute_gate_interaction_density(
        self, gates: List[Tuple], width: int
    ) -> float:
        """Compute the density of the gate interaction graph."""
        if not gates:
            return 0.0

        # Create adjacency matrix for gate interactions
        adjacency = np.zeros((width, width))

        for controls, target in gates:
            # Add edges from controls to target
            for control in controls:
                adjacency[control, target] += 1
                adjacency[target, control] += 1  # Undirected graph

        # Compute density (actual edges / possible edges)
        total_possible_edges = width * (width - 1) // 2
        actual_edges = np.sum(adjacency > 0) // 2  # Divide by 2 for undirected graph

        return actual_edges / total_possible_edges if total_possible_edges > 0 else 0.0

    def _estimate_non_triviality(self, gates: List[Tuple], width: int) -> float:
        """Estimate non-triviality score based on circuit structure."""
        if not gates:
            return 0.0

        # Factors that make a circuit more non-trivial:
        # 1. More gates
        # 2. More complex gates (more controls)
        # 3. Higher connectivity
        # 4. More diverse gate types

        gate_complexity = sum(len(controls) for controls, _ in gates) / len(gates)
        connectivity = self._compute_qubit_connectivity(gates, width)

        # Simple heuristic score
        score = (len(gates) / width) * gate_complexity * (1 + connectivity)

        return min(score, 10.0)  # Cap at 10.0

    def _estimate_cancellation_difficulty(self, gates: List[Tuple]) -> float:
        """Estimate how difficult it is to cancel gates in this circuit."""
        if not gates:
            return 0.0

        # Factors that make cancellation harder:
        # 1. Fewer consecutive identical gates
        # 2. More complex gate patterns
        # 3. Gates that don't commute well

        # Count consecutive identical gates
        consecutive_count = 0
        for i in range(len(gates) - 1):
            if gates[i] == gates[i + 1]:
                consecutive_count += 1

        consecutive_ratio = (
            consecutive_count / (len(gates) - 1) if len(gates) > 1 else 0
        )

        # Higher difficulty for lower consecutive ratios
        difficulty = 1.0 - consecutive_ratio

        # Add complexity penalty
        avg_controls = sum(len(controls) for controls, _ in gates) / len(gates)
        difficulty += avg_controls * 0.1

        return min(difficulty, 1.0)

    def _count_consecutive_same_gates(self, gates: List[Tuple]) -> int:
        """Count the number of consecutive identical gates."""
        if len(gates) < 2:
            return 0

        count = 0
        for i in range(len(gates) - 1):
            if gates[i] == gates[i + 1]:
                count += 1

        return count

    def _compute_gate_repetition_ratio(self, gates: List[Tuple]) -> float:
        """Compute the ratio of unique gates to total gates."""
        if not gates:
            return 0.0

        # Convert controls to tuples to make gates hashable
        try:
            hashable_gates = set(
                (tuple(controls), target) for controls, target in gates
            )
            return len(hashable_gates) / len(gates)
        except Exception:
            # Fallback for malformed gates
            return 1.0

    def _compute_control_target_overlap(self, gates: List[Tuple]) -> float:
        """Compute the fraction of gates where the target is also a control qubit."""
        if not gates:
            return 0.0

        overlaps = 0
        total_gates = len(gates)

        for controls, target in gates:
            if target in controls:
                overlaps += 1

        return overlaps / total_gates


class ComplexityPredictor:
    """Predicts circuit complexity using extracted features."""

    def __init__(self):
        self.model = None  # Would be a trained ML model
        self.feature_weights = {
            "gate_count": 0.3,
            "max_control_count": 0.2,
            "qubit_connectivity": 0.15,
            "control_entropy": 0.1,
            "non_triviality_score": 0.25,
        }

    def predict_complexity(self, features: CircuitFeatures) -> float:
        """
        Predict circuit complexity score.
        Higher scores indicate more complex circuits.
        """
        if self.model is not None:
            # Use trained ML model
            feature_dict = features.to_dict()
            return self.model.predict([list(feature_dict.values())])[0]
        else:
            # Use simple weighted scoring
            return self._simple_complexity_score(features)

    def _simple_complexity_score(self, features: CircuitFeatures) -> float:
        """Simple complexity scoring using weighted features."""
        score = 0.0

        # Normalize features to 0-1 range
        normalized_gate_count = min(features.gate_count / 100.0, 1.0)
        normalized_max_controls = min(features.max_control_count / 5.0, 1.0)
        normalized_non_triviality = min(features.non_triviality_score / 10.0, 1.0)

        score += self.feature_weights["gate_count"] * normalized_gate_count
        score += self.feature_weights["max_control_count"] * normalized_max_controls
        score += (
            self.feature_weights["qubit_connectivity"] * features.qubit_connectivity
        )
        score += self.feature_weights["control_entropy"] * features.control_entropy
        score += (
            self.feature_weights["non_triviality_score"] * normalized_non_triviality
        )

        return min(score, 1.0)


class OptimizationAdvisor:
    """Provides optimization suggestions based on circuit features."""

    def __init__(self):
        self.optimization_rules = [
            self._check_high_gate_count,
            self._check_control_complexity,
            self._check_connectivity_issues,
            self._check_cancellation_opportunities,
            self._check_debris_opportunities,
        ]

    def get_optimization_suggestions(
        self, features: CircuitFeatures, complexity_score: float
    ) -> List[str]:
        """Get optimization suggestions for a circuit."""
        suggestions = []

        for rule in self.optimization_rules:
            suggestion = rule(features, complexity_score)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _check_high_gate_count(
        self, features: CircuitFeatures, complexity: float
    ) -> Optional[str]:
        """Check if circuit has too many gates."""
        if features.gate_count > 50:
            return "Consider circuit decomposition or alternative synthesis methods"
        elif features.gate_count > 20:
            return "Circuit may benefit from optimization passes"
        return None

    def _check_control_complexity(
        self, features: CircuitFeatures, complexity: float
    ) -> Optional[str]:
        """Check for high control complexity."""
        if features.max_control_count > 3:
            return "High control complexity detected - consider using ancilla qubits"
        elif features.avg_control_count > 1.5:
            return "Consider decomposing multi-control gates"
        return None

    def _check_connectivity_issues(
        self, features: CircuitFeatures, complexity: float
    ) -> Optional[str]:
        """Check for connectivity issues."""
        if features.qubit_connectivity < 0.3:
            return "Low qubit connectivity - may benefit from SWAP insertion"
        return None

    def _check_cancellation_opportunities(
        self, features: CircuitFeatures, complexity: float
    ) -> Optional[str]:
        """Check for cancellation opportunities."""
        if features.gate_repetition_ratio > 0.3:
            return "High gate repetition detected - good cancellation opportunities"
        elif features.consecutive_same_gate_count > 0:
            return "Consecutive identical gates found - apply cancellation"
        return None

    def _check_debris_opportunities(
        self, features: CircuitFeatures, complexity: float
    ) -> Optional[str]:
        """Check for debris cancellation opportunities."""
        if (
            features.cancellation_difficulty > 0.7
            and features.non_triviality_score > 5.0
        ):
            return "High cancellation difficulty - consider debris insertion analysis"
        return None


class MLFeatureManager:
    """Manages ML feature extraction and analysis for the factory."""

    def __init__(self, database):
        self.database = database
        self.extractor = MLFeatureExtractor()
        self.predictor = ComplexityPredictor()
        self.advisor = OptimizationAdvisor()

    def analyze_circuit(
        self,
        circuit_id: int,
        dim_group_id: int,
        circuit: Circuit,
        non_triviality_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a circuit and store ML features.

        Returns:
            Dictionary with features, complexity prediction, and optimization suggestions
        """
        # Extract features
        features = self.extractor.extract_features(circuit_id, circuit)

        # Predict complexity
        complexity_score = self.predictor.predict_complexity(features)

        # Get optimization suggestions
        suggestions = self.advisor.get_optimization_suggestions(
            features, complexity_score
        )

        # Store in database
        # from .database import MLFeatureRecord  # Removed - doesn't exist in simplified database
        # ml_record = MLFeatureRecord(
        #     id=None,
        #     circuit_id=circuit_id,
        #     dim_group_id=dim_group_id,
        #     features=features.to_dict(),
        #     complexity_prediction=complexity_score,
        #     optimization_suggestion="; ".join(suggestions) if suggestions else None
        # )
        #
        # self.database.store_ml_features(ml_record)

        return {
            "features": features.to_dict(),
            "complexity_prediction": complexity_score,
            "optimization_suggestions": suggestions,
            "feature_summary": {
                "gate_count": features.gate_count,
                "width": features.width,
                "max_controls": features.max_control_count,
                "connectivity": features.qubit_connectivity,
                "non_triviality": features.non_triviality_score,
            },
        }

    def get_circuit_analysis(self, circuit_id: int) -> Optional[Dict[str, Any]]:
        """Get stored ML analysis for a circuit."""
        # from .database import MLFeatureRecord  # Removed - doesn't exist in simplified database
        # ml_record = self.database.get_ml_features(circuit_id)
        # if ml_record:
        #     return {
        #         'features': ml_record.features,
        #         'complexity_prediction': ml_record.complexity_prediction,
        #         'optimization_suggestion': ml_record.optimization_suggestion
        #     }
        return None

    def get_dim_group_statistics(self, dim_group_id: int) -> Dict[str, Any]:
        """Get statistical analysis of a dimension group."""
        # This would require additional database queries
        # For now, return placeholder
        return {
            "avg_complexity": 0.0,
            "complexity_distribution": [],
            "common_optimizations": [],
            "feature_correlations": {},
        }
