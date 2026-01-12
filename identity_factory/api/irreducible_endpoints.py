"""
API endpoints for irreducible circuit database.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..inverse_finder import InverseFinder
from ..irreducible_db import (
    ForwardCircuit,
    IdentityCircuit,
    InverseCircuit,
    IrreducibleDatabase,
)
from ..irreducible_generator import IrreducibleGenerator

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances
_db: Optional[IrreducibleDatabase] = None
_generator: Optional[IrreducibleGenerator] = None
_finder: Optional[InverseFinder] = None


def get_database() -> IrreducibleDatabase:
    global _db
    if _db is None:
        _db = IrreducibleDatabase()
    return _db


def get_generator() -> IrreducibleGenerator:
    global _generator
    if _generator is None:
        _generator = IrreducibleGenerator(get_database())
    return _generator


def get_finder() -> InverseFinder:
    global _finder
    if _finder is None:
        _finder = InverseFinder(get_database(), method="reverse")
    return _finder


# Pydantic models
class GenerateRequest(BaseModel):
    width: int = Field(..., ge=2, le=10, description="Number of qubits")
    repetitions: int = Field(..., ge=1, le=10, description="Pattern repetitions")
    count: int = Field(1, ge=1, le=1000, description="Number of circuits to generate")


class GenerateResponse(BaseModel):
    success: bool
    generated: int
    total_gates: int
    sample_circuit_id: Optional[int] = None


class FindInverseRequest(BaseModel):
    method: str = Field(
        "reverse", description="Method: 'reverse', 'sat_optimal', 'sat_bounded'"
    )


class FindInverseResponse(BaseModel):
    success: bool
    inverse_id: Optional[int] = None
    gate_count: int
    method: str
    error: Optional[str] = None


class StatsResponse(BaseModel):
    forward_circuits: int
    inverse_circuits: int
    identity_circuits: int
    by_width: dict


@router.post("/generate", response_model=GenerateResponse)
async def generate_circuits(
    request: GenerateRequest, generator: IrreducibleGenerator = Depends(get_generator)
) -> GenerateResponse:
    """
    Generate batch of forward circuits.

    Creates circuits using width × repetitions pattern where each
    repetition touches all wires.
    """
    try:
        circuits = generator.generate_batch(
            width=request.width,
            repetitions=request.repetitions,
            count=request.count,
            store=True,
        )

        return GenerateResponse(
            success=True,
            generated=len(circuits),
            total_gates=circuits[0].gate_count if circuits else 0,
            sample_circuit_id=circuits[0].id if circuits else None,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/find-inverse/{forward_id}", response_model=FindInverseResponse)
async def find_inverse(
    forward_id: int,
    request: FindInverseRequest = FindInverseRequest(),
    database: IrreducibleDatabase = Depends(get_database),
) -> FindInverseResponse:
    """
    Find inverse for a forward circuit.
    """
    try:
        # Get forward circuit
        forward = database.get_forward(forward_id)
        if not forward:
            raise HTTPException(status_code=404, detail="Forward circuit not found")

        # Create finder with requested method
        finder = InverseFinder(database, method=request.method)

        # Find and store inverse
        inverse = finder.find_and_store(forward)

        return FindInverseResponse(
            success=True,
            inverse_id=inverse.id,
            gate_count=inverse.gate_count,
            method=inverse.synthesis_method,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inverse finding failed: {e}")
        return FindInverseResponse(
            success=False, gate_count=0, method=request.method, error=str(e)
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    database: IrreducibleDatabase = Depends(get_database),
) -> StatsResponse:
    """Get database statistics."""
    stats = database.get_stats()
    return StatsResponse(**stats)


@router.get("/forward/{width}")
async def list_forward_circuits(
    width: int,
    limit: int = Query(100, ge=1, le=1000),
    database: IrreducibleDatabase = Depends(get_database),
):
    """List forward circuits by width."""
    circuits = database.list_forward_by_width(width, limit)

    return {
        "width": width,
        "count": len(circuits),
        "circuits": [
            {
                "id": c.id,
                "gate_count": c.gate_count,
                "permutation_hash": c.permutation_hash,
                "created_at": c.created_at,
            }
            for c in circuits
        ],
    }


@router.post("/batch-pipeline")
async def run_batch_pipeline(
    dimensions: List[List[int]] = [[3, 5], [4, 4], [5, 3]],
    circuits_per_dimension: int = Query(10, ge=1, le=100),
    generator: IrreducibleGenerator = Depends(get_generator),
    finder: InverseFinder = Depends(get_finder),
    database: IrreducibleDatabase = Depends(get_database),
):
    """
    Run complete pipeline: generate → find inverses → create identities.
    """
    try:
        results = {
            "dimensions_processed": 0,
            "forward_generated": 0,
            "inverses_found": 0,
            "identities_created": 0,
        }

        for width, reps in dimensions:
            # Generate forward circuits
            forwards = generator.generate_batch(
                width=width, repetitions=reps, count=circuits_per_dimension, store=True
            )
            results["forward_generated"] += len(forwards)

            # Find inverses
            inverses = finder.batch_find_inverses(forwards, store=True)
            results["inverses_found"] += len(inverses)

            # Create identities
            for fwd, inv in zip(forwards, inverses):
                identity = IdentityCircuit(
                    id=None,
                    forward_id=fwd.id,
                    inverse_id=inv.id,
                    width=fwd.width,
                    total_gates=fwd.gate_count + inv.gate_count,
                )
                identity.id = database.store_identity(identity)
                results["identities_created"] += 1

            results["dimensions_processed"] += 1

        return {"success": True, "results": results}

    except Exception as e:
        logger.error(f"Batch pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
