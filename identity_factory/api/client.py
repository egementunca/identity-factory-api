"""
Python client for Identity Circuit Factory API.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel

from .models import *

logger = logging.getLogger(__name__)


class IdentityFactoryClient:
    """Client for Identity Circuit Factory API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_version: str = "v1",
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the API server
            api_version: API version
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.api_url = f"{self.base_url}/api/{api_version}"

        # Create HTTP client
        self.client = httpx.AsyncClient(
            timeout=timeout,
            verify=verify_ssl,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        try:
            response = await self.client.request(
                method=method, url=url, json=data, params=params
            )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    # Health and status methods
    async def health_check(self) -> HealthResponse:
        """Check API health."""
        data = await self._make_request("GET", "/health")
        return HealthResponse(**data)

    async def get_stats(self) -> FactoryStatsResponse:
        """Get factory statistics."""
        data = await self._make_request("GET", "/stats")
        return FactoryStatsResponse(**data)

    async def get_detailed_stats(self) -> DetailedStatsResponse:
        """Get detailed statistics."""
        data = await self._make_request("GET", "/stats/detailed")
        return DetailedStatsResponse(**data)

    # Circuit generation methods
    async def generate_circuit(
        self, request: CircuitRequest
    ) -> GenerationResultResponse:
        """Generate a single identity circuit."""
        data = await self._make_request("POST", "/generate", data=request.dict())
        return GenerationResultResponse(**data)

    async def generate_circuits_batch(
        self, request: BatchCircuitRequest
    ) -> Dict[str, GenerationResultResponse]:
        """Generate multiple identity circuits."""
        data = await self._make_request("POST", "/generate/batch", data=request.dict())

        # Convert response data
        results = {}
        for key, result_data in data.items():
            results[key] = GenerationResultResponse(**result_data)

        return results

    # Unrolling methods
    async def unroll_dimension_group(
        self, request: UnrollRequest
    ) -> UnrollResultResponse:
        """Unroll a dimension group."""
        data = await self._make_request("POST", "/unroll", data=request.dict())
        return UnrollResultResponse(**data)

    async def unroll_all_dimension_groups(
        self, request: UnrollRequest
    ) -> Dict[int, UnrollResultResponse]:
        """Unroll all dimension groups."""
        data = await self._make_request("POST", "/unroll/all", data=request.dict())

        # Convert response data
        results = {}
        for key, result_data in data.items():
            results[int(key)] = UnrollResultResponse(**result_data)

        return results

    # Simplification methods
    async def simplify_dimension_group(
        self, request: SimplificationRequest
    ) -> Dict[int, SimplificationResultResponse]:
        """Simplify a dimension group."""
        data = await self._make_request("POST", "/simplify", data=request.dict())

        # Convert response data
        results = {}
        for key, result_data in data.items():
            results[int(key)] = SimplificationResultResponse(**result_data)

        return results

    # Database query methods
    async def get_circuits(
        self,
        page: int = 1,
        size: int = 50,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
        width: Optional[int] = None,
        length: Optional[int] = None,
        min_equivalents: Optional[int] = None,
        max_equivalents: Optional[int] = None,
        gate_type: Optional[str] = None,
    ) -> PaginatedResponse:
        """Get circuits with pagination and filtering."""
        params = {"page": page, "size": size, "sort_order": sort_order}

        if sort_by:
            params["sort_by"] = sort_by
        if width:
            params["width"] = width
        if length:
            params["length"] = length
        if min_equivalents:
            params["min_equivalents"] = min_equivalents
        if max_equivalents:
            params["max_equivalents"] = max_equivalents
        if gate_type:
            params["gate_type"] = gate_type

        data = await self._make_request("GET", "/circuits", params=params)
        return PaginatedResponse(**data)

    async def get_circuit(self, circuit_id: int) -> CircuitResponse:
        """Get a specific circuit."""
        data = await self._make_request("GET", f"/circuits/{circuit_id}")
        return CircuitResponse(**data)

    async def get_dimension_groups(self) -> List[DimGroupResponse]:
        """Get all dimension groups."""
        data = await self._make_request("GET", "/dim-groups")
        return [DimGroupResponse(**item) for item in data]

    async def get_dimension_group(self, dim_group_id: int) -> DimGroupResponse:
        """Get a specific dimension group."""
        data = await self._make_request("GET", f"/dim-groups/{dim_group_id}")
        return DimGroupResponse(**data)

    async def get_circuits_in_dimension_group(
        self, dim_group_id: int
    ) -> List[CircuitResponse]:
        """Get all circuits in a dimension group."""
        data = await self._make_request("GET", f"/dim-groups/{dim_group_id}/circuits")
        return [CircuitResponse(**item) for item in data]

    # Export/Import methods
    async def export_dimension_group(self, request: ExportRequest) -> bytes:
        """Export a dimension group."""
        url = f"{self.api_url}/export"

        try:
            response = await self.client.post(url, json=request.dict())
            response.raise_for_status()
            return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"Export failed: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Export error: {e}")
            raise

    async def import_dimension_group(self, request: ImportRequest) -> Dict[str, Any]:
        """Import a dimension group."""
        data = await self._make_request("POST", "/import", data=request.dict())
        return data

    # Utility methods
    async def get_recommendations(
        self, target_width: int, max_length: int = 20, limit: int = 10
    ) -> List[Tuple[int, int]]:
        """Get dimension recommendations."""
        params = {
            "target_width": target_width,
            "max_length": max_length,
            "limit": limit,
        }

        data = await self._make_request("GET", "/recommendations", params=params)
        return [tuple(item) for item in data]

    async def delete_dimension_group(self, dim_group_id: int) -> Dict[str, Any]:
        """Delete a dimension group."""
        return await self._make_request("DELETE", f"/dim-groups/{dim_group_id}")

    async def delete_circuit(self, circuit_id: int) -> Dict[str, Any]:
        """Delete a circuit."""
        return await self._make_request("DELETE", f"/circuits/{circuit_id}")


# Synchronous wrapper for convenience
class IdentityFactoryClientSync:
    """Synchronous wrapper for the API client."""

    def __init__(self, *args, **kwargs):
        """Initialize synchronous client."""
        self.client = IdentityFactoryClient(*args, **kwargs)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.client.close())

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        return asyncio.run(coro)

    # Health and status methods
    def health_check(self) -> HealthResponse:
        """Check API health."""
        return self._run_async(self.client.health_check())

    def get_stats(self) -> FactoryStatsResponse:
        """Get factory statistics."""
        return self._run_async(self.client.get_stats())

    def get_detailed_stats(self) -> DetailedStatsResponse:
        """Get detailed statistics."""
        return self._run_async(self.client.get_detailed_stats())

    # Circuit generation methods
    def generate_circuit(self, request: CircuitRequest) -> GenerationResultResponse:
        """Generate a single identity circuit."""
        return self._run_async(self.client.generate_circuit(request))

    def generate_circuits_batch(
        self, request: BatchCircuitRequest
    ) -> Dict[str, GenerationResultResponse]:
        """Generate multiple identity circuits."""
        return self._run_async(self.client.generate_circuits_batch(request))

    # Unrolling methods
    def unroll_dimension_group(self, request: UnrollRequest) -> UnrollResultResponse:
        """Unroll a dimension group."""
        return self._run_async(self.client.unroll_dimension_group(request))

    def unroll_all_dimension_groups(
        self, request: UnrollRequest
    ) -> Dict[int, UnrollResultResponse]:
        """Unroll all dimension groups."""
        return self._run_async(self.client.unroll_all_dimension_groups(request))

    # Simplification methods
    def simplify_dimension_group(
        self, request: SimplificationRequest
    ) -> Dict[int, SimplificationResultResponse]:
        """Simplify a dimension group."""
        return self._run_async(self.client.simplify_dimension_group(request))

    # Database query methods
    def get_circuits(self, **kwargs) -> PaginatedResponse:
        """Get circuits with pagination and filtering."""
        return self._run_async(self.client.get_circuits(**kwargs))

    def get_circuit(self, circuit_id: int) -> CircuitResponse:
        """Get a specific circuit."""
        return self._run_async(self.client.get_circuit(circuit_id))

    def get_dimension_groups(self) -> List[DimGroupResponse]:
        """Get all dimension groups."""
        return self._run_async(self.client.get_dimension_groups())

    def get_dimension_group(self, dim_group_id: int) -> DimGroupResponse:
        """Get a specific dimension group."""
        return self._run_async(self.client.get_dimension_group(dim_group_id))

    def get_circuits_in_dimension_group(
        self, dim_group_id: int
    ) -> List[CircuitResponse]:
        """Get all circuits in a dimension group."""
        return self._run_async(
            self.client.get_circuits_in_dimension_group(dim_group_id)
        )

    # Export/Import methods
    def export_dimension_group(self, request: ExportRequest) -> bytes:
        """Export a dimension group."""
        return self._run_async(self.client.export_dimension_group(request))

    def import_dimension_group(self, request: ImportRequest) -> Dict[str, Any]:
        """Import a dimension group."""
        return self._run_async(self.client.import_dimension_group(request))

    # Utility methods
    def get_recommendations(
        self, target_width: int, max_length: int = 20, limit: int = 10
    ) -> List[Tuple[int, int]]:
        """Get dimension recommendations."""
        return self._run_async(
            self.client.get_recommendations(target_width, max_length, limit)
        )

    def delete_dimension_group(self, dim_group_id: int) -> Dict[str, Any]:
        """Delete a dimension group."""
        return self._run_async(self.client.delete_dimension_group(dim_group_id))

    def delete_circuit(self, circuit_id: int) -> Dict[str, Any]:
        """Delete a circuit."""
        return self._run_async(self.client.delete_circuit(circuit_id))
