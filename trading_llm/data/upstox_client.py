"""Upstox API client for fetching Indian market data."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp

from ..core import config


class UpstoxAPIError(Exception):
    """Custom exception for Upstox API errors."""
    
    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []
        self.status_code = status_code
    
    def __str__(self) -> str:
        if self.errors:
            error_details = ", ".join(str(e) for e in self.errors)
            return f"{self.message}: {error_details}"
        return self.message


class UpstoxClient:
    """Client for interacting with Upstox API."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 auth_token: Optional[str] = None, base_url: Optional[str] = None,
                 session: Optional[aiohttp.ClientSession] = None):
        """Initialize Upstox client with credentials.
        
        Args:
            api_key: Upstox API key (defaults to config.UPSTOX_API_KEY)
            api_secret: Upstox API secret (defaults to config.UPSTOX_API_SECRET)
            auth_token: Access token for API requests (defaults to config.UPSTOX_AUTH_TOKEN)
            base_url: Base URL for API (defaults to config.UPSTOX_BASE_URL)
            session: Optional aiohttp ClientSession (creates new one if not provided)
        """
        self.api_key = api_key or config.UPSTOX_API_KEY
        self.api_secret = api_secret or config.UPSTOX_API_SECRET
        self.auth_token = auth_token or config.UPSTOX_AUTH_TOKEN
        self.base_url = (base_url or config.UPSTOX_BASE_URL).rstrip("/")
        self._session = session
        self._own_session = session is None
        
        if not self.auth_token:
            raise ValueError("UPSTOX_AUTH_TOKEN is required for API requests")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._own_session = True
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session if we own it."""
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Upstox API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., '/user/profile')
            params: Query parameters for GET requests
            data: Form data for POST/PUT requests (application/x-www-form-urlencoded)
            json_data: JSON data for POST/PUT requests (application/json)
        
        Returns:
            Parsed response data from the API
        
        Raises:
            UpstoxAPIError: If the API returns an error response
            aiohttp.ClientError: If the HTTP request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        
        session = await self._get_session()
        retries = config.RETRIES
        backoff = config.BACKOFF
        last_exc = None
        
        for i in range(retries + 1):
            try:
                async with session.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    params=params,
                    data=data,
                    json=json_data,
                ) as response:
                    # Read response data first
                    try:
                        response_data = await response.json()
                    except Exception:
                        # If JSON parsing fails, try to get text
                        try:
                            text = await response.text()
                            response_data = json.loads(text) if text else {}
                        except Exception:
                            response_data = {}
                    
                    # Check for HTTP errors
                    if response.status >= 400:
                        self._handle_error_response(response_data, response.status)
                    
                    return self._handle_response_data(response_data, response.status)
            except UpstoxAPIError:
                # Re-raise UpstoxAPIError immediately (already handled)
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = UpstoxAPIError(f"Request failed: {str(e)}")
                if i == retries:
                    raise last_exc
            except Exception as e:
                last_exc = UpstoxAPIError(f"Unexpected error: {str(e)}")
                if i == retries:
                    raise last_exc
            
            # Exponential backoff
            if i < retries:
                await asyncio.sleep(backoff ** i)
        
        if last_exc:
            raise last_exc
        raise UpstoxAPIError("Request failed after retries")
    
    def _handle_response_data(self, response_data: Dict[str, Any], status_code: int) -> Dict[str, Any]:
        """Parse and validate Upstox API response.
        
        Args:
            response_data: Parsed JSON response data
            status_code: HTTP status code
        
        Returns:
            Parsed response data
        
        Raises:
            UpstoxAPIError: If response indicates an error
        """
        status = response_data.get("status", "").lower()
        
        if status == "success":
            # Return the data field if present, otherwise return the whole response
            return response_data.get("data", response_data)
        elif status == "error":
            errors = response_data.get("errors", [])
            error_message = "API returned error status"
            if errors and isinstance(errors, list) and len(errors) > 0:
                first_error = errors[0]
                if isinstance(first_error, dict):
                    error_message = first_error.get("message", error_message)
                else:
                    error_message = str(first_error)
            raise UpstoxAPIError(
                error_message,
                errors=errors if isinstance(errors, list) else [errors],
                status_code=status_code
            )
        else:
            # Unknown status, return data as-is but log warning
            return response_data.get("data", response_data)
    
    def _handle_error_response(self, error_data: Dict[str, Any], status_code: Optional[int]) -> None:
        """Handle error response from API.
        
        Args:
            error_data: Parsed error response data
            status_code: HTTP status code
        
        Raises:
            UpstoxAPIError: Always raises an exception
        """
        errors = error_data.get("errors", [])
        error_message = error_data.get("message", f"API error (status {status_code})")
        
        raise UpstoxAPIError(
            error_message,
            errors=errors if isinstance(errors, list) else [errors],
            status_code=status_code
        )
    
    # Placeholder methods for common endpoints
    
    async def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile information.
        
        Returns:
            User profile data
        """
        return await self._make_request("GET", "/user/profile")
    
    async def get_market_quote(self, instrument_key: str) -> Dict[str, Any]:
        """Get market quote for an instrument.
        
        Args:
            instrument_key: Instrument key (e.g., 'NSE_EQ|INE467B01029')
        
        Returns:
            Market quote data
        """
        params = {"instrument_key": instrument_key}
        return await self._make_request("GET", "/market-quote/quotes", params=params)
    
    async def get_market_quotes(self, instrument_keys: List[str]) -> Dict[str, Any]:
        """Get market quotes for multiple instruments.
        
        Args:
            instrument_keys: List of instrument keys
        
        Returns:
            Market quotes data
        """
        # Upstox API typically expects comma-separated values
        params = {"instrument_key": ",".join(instrument_keys)}
        return await self._make_request("GET", "/market-quote/quotes", params=params)

