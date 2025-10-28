"""LLM pricing calculator with support for tiered pricing."""
from __future__ import annotations

from typing import Any


class PricingCalculator:
    """Calculate costs for LLM usage based on configurable pricing."""
    
    # Map config provider names to runtime provider names
    PROVIDER_NAME_MAP = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "xai": "XAI",
        "deepseek": "DeepSeek",
        "mock": "Mock"
    }
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize pricing calculator with config.
        
        Args:
            config: Configuration dictionary containing models with pricing info
        """
        self.config = config
        self._pricing_cache: dict[str, dict] = {}
        self._load_pricing()
    
    def _load_pricing(self):
        """Load pricing information from config for all models."""
        models_cfg = self.config.get("models", {})
        
        for profile, model_cfg in models_cfg.items():
            if not isinstance(model_cfg, dict):
                continue
                
            provider = model_cfg.get("provider", "openai").lower()
            model_name = model_cfg.get("model", "")
            # Map config provider name to runtime provider name
            provider_normalized = self.PROVIDER_NAME_MAP.get(provider, provider.capitalize())
            model_key = f"{provider_normalized}:{model_name}"
            
            pricing = model_cfg.get("pricing", {})
            if not pricing:
                continue
            
            # Parse pricing configuration
            unit = pricing.get("unit", 1_000_000)  # Default to per 1M tokens
            
            # Check if tiered or simple pricing
            if "tiers" in pricing:
                # Tiered pricing
                tiers = []
                for tier in pricing["tiers"]:
                    tiers.append({
                        "threshold": tier.get("threshold", 0),
                        "input_cost": tier.get("input_cost", 0.0),
                        "output_cost": tier.get("output_cost", 0.0),
                    })
                # Sort by threshold ascending
                tiers.sort(key=lambda t: t["threshold"])
                self._pricing_cache[model_key] = {
                    "type": "tiered",
                    "unit": unit,
                    "tiers": tiers
                }
            else:
                # Simple pricing
                self._pricing_cache[model_key] = {
                    "type": "simple",
                    "unit": unit,
                    "input_cost": pricing.get("input_cost", 0.0),
                    "output_cost": pricing.get("output_cost", 0.0)
                }
    
    def calculate_cost(
        self,
        model_key: str,
        input_tokens: int,
        output_tokens: int,
        cumulative_input_tokens: int = 0,
        cumulative_output_tokens: int = 0
    ) -> tuple[float, float, float]:
        """
        Calculate cost for a given token usage.
        
        Args:
            model_key: Model identifier in format "provider:model"
            input_tokens: Number of input tokens for this call
            output_tokens: Number of output tokens for this call
            cumulative_input_tokens: Total input tokens used so far (for tier calculation)
            cumulative_output_tokens: Total output tokens used so far (for tier calculation)
        
        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        if model_key not in self._pricing_cache:
            return (0.0, 0.0, 0.0)
        
        pricing = self._pricing_cache[model_key]
        unit = pricing["unit"]
        
        if pricing["type"] == "simple":
            # Simple pricing - straightforward calculation
            input_cost = (input_tokens / unit) * pricing["input_cost"]
            output_cost = (output_tokens / unit) * pricing["output_cost"]
            return (input_cost, output_cost, input_cost + output_cost)
        
        elif pricing["type"] == "tiered":
            # Tiered pricing - calculate based on cumulative usage
            tiers = pricing["tiers"]
            
            # Calculate input cost with tiering
            input_cost = self._calculate_tiered_cost(
                tokens=input_tokens,
                cumulative_tokens=cumulative_input_tokens,
                tiers=tiers,
                unit=unit,
                cost_type="input_cost"
            )
            
            # Calculate output cost with tiering
            output_cost = self._calculate_tiered_cost(
                tokens=output_tokens,
                cumulative_tokens=cumulative_output_tokens,
                tiers=tiers,
                unit=unit,
                cost_type="output_cost"
            )
            
            return (input_cost, output_cost, input_cost + output_cost)
        
        return (0.0, 0.0, 0.0)
    
    def _calculate_tiered_cost(
        self,
        tokens: int,
        cumulative_tokens: int,
        tiers: list[dict],
        unit: int,
        cost_type: str
    ) -> float:
        """
        Calculate cost for tokens with tiered pricing.
        
        Args:
            tokens: Number of tokens for this call
            cumulative_tokens: Total tokens used before this call
            tiers: List of tier definitions
            unit: Token unit for pricing (e.g., 1_000_000)
            cost_type: "input_cost" or "output_cost"
        
        Returns:
            Cost for these tokens
        """
        if not tokens or not tiers:
            return 0.0
        
        total_cost = 0.0
        tokens_remaining = tokens
        current_cumulative = cumulative_tokens
        
        # Find which tier we're starting in
        current_tier_idx = 0
        for i, tier in enumerate(tiers):
            if current_cumulative >= tier["threshold"]:
                current_tier_idx = i
            else:
                break
        
        # Calculate cost across potentially multiple tiers
        while tokens_remaining > 0 and current_tier_idx < len(tiers):
            current_tier = tiers[current_tier_idx]
            rate = current_tier.get(cost_type, 0.0)
            
            # Determine how many tokens to charge at this tier's rate
            if current_tier_idx + 1 < len(tiers):
                # Not the last tier - check if we cross into next tier
                next_threshold = tiers[current_tier_idx + 1]["threshold"]
                tokens_until_next_tier = next_threshold - current_cumulative
                tokens_at_this_rate = min(tokens_remaining, tokens_until_next_tier)
            else:
                # Last tier - all remaining tokens use this rate
                tokens_at_this_rate = tokens_remaining
            
            # Calculate cost for these tokens
            cost = (tokens_at_this_rate / unit) * rate
            total_cost += cost
            
            # Move to next tier if needed
            tokens_remaining -= tokens_at_this_rate
            current_cumulative += tokens_at_this_rate
            current_tier_idx += 1
        
        return total_cost
    
    def has_pricing(self, model_key: str) -> bool:
        """Check if pricing is configured for a model."""
        return model_key in self._pricing_cache

