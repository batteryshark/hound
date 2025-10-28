"""Token usage tracking for LLM providers."""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    profile: str | None = None  # e.g., "agent", "graph", "guidance"
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'provider': self.provider,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'profile': self.profile,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost
        }


class TokenTracker:
    """Tracks token usage across all LLM calls."""
    
    def __init__(self):
        self.usage_history: list[TokenUsage] = []
        self.usage_by_model: dict[str, dict[str, int | float]] = {}
        self._lock = Lock()
        self._output_file: Path | None = None
        # Track cumulative tokens per model for tiered pricing
        self._cumulative_tokens: dict[str, dict[str, int]] = {}
    
    def set_output_file(self, file_path: Path):
        """Set the output file for real-time updates."""
        self._output_file = file_path
        # Initialize the file with empty data
        self._save_to_file()
    
    def track_usage(self, 
                   provider: str,
                   model: str, 
                   input_tokens: int, 
                   output_tokens: int,
                   profile: str | None = None,
                   input_cost: float = 0.0,
                   output_cost: float = 0.0,
                   total_cost: float = 0.0):
        """Track token usage for a single LLM call."""
        with self._lock:
            usage = TokenUsage(
                timestamp=datetime.now().isoformat(),
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                profile=profile,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost
            )
            self.usage_history.append(usage)
            
            # Update model aggregates
            model_key = f"{provider}:{model}"
            if model_key not in self.usage_by_model:
                self.usage_by_model[model_key] = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'call_count': 0,
                    'input_cost': 0.0,
                    'output_cost': 0.0,
                    'total_cost': 0.0
                }
            
            self.usage_by_model[model_key]['input_tokens'] += input_tokens
            self.usage_by_model[model_key]['output_tokens'] += output_tokens
            self.usage_by_model[model_key]['total_tokens'] += input_tokens + output_tokens
            self.usage_by_model[model_key]['call_count'] += 1
            self.usage_by_model[model_key]['input_cost'] += input_cost
            self.usage_by_model[model_key]['output_cost'] += output_cost
            self.usage_by_model[model_key]['total_cost'] += total_cost
            
            # Update cumulative tokens for tiered pricing
            if model_key not in self._cumulative_tokens:
                self._cumulative_tokens[model_key] = {
                    'input_tokens': 0,
                    'output_tokens': 0
                }
            self._cumulative_tokens[model_key]['input_tokens'] += input_tokens
            self._cumulative_tokens[model_key]['output_tokens'] += output_tokens
            
            # Save to file if configured
            if self._output_file:
                self._save_to_file()
    
    def _save_to_file(self):
        """Save current state to file (called within lock)."""
        if not self._output_file:
            return
        
        data = self.get_summary()
        with open(self._output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self) -> dict:
        """Get summary of token usage."""
        with self._lock:
            return {
                'total_usage': {
                    'input_tokens': sum(u.input_tokens for u in self.usage_history),
                    'output_tokens': sum(u.output_tokens for u in self.usage_history),
                    'total_tokens': sum(u.total_tokens for u in self.usage_history),
                    'call_count': len(self.usage_history),
                    'input_cost': sum(u.input_cost for u in self.usage_history),
                    'output_cost': sum(u.output_cost for u in self.usage_history),
                    'total_cost': sum(u.total_cost for u in self.usage_history)
                },
                'by_model': dict(self.usage_by_model),
                'history': [u.to_dict() for u in self.usage_history]
            }

    def get_last_usage(self) -> dict | None:
        """Return the most recent usage entry as a dict, or None if empty."""
        with self._lock:
            if not self.usage_history:
                return None
            return self.usage_history[-1].to_dict()
    
    def get_cumulative_tokens(self, model_key: str) -> tuple[int, int]:
        """Get cumulative tokens for a model (for tiered pricing)."""
        with self._lock:
            if model_key not in self._cumulative_tokens:
                return (0, 0)
            return (
                self._cumulative_tokens[model_key]['input_tokens'],
                self._cumulative_tokens[model_key]['output_tokens']
            )
    
    def reset(self):
        """Reset all tracking data."""
        with self._lock:
            self.usage_history.clear()
            self.usage_by_model.clear()
            self._cumulative_tokens.clear()
            if self._output_file:
                self._save_to_file()


# Global token tracker instance
_token_tracker = TokenTracker()


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return _token_tracker
