"""
SimpleMem Logging Module

Provides structured logging with configurable levels for debugging and monitoring.
Supports both console and file output with token counting for LLM operations.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.debug("Processing window", tokens=1234, entries=5)

Log Levels (via LOG_LEVEL env var):
    - DEBUG: All details including token counts, similarity scores, per-entry logs
    - INFO: High-level operations (window processed, memories stored)
    - WARNING: Recoverable issues (retries, fallbacks)
    - ERROR: Failures that impact functionality
"""
import logging
import os
import sys
from typing import Any, Optional
from functools import lru_cache

# ANSI color codes for console output
COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m',       # Reset
    'BOLD': '\033[1m',        # Bold
    'DIM': '\033[2m',         # Dim
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output"""
    
    def __init__(self, fmt: str = None, use_colors: bool = True):
        super().__init__(fmt or '%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        # Add extra fields to message if present
        extras = []
        if hasattr(record, 'tokens'):
            extras.append(f"tokens={record.tokens}")
        if hasattr(record, 'input_tokens'):
            extras.append(f"in={record.input_tokens}")
        if hasattr(record, 'output_tokens'):
            extras.append(f"out={record.output_tokens}")
        if hasattr(record, 'entries'):
            extras.append(f"entries={record.entries}")
        if hasattr(record, 'action'):
            extras.append(f"action={record.action}")
        if hasattr(record, 'similarity'):
            extras.append(f"sim={record.similarity:.3f}")
        if hasattr(record, 'agent_id'):
            extras.append(f"agent={record.agent_id}")
        if hasattr(record, 'duration_ms'):
            extras.append(f"took={record.duration_ms}ms")
        
        if extras:
            record.msg = f"{record.msg} [{', '.join(extras)}]"
        
        if not self.use_colors:
            return super().format(record)
        
        # Add colors
        color = COLORS.get(record.levelname, COLORS['RESET'])
        reset = COLORS['RESET']
        dim = COLORS['DIM']
        
        # Format: timestamp [LEVEL] name: message
        formatted = super().format(record)
        
        # Color the level name
        formatted = formatted.replace(
            f'[{record.levelname}]',
            f'{color}[{record.levelname}]{reset}'
        )
        
        # Dim the timestamp
        if formatted.startswith('20'):  # Starts with year
            parts = formatted.split(' ', 2)
            if len(parts) >= 2:
                formatted = f"{dim}{parts[0]} {parts[1]}{reset} {parts[2]}"
        
        return formatted


class TokenCountingAdapter(logging.LoggerAdapter):
    """Logger adapter that handles extra fields for token counting"""
    
    def process(self, msg: str, kwargs: dict) -> tuple:
        # Move our custom fields from kwargs to extra
        extra = kwargs.get('extra', {})
        
        custom_fields = ['tokens', 'input_tokens', 'output_tokens', 'entries', 
                        'action', 'similarity', 'agent_id', 'user_id', 'duration_ms',
                        'window_size', 'conflict_action', 'memory_count']
        
        for field in custom_fields:
            if field in kwargs:
                extra[field] = kwargs.pop(field)
        
        kwargs['extra'] = extra
        return msg, kwargs


def _get_log_level() -> int:
    """Get log level from environment variable"""
    level_name = os.getenv('LOG_LEVEL', 'INFO').upper()
    return getattr(logging, level_name, logging.INFO)


def _get_log_file() -> Optional[str]:
    """Get optional log file path from environment"""
    return os.getenv('LOG_FILE')


def _should_use_colors() -> bool:
    """Determine if we should use colored output"""
    # Disable colors if LOG_COLORS=false or if not a TTY
    if os.getenv('LOG_COLORS', 'true').lower() in ('false', '0', 'no'):
        return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


@lru_cache(maxsize=32)
def get_logger(name: str) -> TokenCountingAdapter:
    """
    Get a logger instance with token counting support.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        TokenCountingAdapter: Logger with extra field support
        
    Example:
        logger = get_logger(__name__)
        logger.debug("LLM call complete", input_tokens=1000, output_tokens=500)
        logger.info("Window processed", entries=5, duration_ms=1500)
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(_get_log_level())
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_get_log_level())
        console_handler.setFormatter(ColoredFormatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            use_colors=_should_use_colors()
        ))
        logger.addHandler(console_handler)
        
        # Optional file handler
        log_file = _get_log_file()
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_handler.setFormatter(ColoredFormatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                use_colors=False
            ))
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return TokenCountingAdapter(logger, {})


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text.
    
    Uses a simple character-based estimation (4 chars ≈ 1 token for English).
    This is a rough estimate - actual tokenization depends on the model.
    
    For Bedrock models:
    - Nova: ~4 chars/token
    - Claude: ~3.5 chars/token
    - Titan Embeddings: ~4 chars/token (max 8192 tokens)
    
    Args:
        text: The text to estimate tokens for
        chars_per_token: Average characters per token (default 4.0)
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return int(len(text) / chars_per_token)


def format_token_usage(input_tokens: int, output_tokens: int, 
                       model_context_limit: int = 128000) -> str:
    """
    Format token usage for logging with context utilization.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_context_limit: Model's context window size
    
    Returns:
        Formatted string like "in=1500 out=800 total=2300 (1.8% of 128K)"
    """
    total = input_tokens + output_tokens
    pct = (total / model_context_limit) * 100
    limit_k = model_context_limit // 1000
    return f"in={input_tokens} out={output_tokens} total={total} ({pct:.1f}% of {limit_k}K)"


# Convenience: Create a default logger for simple imports
default_logger = get_logger('simplemem')
