"""
LLM Client - Unified interface for LLM interactions

Supports multiple backends:
- OpenAI: OpenAI-compatible API (OpenAI, Qwen DashScope, Azure, etc.)
- Bedrock: Amazon Nova, Claude via AWS Bedrock

The backend is selected based on config.ENDPOINT
"""
import json
import time
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass
from utils.logger import get_logger, estimate_tokens, format_token_usage

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics for an LLM call"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool = True  # True if estimated, False if from API response
    
    def __str__(self):
        est = " (est)" if self.estimated else ""
        return f"in={self.input_tokens} out={self.output_tokens} total={self.total_tokens}{est}"


class LLMClient:
    """
    Unified LLM client interface supporting OpenAI and Bedrock
    """
    # Model context limits for token tracking
    MODEL_CONTEXT_LIMITS = {
        'amazon.nova-micro-v1:0': 128000,
        'amazon.nova-lite-v1:0': 300000,
        'amazon.nova-pro-v1:0': 300000,
        'anthropic.claude-3-5-sonnet-20241022-v2:0': 200000,
        'gpt-4.1-mini': 128000,
        'gpt-4.1': 128000,
        'gpt-4o': 128000,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None
    ):
        import config
        
        self.endpoint = getattr(config, 'ENDPOINT', 'openai').lower()
        
        if self.endpoint == 'bedrock':
            self._init_bedrock(config, use_streaming)
        else:
            self._init_openai(api_key, model, base_url, enable_thinking, use_streaming, config)
    
    def _init_bedrock(self, config, use_streaming):
        """Initialize AWS Bedrock client"""
        import boto3
        
        self.model_id = getattr(config, 'BEDROCK_LLM_MODEL', 'amazon.nova-micro-v1:0')
        self.region_name = getattr(config, 'AWS_REGION', 'us-east-1')
        self.max_tokens = getattr(config, 'MAX_TOKENS', 4096)
        self.use_streaming = use_streaming if use_streaming is not None else getattr(config, 'USE_STREAMING', True)
        
        # Token tracking
        self.last_usage: Optional[TokenUsage] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.context_limit = self.MODEL_CONTEXT_LIMITS.get(self.model_id, 128000)
        
        # Detect model type
        self.is_nova = self.model_id.startswith("amazon.nova")
        self.is_claude = self.model_id.startswith("anthropic.claude")
        
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )
        
        self.client_type = "bedrock"
        provider = "Nova" if self.is_nova else "Claude" if self.is_claude else "Unknown"
        logger.info(f"Initialized Bedrock LLM: {self.model_id} ({provider}) in {self.region_name}, context={self.context_limit//1000}K")
    
    def _init_openai(self, api_key, model, base_url, enable_thinking, use_streaming, config):
        """Initialize OpenAI-compatible client"""
        from openai import OpenAI
        
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING

        # Token tracking
        self.last_usage: Optional[TokenUsage] = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.context_limit = self.MODEL_CONTEXT_LIMITS.get(self.model, 128000)

        # Initialize OpenAI client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            logger.debug(f"Using custom OpenAI base URL: {self.base_url}")

        if self.enable_thinking:
            logger.info("Deep thinking mode enabled")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        self.client_type = "openai"
        logger.info(f"Initialized OpenAI LLM: {self.model}, context={self.context_limit//1000}K")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Chat completion with automatic backend selection and token tracking
        """
        start_time = time.time()
        
        # Estimate input tokens from messages
        input_text = " ".join(m.get("content", "") for m in messages)
        input_tokens_est = estimate_tokens(input_text)
        
        logger.debug(f"LLM call starting", input_tokens=input_tokens_est)
        
        if self.client_type == "bedrock":
            result = self._chat_completion_bedrock(messages, temperature, response_format, max_retries)
        else:
            result = self._chat_completion_openai(messages, temperature, response_format, max_retries)
        
        # Track output tokens
        output_tokens_est = estimate_tokens(result)
        self.last_usage = TokenUsage(
            input_tokens=input_tokens_est,
            output_tokens=output_tokens_est,
            total_tokens=input_tokens_est + output_tokens_est,
            estimated=True
        )
        
        # Update cumulative stats
        self.total_input_tokens += input_tokens_est
        self.total_output_tokens += output_tokens_est
        self.call_count += 1
        
        duration_ms = int((time.time() - start_time) * 1000)
        pct_context = (self.last_usage.total_tokens / self.context_limit) * 100
        
        logger.info(
            f"LLM call #{self.call_count} complete: {format_token_usage(input_tokens_est, output_tokens_est, self.context_limit)}",
            duration_ms=duration_ms
        )
        
        if pct_context > 50:
            logger.warning(f"High context usage: {pct_context:.1f}% of {self.context_limit//1000}K limit")
        
        return result
    
    def _chat_completion_bedrock(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        response_format: Optional[Dict[str, str]],
        max_retries: int
    ) -> str:
        """Chat completion using AWS Bedrock"""
        # Detect if JSON output is requested (like OpenAI's response_format)
        request_json = response_format and response_format.get("type") == "json_object"
        
        if self.is_nova:
            body = self._build_nova_request(messages, temperature, request_json=request_json)
        else:
            body = self._build_claude_request(messages, temperature, request_json=request_json)
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                if self.use_streaming:
                    return self._stream_bedrock_response(body)
                else:
                    response = self.bedrock_runtime.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(body),
                        contentType='application/json',
                        accept='application/json'
                    )
                    response_body = json.loads(response['body'].read())
                    return self._parse_bedrock_response(response_body)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Bedrock API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
        raise last_exception
    
    def _build_nova_request(self, messages: List[Dict[str, str]], temperature: float, request_json: bool = False) -> Dict:
        """
        Build request body for Amazon Nova models
        
        Matches legacy OpenAI behavior:
        - When response_format={"type": "json_object"} is passed, append JSON instruction
          to system prompt (Nova doesn't have native JSON mode like OpenAI)
        """
        system_prompt = ""
        nova_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                nova_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]
                })
        
        # Append JSON instruction to system prompt if JSON mode requested
        # This mimics OpenAI's response_format={"type": "json_object"} behavior
        if request_json:
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. No markdown, no explanation, just pure JSON."
            system_prompt = (system_prompt + json_instruction) if system_prompt else json_instruction.strip()
        
        body = {
            "messages": nova_messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            body["system"] = [{"text": system_prompt}]
        
        return body
    
    def _build_claude_request(self, messages: List[Dict[str, str]], temperature: float, request_json: bool = False) -> Dict:
        """
        Build request body for Anthropic Claude models
        
        Matches legacy OpenAI behavior for JSON mode.
        """
        system_prompt = ""
        bedrock_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                bedrock_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
        
        # Append JSON instruction if JSON mode requested
        if request_json:
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. No markdown, no explanation, just pure JSON."
            system_prompt = (system_prompt + json_instruction) if system_prompt else json_instruction.strip()
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": bedrock_messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        return body
    
    def _parse_bedrock_response(self, response_body: Dict) -> str:
        """Parse response based on model type"""
        if self.is_nova:
            return response_body['output']['message']['content'][0]['text']
        else:
            return response_body['content'][0]['text']
    
    def _stream_bedrock_response(self, body: Dict) -> str:
        """Handle streaming response from Bedrock"""
        response = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )
        
        full_content = []
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            if self.is_nova:
                if 'contentBlockDelta' in chunk:
                    text = chunk['contentBlockDelta'].get('delta', {}).get('text', '')
                    full_content.append(text)
            else:
                if chunk.get('type') == 'content_block_delta':
                    text = chunk['delta'].get('text', '')
                    full_content.append(text)
        
        return ''.join(full_content)

    def _chat_completion_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        response_format: Optional[Dict[str, str]],
        max_retries: int
    ) -> str:
        """Standard chat completion with OpenAI-compatible API"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Enable thinking mode if configured (for Qwen and compatible models only)
        # Only add enable_thinking parameter for Qwen API (identified by base_url)
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        
        if is_qwen_api:
            # Qwen API requires explicit enable_thinking parameter
            # - Streaming + thinking: enable_thinking=True
            # - Non-streaming: enable_thinking=False (required, not optional)
            # - JSON format: enable_thinking=False (incompatible with thinking mode)
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                # Explicitly set to False for non-streaming calls or JSON format
                kwargs["extra_body"] = {"enable_thinking": False}
        # For OpenAI and other APIs, don't add extra_body parameters

        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    return self._handle_streaming_response(**kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                    
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM API call failed after {max_retries} attempts: {e}")
        
        # If all retries failed, raise the last exception
        raise last_exception

    def _handle_streaming_response(self, **kwargs) -> str:
        """
        Handle streaming response and collect full content
        """
        full_content = []
        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_content.append(content)
        
        return ''.join(full_content)
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get cumulative token usage statistics.
        
        Returns:
            Dict with total_input_tokens, total_output_tokens, call_count, avg_per_call
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "call_count": self.call_count,
            "avg_input_per_call": self.total_input_tokens // max(1, self.call_count),
            "avg_output_per_call": self.total_output_tokens // max(1, self.call_count),
            "context_limit": self.context_limit,
            "last_usage": str(self.last_usage) if self.last_usage else None
        }

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
