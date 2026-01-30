"""
Memory Builder - Stage 1: Semantic Structured Compression (Section 3.1)

Implements Semantic Structured Compression:
- Entropy-based non-linear filter: Φ_gate (conceptual - filters low-density dialogue)
- De-linearization transformation: F_θ (converts dialogue to atomic entries)
- Generates self-contained Atomic Entries {m_k} via coreference resolution and temporal anchoring
- Conflict Resolution: Mem0-style ADD/UPDATE/DELETE for memory consistency
"""
import time
from typing import List, Optional
from models.memory_entry import MemoryEntry, Dialogue
from utils.llm_client import LLMClient
from utils.logger import get_logger, estimate_tokens
from database.vector_store import VectorStore
import config
import json
import asyncio
import concurrent.futures
from functools import partial

logger = get_logger(__name__)


def filter_for_storage(entries: List[MemoryEntry]) -> List[MemoryEntry]:
    """
    Filter entries before storing to database.
    
    Only stores Insights (pattern) and Feedback (correction).
    Tool call results (factual) are skipped - they're volatile and not useful for agent context.
    
    This keeps the database focused on stable, valuable memories.
    Easy to re-enable factual storage by modifying this filter.
    """
    stored_types = {'pattern', 'correction'}  # Insights + Feedback only
    
    filtered = [e for e in entries if e.memory_type in stored_types]
    skipped = len(entries) - len(filtered)
    
    if skipped > 0:
        logger.debug(f"Filtered out {skipped} factual entries (volatile tool call data)")
    
    return filtered


class MemoryBuilder:
    """
    Memory Builder - Stage 1: Semantic Structured Compression

    Paper Reference: Section 3.1 - Semantic Structured Compression

    Core Functions:
    1. Entropy-based filtering (implicit via window processing)
    2. De-linearization transformation F_θ: Dialogue → Atomic Entries
    3. Coreference resolution Φ_coref (no pronouns)
    4. Temporal anchoring Φ_time (absolute timestamps)
    5. Generate self-contained Atomic Entries {m_k}
    6. Conflict Resolution: ADD/UPDATE/DELETE based on semantic similarity (Mem0-style)
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: int = None,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 3,
        enable_conflict_resolution: bool = None
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or config.WINDOW_SIZE
        
        # Use config values as default if not explicitly provided
        self.enable_parallel_processing = enable_parallel_processing if enable_parallel_processing is not None else getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
        self.max_parallel_workers = max_parallel_workers if max_parallel_workers is not None else getattr(config, 'MAX_PARALLEL_WORKERS', 4)
        
        # Conflict resolution (Mem0-style ADD/UPDATE/DELETE)
        self.enable_conflict_resolution = enable_conflict_resolution if enable_conflict_resolution is not None else getattr(config, 'ENABLE_CONFLICT_RESOLUTION', True)
        self.conflict_resolver = None
        if self.enable_conflict_resolution:
            from core.conflict_resolver import ConflictResolver
            self.conflict_resolver = ConflictResolver(llm_client, vector_store)

        # Dialogue buffer
        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0

        # Previous window entries (for context)
        self.previous_entries: List[MemoryEntry] = []

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True):
        """
        Add a dialogue to the buffer
        """
        self.dialogue_buffer.append(dialogue)

        # Auto process
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True):
        """
        Batch add dialogues with optional parallel processing
        """
        if self.enable_parallel_processing and len(dialogues) > self.window_size * 2:
            # Use parallel processing for large batches
            self.add_dialogues_parallel(dialogues)
        else:
            # Use sequential processing for smaller batches
            for dialogue in dialogues:
                self.add_dialogue(dialogue, auto_process=False)

            # Process complete windows
            if auto_process:
                while len(self.dialogue_buffer) >= self.window_size:
                    self.process_window()
    
    def add_dialogues_parallel(self, dialogues: List[Dialogue]):
        """
        Add dialogues using parallel processing for better performance
        """
        try:
            # Add all dialogues to buffer first
            self.dialogue_buffer.extend(dialogues)
            
            # Group into windows for parallel processing (including remaining dialogues)
            windows_to_process = []
            while len(self.dialogue_buffer) >= self.window_size:
                window = self.dialogue_buffer[:self.window_size]
                self.dialogue_buffer = self.dialogue_buffer[self.window_size:]
                windows_to_process.append(window)
            
            # Add remaining dialogues as a smaller batch (no need to process separately)
            if self.dialogue_buffer:
                windows_to_process.append(self.dialogue_buffer)
                self.dialogue_buffer = []  # Clear buffer since we're processing all
            
            if windows_to_process:
                batch_sizes = [len(w) for w in windows_to_process]
                total_dialogues = sum(batch_sizes)
                logger.info(f"Parallel processing: {len(windows_to_process)} batches, {total_dialogues} dialogues, {self.max_parallel_workers} workers")
                logger.debug(f"Batch sizes: {batch_sizes}")
                
                # Process all windows/batches in parallel (including remaining dialogues)
                self._process_windows_parallel(windows_to_process)
                
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential...")
            # Fallback to sequential processing
            for window in windows_to_process:
                self.dialogue_buffer = window + self.dialogue_buffer
                self.process_window()

    def process_window(self, agent_id: str = None, user_id: str = None):
        """
        Process current window dialogues - Core logic
        
        Args:
            agent_id: Optional agent ID for scoped conflict resolution
            user_id: Optional user ID for scoped conflict resolution
        """
        if not self.dialogue_buffer:
            return

        start_time = time.time()
        
        # Extract window
        window = self.dialogue_buffer[:self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.window_size:]

        # Estimate input tokens for the window
        window_text = "\n".join([str(d) for d in window])
        input_tokens = estimate_tokens(window_text)
        
        logger.info(f"Processing window: {len(window)} dialogues (total processed: {self.processed_count})", 
                   input_tokens=input_tokens, agent_id=agent_id)

        # Call LLM to generate memory entries
        entries = self._generate_memory_entries(window)

        # Store to database with conflict resolution
        if entries:
            logger.info(f"Extracted {len(entries)} memory entries from window")
            for i, entry in enumerate(entries, 1):
                logger.debug(f"  [{i}] {entry.lossless_restatement[:80]}{'...' if len(entry.lossless_restatement) > 80 else ''}")
            
            # Filter out factual entries (volatile tool call data)
            entries_to_store = filter_for_storage(entries)
            
            # Use conflict resolution if enabled
            if entries_to_store:
                if self.conflict_resolver and self.enable_conflict_resolution:
                    logger.debug("Running conflict resolution...")
                    resolutions = self.conflict_resolver.resolve_conflicts(
                        entries_to_store, agent_id=agent_id, user_id=user_id
                    )
                    counts = self.conflict_resolver.apply_resolutions(resolutions)
                else:
                    # Direct add without conflict resolution
                    self.vector_store.add_entries(entries_to_store)
            
            self.previous_entries = entries  # Save as context (include all for LLM reference)
            self.processed_count += len(window)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Window complete: {len(entries)} entries generated", 
                   entries=len(entries), duration_ms=duration_ms)

    def process_remaining(self, agent_id: str = None, user_id: str = None, user_name: str = None):
        """
        Process remaining dialogues (fallback method, normally handled in parallel)
        
        Args:
            agent_id: Optional agent ID for scoped conflict resolution
            user_id: Optional user ID for scoped conflict resolution
            user_name: Optional human-readable user name
        """
        if self.dialogue_buffer:
            start_time = time.time()
            remaining_count = len(self.dialogue_buffer)
            
            logger.info(f"Processing remaining {remaining_count} dialogues (fallback mode)", 
                       agent_id=agent_id)
            
            entries = self._generate_memory_entries(self.dialogue_buffer)
            if entries:
                # Set agent_id, user_id, and user_name on all entries
                for entry in entries:
                    entry.agent_id = agent_id
                    entry.user_id = user_id
                    entry.user_name = user_name
                
                logger.info(f"Extracted {len(entries)} memory entries from remaining dialogues")
                for i, entry in enumerate(entries, 1):
                    logger.debug(f"  [{i}] {entry.lossless_restatement[:80]}{'...' if len(entry.lossless_restatement) > 80 else ''}")
                
                # Filter out factual entries (volatile tool call data)
                entries_to_store = filter_for_storage(entries)
                
                # Use conflict resolution if enabled
                if entries_to_store:
                    if self.conflict_resolver and self.enable_conflict_resolution:
                        logger.debug("Running conflict resolution...")
                        resolutions = self.conflict_resolver.resolve_conflicts(
                            entries_to_store, agent_id=agent_id, user_id=user_id
                        )
                        self.conflict_resolver.apply_resolutions(resolutions)
                    else:
                        self.vector_store.add_entries(entries_to_store)
                
                self.processed_count += len(self.dialogue_buffer)
            
            self.dialogue_buffer = []
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Remaining dialogues complete: {len(entries)} entries", 
                       entries=len(entries), duration_ms=duration_ms)

    def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        De-linearization Transformation F_θ: W_t → {m_k}

        Paper Reference: Section 3.1 - Eq. (3)
        Applies composite transformation: F_θ = Φ_time ∘ Φ_coref ∘ Φ_extract

        Key requirements:
        1. Generate multiple Atomic Entries to cover all information
        2. Φ_coref: Force coreference resolution (no pronouns)
        3. Φ_time: Temporal anchoring (convert relative to absolute time)
        4. Reference previous window entries to avoid duplication
        """
        start_time = time.time()
        
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in dialogues])
        dialogue_ids = [d.dialogue_id for d in dialogues]
        
        # Estimate tokens for the dialogue
        dialogue_tokens = estimate_tokens(dialogue_text)
        logger.debug(f"Dialogue text: {len(dialogues)} turns, ~{dialogue_tokens} tokens")

        # Build context
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:  # Only show first 3
                context += f"- {entry.lossless_restatement}\n"
            logger.debug(f"Using {len(self.previous_entries[:3])} previous entries for context")

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)
        prompt_tokens = estimate_tokens(prompt)
        logger.debug(f"Extraction prompt: ~{prompt_tokens} tokens")

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Log raw LLM response for debugging
                logger.debug(f"[RAW LLM RESPONSE] First 1000 chars: {response[:1000]}")

                # Parse response
                entries = self._parse_llm_response(response, dialogue_ids)
                
                # Log if entries are missing classification
                for entry in entries:
                    if not entry.memory_type:
                        logger.warning(f"[MISSING TYPE] Entry has no memory_type: {entry.lossless_restatement[:80]}")
                
                duration_ms = int((time.time() - start_time) * 1000)
                logger.debug(f"Memory extraction complete: {len(entries)} entries", 
                           entries=len(entries), duration_ms=duration_ms)
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}, retrying...")
                else:
                    logger.error(f"All {max_retries} attempts failed to parse LLM response: {e}")
                    if 'response' in locals():
                        logger.debug(f"Raw response (first 500 chars): {response[:500]}")
                    return []

    def _build_extraction_prompt(
        self,
        dialogue_text: str,
        dialogue_ids: List[int],
        context: str
    ) -> str:
        """
        Build LLM extraction prompt - Optimized for Nova Micro
        Uses XML tags, simple structure, and clear examples
        """
        return f"""<task>
Extract memories from the conversation below. Return a JSON array.
</task>

<rules>
1. Each memory MUST have these 3 required fields:
   - memory_type: "correction" OR "pattern" OR "factual"
   - scope: "universal" OR "entity"
   - confidence: 1.0 OR 0.8 OR 0.6

2. Use these values:
   - User corrected agent → memory_type="correction", confidence=1.0
   - Error or success pattern → memory_type="pattern", confidence=0.8
   - Data from tools → memory_type="factual", confidence=0.6

3. Include source context: WHY was this learned?
</rules>

<output_format>
Return ONLY a JSON array. No markdown. No explanation.
Start with [ and end with ]

Each object must have:
- lossless_restatement (string): The fact with source context
- keywords (array): Key terms
- memory_type (string): "correction" or "pattern" or "factual"
- scope (string): "universal" or "entity"
- confidence (number): 1.0 or 0.8 or 0.6
- topic (string): Short topic
- entities (array): Related entities
</output_format>

<example_input>
user: What is the payment schedule?
agent: The payment schedule is Same Day Payment.
user: No that's wrong. It's Annual Payment.
</example_input>

<example_output>
[
  {{
    "lossless_restatement": "The payment schedule is Annual Payment, not Same Day Payment. (Source: User corrected agent saying 'No that's wrong. It's Annual Payment.')",
    "keywords": ["payment schedule", "Annual Payment"],
    "memory_type": "correction",
    "scope": "universal",
    "confidence": 1.0,
    "topic": "Payment schedule correction",
    "entities": []
  }}
]
</example_output>

<example_input>
user: Query the database
agent: Running: SELECT * FROM users LIMIT 10
agent: Error: syntax error at or near LIMIT
agent: Running: SELECT * FROM users
agent: Success: Found 5 users
</example_input>

<example_output>
[
  {{
    "lossless_restatement": "LIMIT clause causes 'syntax error at or near LIMIT' in this database. Do not use LIMIT.",
    "keywords": ["LIMIT", "syntax error", "avoid"],
    "memory_type": "pattern",
    "scope": "universal",
    "confidence": 0.8,
    "topic": "SQL error pattern",
    "entities": []
  }},
  {{
    "lossless_restatement": "SELECT without LIMIT works in this database. Use simple SELECT for queries.",
    "keywords": ["SELECT", "working query"],
    "memory_type": "pattern",
    "scope": "universal",
    "confidence": 0.8,
    "topic": "SQL success pattern",
    "entities": []
  }}
]
</example_output>

{context}

<conversation>
{dialogue_text}
</conversation>

<instruction>
Extract memories from the conversation above. Return ONLY a JSON array.
Remember: EVERY object needs memory_type, scope, and confidence.
</instruction>
"""

    def _parse_llm_response(
        self,
        response: str,
        dialogue_ids: List[int]
    ) -> List[MemoryEntry]:
        """
        Parse LLM response to MemoryEntry list
        """
        # Extract JSON
        data = self.llm_client.extract_json(response)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")

        entries = []
        for item in data:
            # Create MemoryEntry with classification fields
            # Default to 'pattern' if memory_type not provided (safer than filtering out)
            memory_type = item.get("memory_type")
            if not memory_type:
                # Infer type from content
                content = item.get("lossless_restatement", "").lower()
                if "correction" in content or "corrected" in content or "source: user" in content:
                    memory_type = "correction"
                elif "failed" in content or "error" in content or "success" in content:
                    memory_type = "pattern"
                else:
                    memory_type = "pattern"  # Default to pattern so it gets stored
                logger.info(f"[AUTO-TYPE] Assigned memory_type='{memory_type}' to entry: {item.get('lossless_restatement', '')[:60]}...")
            
            entry = MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=item.get("timestamp"),
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic"),
                # Classification fields with defaults
                memory_type=memory_type,
                scope=item.get("scope", "universal"),
                source_entity=item.get("source_entity"),
                confidence=item.get("confidence", 0.8)
            )
            entries.append(entry)

        return entries
    
    def _process_windows_parallel(self, windows: List[List[Dialogue]]):
        """
        Process multiple windows in parallel using ThreadPoolExecutor
        """
        all_entries = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all window processing tasks
            future_to_window = {}
            for i, window in enumerate(windows):
                dialogue_ids = [d.dialogue_id for d in window]
                future = executor.submit(self._generate_memory_entries_worker, window, dialogue_ids, i+1)
                future_to_window[future] = (window, i+1)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_window):
                window, window_num = future_to_window[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"[Parallel Processing] Window {window_num} completed: {len(entries)} entries")
                except Exception as e:
                    print(f"[Parallel Processing] Window {window_num} failed: {e}")
        
        # Store all entries to database in batch with conflict resolution
        if all_entries:
            print(f"\n[Parallel Processing] Processing {len(all_entries)} entries...")
            
            # Filter out factual entries (volatile tool call data)
            entries_to_store = filter_for_storage(all_entries)
            print(f"[Parallel Processing] Storing {len(entries_to_store)} entries (filtered {len(all_entries) - len(entries_to_store)} factual)")
            
            # Use conflict resolution if enabled
            if entries_to_store:
                if self.conflict_resolver and self.enable_conflict_resolution:
                    print(f"[ConflictResolver] Checking for conflicts in batch...")
                    resolutions = self.conflict_resolver.resolve_conflicts(entries_to_store)
                    self.conflict_resolver.apply_resolutions(resolutions)
                else:
                    self.vector_store.add_entries(entries_to_store)
            
            self.processed_count += sum(len(window) for window in windows)
            
            # Update previous entries (use last window's entries for context - include all)
            if all_entries:
                self.previous_entries = all_entries[-10:]  # Keep last 10 entries for context
        
        print(f"[Parallel Processing] Completed processing {len(windows)} windows")
    
    def _generate_memory_entries_worker(self, window: List[Dialogue], dialogue_ids: List[int], window_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel processing of a single batch (full window or remaining dialogues)
        """
        batch_size = len(window)
        batch_type = "full window" if batch_size == self.window_size else f"remaining batch"
        print(f"[Worker {window_num}] Processing {batch_type} with {batch_size} dialogues")
        
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in window])
        
        # Build context (shared across all workers - this is fine for parallel processing)
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:  # Only show first 3
                context += f"- {entry.lossless_restatement}\n"

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Parse response
                entries = self._parse_llm_response(response, dialogue_ids)
                print(f"[Worker {window_num}] Generated {len(entries)} entries")
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Worker {window_num}] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"[Worker {window_num}] All {max_retries} attempts failed: {e}")
                    return []
