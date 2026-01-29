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
            
            # Use conflict resolution if enabled
            if self.conflict_resolver and self.enable_conflict_resolution:
                logger.debug("Running conflict resolution...")
                resolutions = self.conflict_resolver.resolve_conflicts(
                    entries, agent_id=agent_id, user_id=user_id
                )
                counts = self.conflict_resolver.apply_resolutions(resolutions)
            else:
                # Direct add without conflict resolution
                self.vector_store.add_entries(entries)
            
            self.previous_entries = entries  # Save as context
            self.processed_count += len(window)

        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Window complete: {len(entries)} entries generated", 
                   entries=len(entries), duration_ms=duration_ms)

    def process_remaining(self, agent_id: str = None, user_id: str = None):
        """
        Process remaining dialogues (fallback method, normally handled in parallel)
        
        Args:
            agent_id: Optional agent ID for scoped conflict resolution
            user_id: Optional user ID for scoped conflict resolution
        """
        if self.dialogue_buffer:
            start_time = time.time()
            remaining_count = len(self.dialogue_buffer)
            
            logger.info(f"Processing remaining {remaining_count} dialogues (fallback mode)", 
                       agent_id=agent_id)
            
            entries = self._generate_memory_entries(self.dialogue_buffer)
            if entries:
                # Set agent_id and user_id on all entries
                for entry in entries:
                    entry.agent_id = agent_id
                    entry.user_id = user_id
                
                logger.info(f"Extracted {len(entries)} memory entries from remaining dialogues")
                for i, entry in enumerate(entries, 1):
                    logger.debug(f"  [{i}] {entry.lossless_restatement[:80]}{'...' if len(entry.lossless_restatement) > 80 else ''}")
                
                # Use conflict resolution if enabled
                if self.conflict_resolver and self.enable_conflict_resolution:
                    logger.debug("Running conflict resolution...")
                    resolutions = self.conflict_resolver.resolve_conflicts(
                        entries, agent_id=agent_id, user_id=user_id
                    )
                    self.conflict_resolver.apply_resolutions(resolutions)
                else:
                    self.vector_store.add_entries(entries)
                
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

                # Parse response
                entries = self._parse_llm_response(response, dialogue_ids)
                
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
        Build LLM extraction prompt
        """
        return f"""
Your task is to extract all valuable KNOWLEDGE and FACTS from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[CRITICAL: What to Extract]
Focus on extracting FACTS and KNOWLEDGE, not just conversation flow:

1. **Tool/Action Results** (memory_type: "factual", confidence: 0.8)
   - Extract the ACTUAL DATA returned from tools
   - BAD: "The user queried the database for tables"
   - GOOD: "The database contains one table called 'memory_entries' in the public schema"

2. **User Corrections** (memory_type: "correction", confidence: 1.0)
   - When user corrects the agent, extract the CORRECTED information
   - This is high-confidence because user explicitly confirmed it
   - BAD: "The user corrected the agent about payment schedule"
   - GOOD: "Loan #12345 has Annual Payment schedule, not Same Day Payment"

3. **ERROR PATTERNS - CRITICAL FOR LEARNING** (memory_type: "pattern", scope: "universal", confidence: 0.7)
   - When agent tries something and it FAILS, extract what went wrong
   - Include the specific error message and what caused it
   - BAD: "The agent ran a SQL query"
   - GOOD: "SQL query with LIMIT clause fails with 'syntax error at or near LIMIT' in this database - use standard SELECT without LIMIT instead"
   - This prevents repeating the same failed approach

4. **SUCCESS PATTERNS - CRITICAL FOR LEARNING** (memory_type: "pattern", scope: "universal", confidence: 0.8)
   - When agent finally succeeds after failures, extract what WORKED
   - BAD: "The agent got the schema"
   - GOOD: "To get table schema in this database, use: SELECT column_name, data_type FROM information_schema.columns WHERE table_name='<table>' - this works reliably"
   - This helps reuse successful approaches

5. **Derived Patterns** (memory_type: "pattern", confidence: 0.6)
   - General insights derived from corrections with provenance
   - Helps avoid same mistake in future
   - MUST include source_entity to show where insight came from

6. **Technical Details** (memory_type: "factual")
   - Extract specific technical facts discovered

7. **Learned Preferences/Rules** (memory_type: "correction" if user stated, "pattern" if inferred)
   - User preferences, constraints, or rules mentioned

[CRITICAL: Memory Classification]

**memory_type** - How was this knowledge obtained?
- "factual": Direct data from tool results, documents, APIs
- "correction": User explicitly corrected agent (highest trust)
- "pattern": Derived insight/rule with provenance (use cautiously)

**scope** - How broadly should this apply?
- "entity": Applies ONLY to a specific entity (loan #12345, customer ABC)
- "universal": Applies broadly (all 30-year loans, all users, system-wide rule)

**source_entity** - For patterns, which entity was this derived from?
- Example: "Loan #12345" - so Ranger knows the provenance

**confidence** - How reliable is this memory?
- 1.0: User correction (explicit confirmation)
- 0.8: Tool/document result (direct data)
- 0.6: Derived pattern (inferred, use with caution)

[Output Format]
Return a JSON array:

```json
[
  {{
    "lossless_restatement": "Complete factual statement",
    "keywords": ["keyword1", "keyword2", ...],
    "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
    "location": "location name or null",
    "persons": ["name1", ...],
    "entities": ["entity1", ...],
    "topic": "topic phrase",
    "memory_type": "factual|correction|pattern",
    "scope": "entity|universal",
    "source_entity": "entity ID for patterns, null for direct facts",
    "confidence": 0.6|0.8|1.0
  }}
]
```

[Example - Tool Result (factual, entity-specific)]
```json
{{
  "lossless_restatement": "Loan #12345 (John Smith) has loan amount of $250,000, 30-year term, and 6.5% interest rate.",
  "keywords": ["Loan #12345", "John Smith", "$250,000", "30-year", "6.5%"],
  "entities": ["Loan #12345"],
  "topic": "Loan financial details",
  "memory_type": "factual",
  "scope": "entity",
  "source_entity": null,
  "confidence": 0.8
}}
```

[Example - User Correction (correction, entity-specific)]
```json
{{
  "lossless_restatement": "Loan #12345 (John Smith) has Annual Payment schedule, not Same Day Payment.",
  "keywords": ["Loan #12345", "Annual Payment", "payment schedule"],
  "entities": ["Loan #12345"],
  "topic": "Loan payment schedule correction",
  "memory_type": "correction",
  "scope": "entity",
  "source_entity": null,
  "confidence": 1.0
}}
```

[Example - Derived Pattern (pattern, universal)]
```json
{{
  "lossless_restatement": "When extracting loan payment schedules, verify the value carefully - 'Annual Payment' may be misread as 'Same Day Payment' (discovered from Loan #12345 correction).",
  "keywords": ["loan extraction", "payment schedule", "verification"],
  "entities": [],
  "topic": "Loan extraction quality check",
  "memory_type": "pattern",
  "scope": "universal",
  "source_entity": "Loan #12345",
  "confidence": 0.6
}}
```

[Example - Universal Rule from User (correction, universal)]
```json
{{
  "lossless_restatement": "For 30-year term loans, the payment schedule must be either Annual Payment or Monthly Payment. Same Day Payment is NOT valid.",
  "keywords": ["30-year loans", "payment schedule", "Annual Payment", "Monthly Payment"],
  "entities": [],
  "topic": "Loan payment schedule rule",
  "memory_type": "correction",
  "scope": "universal",
  "source_entity": null,
  "confidence": 1.0
}}
```

[Example - Technical Fact (factual, universal)]
```json
{{
  "lossless_restatement": "The search_vector column in memory_entries table is used for embedding-based semantic search, NOT PostgreSQL full-text search, despite using tsvector data type.",
  "keywords": ["search_vector", "embedding search", "semantic search"],
  "entities": ["search_vector", "memory_entries"],
  "topic": "search_vector column purpose",
  "memory_type": "correction",
  "scope": "universal",
  "source_entity": null,
  "confidence": 1.0
}}
```

[Example - Error Pattern - PREVENTS REPEATING FAILURES]
```json
{{
  "lossless_restatement": "Query execution failed with 'syntax error at or near LIMIT' when using: SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema') LIMIT 10. This database does not support standard LIMIT clause.",
  "keywords": ["LIMIT", "syntax error", "query failed", "information_schema.tables", "avoid LIMIT"],
  "entities": ["information_schema.tables"],
  "topic": "SQL query error pattern",
  "memory_type": "pattern",
  "scope": "universal",
  "source_entity": null,
  "confidence": 0.7
}}
```

[Example - Success Pattern - REUSES WORKING APPROACHES]
```json
{{
  "lossless_restatement": "Successfully retrieved table schema using: SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'memory_entries' ORDER BY ordinal_position. Use this approach for schema queries in this database.",
  "keywords": ["information_schema.columns", "schema query", "successful pattern", "column_name", "data_type"],
  "entities": ["information_schema.columns", "memory_entries"],
  "topic": "SQL successful query pattern",
  "memory_type": "pattern",
  "scope": "universal",
  "source_entity": null,
  "confidence": 0.8
}}
```

Now process the dialogues above. Return ONLY the JSON array with classified knowledge.
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
            entry = MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=item.get("timestamp"),
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic"),
                # New classification fields
                memory_type=item.get("memory_type"),
                scope=item.get("scope"),
                source_entity=item.get("source_entity"),
                confidence=item.get("confidence")
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
            
            # Use conflict resolution if enabled
            if self.conflict_resolver and self.enable_conflict_resolution:
                print(f"[ConflictResolver] Checking for conflicts in batch...")
                resolutions = self.conflict_resolver.resolve_conflicts(all_entries)
                self.conflict_resolver.apply_resolutions(resolutions)
            else:
                self.vector_store.add_entries(all_entries)
            
            self.processed_count += sum(len(window) for window in windows)
            
            # Update previous entries (use last window's entries for context)
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
