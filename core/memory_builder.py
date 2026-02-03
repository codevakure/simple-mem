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
    
    Stores valuable memory types:
    - correction: User corrections (confidence=1.0)
    - feedback: Explicit UI ratings (confidence=1.0)
    - pattern: Tool failures/successes with actionable lesson (confidence=0.9)
    - preference: User preferences about behavior (confidence=0.8)
    
    Skips:
    - factual: Volatile tool call data (raw query results, etc.)
    - insight: Merged into pattern (pattern now includes actionable guidance)
    
    This keeps the database focused on stable, valuable memories.
    """
    stored_types = {'correction', 'feedback', 'pattern', 'preference'}
    
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
Extract LEARNABLE memories from the conversation. Focus on user corrections, patterns, and actionable insights.
</task>

<memory_types>
1. CORRECTION (confidence=1.0) - User explicitly corrected the agent
   Source: User message correcting agent behavior
   MUST include:
   - What agent did wrong (the mistake)
   - WHY it was wrong (agent's flawed reasoning or approach)
   - What to do instead (correct approach)
   - Document/entity context if correction is specific to one document
   Look for: "No", "Wrong", "Actually", "Why did you", "Don't", "Use X instead"

2. FEEDBACK (confidence=1.0) - User gave explicit rating
   Source: UI thumbs up/down or explicit rating
   Look for: [FEEDBACK, thumbs_up, thumbs_down
   IMPORTANT: Messages starting with [FEEDBACK are ALWAYS feedback type!

3. PATTERN (confidence=0.9) - Learned rule, behavior, or requirement
   Source: Tool errors/retries OR user teaching a rule
   MUST include: The specific rule and when/how to apply it
   Examples:
   - From error: "LIMIT causes syntax error. Do NOT use LIMIT in this database."
   - From user: "Always check the corrections document before extracting fields."
   Look for: 
   - Error messages, exceptions, retries after failures
   - User teaching: "Always", "Must", "You need to", "Make sure", "Never", "Remember to"
   DO NOT extract patterns from: successful first-attempt executions, normal tool usage without errors

4. PREFERENCE (confidence=0.8) - User stated behavioral preference
   Source: User expressing how they want agent to behave
   Look for: "I prefer", "I like when", "I want you to"
   Note: Preferences are softer than patterns - they're style choices, not hard rules
</memory_types>

<scope_rules>
CRITICAL: Determine scope correctly to avoid hallucination!

scope="entity" when:
- Correction applies to ONE specific document, loan, account, or record
- Values are specific to that entity (e.g., "payment_amount should be 585.43" for Loan #12345)
- Include the entity identifier in source_entity field!

scope="universal" when:
- Correction applies to ALL documents/entities of this type
- It's a PROCESS or APPROACH issue, not a specific value
- Example: "Always check the corrections document before extracting fields"

WRONG: "payment_amount should be 585.43" with scope="universal" (would apply wrong value to other loans!)
RIGHT: "payment_amount should be 585.43 for Loan #12345" with scope="entity", source_entity="Loan #12345"
</scope_rules>

<reasoning_capture>
For CORRECTIONS, capture the agent's flawed reasoning if visible in [Reasoning] or [Thinking] blocks.
This helps prevent the same mistake on NEW documents.

PRESERVE SPECIFICS - Do NOT over-generalize!
When extracting corrections:
- Include SPECIFIC field names (e.g., "change_timing", "payment_amount")
- Include SPECIFIC wrong values (e.g., "DAILY", "MODIFIED FOLLOWING")
- Include SPECIFIC correct values (e.g., "A:CHANGE SAME DAY", "S:SAME DAY")
- If agent lists multiple field corrections, create SEPARATE memories for each

Example of GOOD memory:
"Field #42 change_timing was extracted as 'DAILY' but should be 'A:CHANGE SAME DAY'. 
Agent's flawed reasoning: 'Prime rates change daily, so Change Timing = DAILY'.
User corrected: 'You used the wrong method'. The mistake was assuming industry defaults 
instead of extracting actual values from the corrections document."

Example of BAD memory (too abstract):
"The agent made mistakes in field extractions by assuming values instead of extracting actual values."
(This is useless - doesn't say WHICH fields, WHAT values, or HOW to fix!)

MULTIPLE CORRECTIONS = MULTIPLE MEMORIES:
If agent admits correcting 5 different fields (#42, #57, #58, #36, #52), extract 5 separate memories 
with the specific field names, wrong values, and correct values for each.
</reasoning_capture>

<do_not_extract>
DO NOT extract memories from:
- Agent system instructions or guidelines (these are static config, not learnings)
- Routine successful tool calls without errors (only failures/retries are patterns)
- Generic agent responses without user feedback
- Normal conversational flow without corrections
- Agent thinking/reasoning about its own instructions
- Formatting guidelines, branding, or identity descriptions
</do_not_extract>

<rules>
1. Extract DISTINCT learnings - each memory should capture a unique insight from USER interaction
2. CONSOLIDATE related learnings - if an error and its fix are the same topic, that's ONE memory
3. For CORRECTIONS: Include the full context (mistake + WHY wrong + correct approach)
4. Include exact quotes from user when they correct or teach
5. Do NOT prefix lossless_restatement with type labels like "CORRECTION:", "PATTERN:", etc.
6. If conversation is routine Q&A without issues, return EMPTY array []
7. Set scope="entity" and source_entity for document-specific corrections!

STRICT: PRESERVE ALL SPECIFICS - NEVER GENERALIZE!
8. ALWAYS include specific identifiers: field names, parameter names, variable names, API names
9. ALWAYS include specific values: the WRONG value AND the CORRECT value with exact text
10. ALWAYS include entity/document IDs when correction is specific to one entity
11. If agent lists 5 field corrections, create 5 SEPARATE memories - one per field!
12. NEVER write vague memories like "agent made mistakes" or "values were wrong"
    BAD: "The agent made critical mistakes in field extractions"
    GOOD: "Field #42 change_timing was 'DAILY' but should be 'A:CHANGE SAME DAY'"
13. Keywords MUST include the specific field/parameter names and values, not generic words
    BAD: ["mistakes", "extraction", "values"]
    GOOD: ["change_timing", "DAILY", "A:CHANGE SAME DAY", "Field #42"]
</rules>

<output_format>
Return ONLY a JSON array. No markdown. No explanation.
Return [] if no learnable memories found.

Each object MUST have:
- lossless_restatement (string): Complete learning with full context and reasoning
- keywords (array): 3-5 key terms for search
- memory_type (string): "correction" | "feedback" | "pattern" | "preference"
- scope (string): "universal" (applies to ALL) | "entity" (specific to one document/record)
- source_entity (string): For entity scope, the document/loan/record ID this applies to
- confidence (number): 1.0 | 0.9 | 0.8
- topic (string): Short descriptive topic
- entities (array): Related entities (tables, tools, names)
</output_format>

<example_conversation_1>
user: Get me the data from the system
agent: [Reasoning] I'll query with a filter to limit results
agent: [Tool Call: fetch_data]
{{"query": "SELECT * FROM items", "limit": 10}}
agent: [Tool Result: fetch_data]
Error: 'limit' parameter is not supported
agent: [Reasoning] The limit parameter caused an error. I should try without it.
agent: [Tool Call: fetch_data]
{{"query": "SELECT * FROM items"}}
agent: [Tool Result: fetch_data]
{{"items": [...]}}
agent: Here's the data you requested.
</example_conversation_1>

<example_output_1>
[
  {{
    "lossless_restatement": "The 'limit' parameter causes error: 'limit parameter is not supported'. Do NOT use limit parameter - it is not supported in this system.",
    "keywords": ["limit", "parameter", "not supported", "error"],
    "memory_type": "pattern",
    "scope": "universal",
    "source_entity": null,
    "confidence": 0.9,
    "topic": "Limit parameter not supported",
    "entities": []
  }}
]
</example_output_1>

<example_conversation_2>
user: Process record ID-12345
agent: [Reasoning] I'll use default values for fields I can't find
agent: [Tool Call: process_record]
{{"record_id": "ID-12345", "field_a": "DEFAULT_A", "field_b": "DEFAULT_B"}}
agent: [Tool Result: process_record]
{{"field_a": "DEFAULT_A", "field_b": "DEFAULT_B"}}
agent: field_a = DEFAULT_A, field_b = DEFAULT_B
user: Wrong! field_a should be VALUE_X and field_b should be VALUE_Y. You used defaults instead of checking the source!
agent: [Reasoning] I made a mistake - I assumed defaults instead of reading the actual source. For ID-12345, the correct values are VALUE_X and VALUE_Y.
agent: You're right. field_a = VALUE_X and field_b = VALUE_Y.
</example_conversation_2>

<example_output_2>
[
  {{
    "lossless_restatement": "For record ID-12345, field_a was set to 'DEFAULT_A' but correct value is 'VALUE_X'. Agent assumed default instead of reading source. User corrected: 'You used defaults instead of checking the source!' This correction is specific to record ID-12345.",
    "keywords": ["field_a", "DEFAULT_A", "VALUE_X", "ID-12345"],
    "memory_type": "correction",
    "scope": "entity",
    "source_entity": "ID-12345",
    "confidence": 1.0,
    "topic": "ID-12345 field_a is VALUE_X",
    "entities": ["ID-12345", "field_a"]
  }},
  {{
    "lossless_restatement": "For record ID-12345, field_b was set to 'DEFAULT_B' but correct value is 'VALUE_Y'. Agent assumed default instead of reading source. User corrected: 'You used defaults instead of checking the source!' This correction is specific to record ID-12345.",
    "keywords": ["field_b", "DEFAULT_B", "VALUE_Y", "ID-12345"],
    "memory_type": "correction",
    "scope": "entity",
    "source_entity": "ID-12345",
    "confidence": 1.0,
    "topic": "ID-12345 field_b is VALUE_Y",
    "entities": ["ID-12345", "field_b"]
  }},
  {{
    "lossless_restatement": "Do NOT assume default values for fields. Always check the actual source data. Agent's mistake was using assumptions instead of extraction.",
    "keywords": ["defaults", "assume", "source", "check"],
    "memory_type": "pattern",
    "scope": "universal",
    "source_entity": null,
    "confidence": 0.9,
    "topic": "Check source instead of assuming defaults",
    "entities": []
  }}
]
</example_output_2>

<example_conversation_3>
user: Format the output for me
agent: [Tool Call: format_output]
{{"style": "compact"}}
agent: [Tool Result: format_output]
{{"formatted": "..."}}
agent: Here's the formatted output.
user: I prefer detailed format, not compact.
</example_conversation_3>

<example_output_3>
[
  {{
    "lossless_restatement": "User prefers 'detailed' format instead of 'compact'. User stated: 'I prefer detailed format, not compact.'",
    "keywords": ["format", "detailed", "compact", "preference"],
    "memory_type": "preference",
    "scope": "universal",
    "source_entity": null,
    "confidence": 0.8,
    "topic": "User prefers detailed format",
    "entities": []
  }}
]
</example_output_3>

{context}

<conversation>
{dialogue_text}
</conversation>

<instruction>
Extract memories from the conversation. Focus on user corrections, explicit feedback, and tool failures.
Return [] if conversation is routine without issues.

Return ONLY a JSON array. Each object needs memory_type, scope, confidence.
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
