"""
Conflict Resolver - Mem0-style ADD/UPDATE/DELETE Resolution

Implements conflict resolution for memory entries:
1. For each new fact, search existing memories for similar entries
2. If similar entries found, use LLM to decide action:
   - ADD: New distinct information, store as new entry
   - UPDATE: Modification of existing info, update the entry
   - DELETE: Contradiction, remove the old entry
   - NONE: Duplicate/redundant, skip storing
3. Apply actions to maintain consistent memory state

Reference: Mem0 paper - "Building Production-Ready AI Agents with Scalable Long-Term Memory"
"""
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from utils.logger import get_logger, estimate_tokens
from database.vector_store import VectorStore, get_vector_store
import config
import json

logger = get_logger(__name__)


class ConflictAction(Enum):
    ADD = "ADD"          # New information, add as new entry
    UPDATE = "UPDATE"    # Modify existing entry
    DELETE = "DELETE"    # Remove conflicting old entry
    NONE = "NONE"        # Skip, already exists or redundant


@dataclass
class ConflictResolution:
    """Result of conflict resolution for a single entry"""
    action: ConflictAction
    new_entry: MemoryEntry
    existing_entry_id: Optional[str] = None  # For UPDATE/DELETE
    merged_content: Optional[str] = None      # For UPDATE - the merged text
    reason: Optional[str] = None              # LLM's reasoning


class ConflictResolver:
    """
    Mem0-style Conflict Resolver
    
    Compares new memory entries against existing memories and uses LLM
    to decide the appropriate action for each entry.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        similarity_threshold: float = None,
        conflict_check_top_k: int = None
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold or getattr(
            config, 'CONFLICT_SIMILARITY_THRESHOLD', 0.75
        )
        self.conflict_check_top_k = conflict_check_top_k or getattr(
            config, 'CONFLICT_CHECK_TOP_K', 5
        )
    
    def resolve_conflicts(
        self,
        new_entries: List[MemoryEntry],
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[ConflictResolution]:
        """
        Resolve conflicts for a batch of new entries.
        
        Args:
            new_entries: List of new memory entries to check
            agent_id: Optional agent filter for scoped memory
            user_id: Optional user filter for scoped memory
            
        Returns:
            List of ConflictResolution objects with actions for each entry
        """
        start_time = time.time()
        logger.info(f"Resolving conflicts for {len(new_entries)} new entries", 
                   entries=len(new_entries), agent_id=agent_id)
        
        resolutions = []
        
        for i, entry in enumerate(new_entries):
            logger.debug(f"[{i+1}/{len(new_entries)}] Checking: {entry.lossless_restatement[:60]}...")
            resolution = self._resolve_single_entry(entry, agent_id, user_id)
            logger.debug(f"  -> {resolution.action.value}: {resolution.reason}")
            resolutions.append(resolution)
        
        duration_ms = int((time.time() - start_time) * 1000)
        actions_summary = {}
        for r in resolutions:
            actions_summary[r.action.value] = actions_summary.get(r.action.value, 0) + 1
        
        logger.info(f"Conflict resolution complete: {actions_summary}", duration_ms=duration_ms)
        
        return resolutions
    
    def _resolve_single_entry(
        self,
        new_entry: MemoryEntry,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ConflictResolution:
        """
        Resolve conflict for a single new entry.
        
        1. Search for similar existing memories
        2. If no similar entries found, return ADD
        3. If similar entries found, use LLM to decide action
        """
        # Search for similar existing memories
        similar_entries = self._find_similar_entries(
            new_entry.lossless_restatement,
            agent_id,
            user_id
        )
        
        # No similar entries - definitely ADD
        if not similar_entries:
            return ConflictResolution(
                action=ConflictAction.ADD,
                new_entry=new_entry,
                reason="No similar existing memories found"
            )
        
        # Similar entries found - use LLM to decide
        return self._llm_decide_action(new_entry, similar_entries)
    
    def _find_similar_entries(
        self,
        query: str,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Find existing entries similar to the query, scoped to agent/user.
        
        The search is scoped to the agent's memory space to ensure we only
        find conflicts within the same agent/user context.
        
        Returns list of (entry, similarity_score) tuples.
        """
        try:
            # Use semantic search with agent/user scoping
            # The vector store now handles filtering at query time for efficiency
            results = self.vector_store.semantic_search(
                query=query,
                top_k=self.conflict_check_top_k,
                agent_id=agent_id,
                user_id=user_id
            )
            
            logger.debug(f"Found {len(results)} similar entries (searched top {self.conflict_check_top_k})")
            
            # We don't have direct access to similarity scores from the current API
            # so we'll just return the entries and let LLM decide
            # The vector store already returns results ordered by similarity
            return [(entry, 0.8) for entry in results]  # Placeholder score
            
        except Exception as e:
            logger.error(f"Error searching for similar entries: {e}")
            return []
    
    def _llm_decide_action(
        self,
        new_entry: MemoryEntry,
        similar_entries: List[Tuple[MemoryEntry, float]]
    ) -> ConflictResolution:
        """
        Use LLM to decide the appropriate action for the new entry
        given similar existing entries.
        """
        # Build the prompt
        prompt = self._build_decision_prompt(new_entry, similar_entries)
        prompt_tokens = estimate_tokens(prompt)
        
        logger.debug(f"LLM conflict decision: comparing against {len(similar_entries)} entries", 
                    input_tokens=prompt_tokens)
        
        messages = [
            {
                "role": "system",
                "content": """You are a smart memory manager that decides how to handle new information relative to existing memories.

You can perform four operations:
1. ADD - The new fact is distinct new information, store it as a new entry
2. UPDATE - The new fact modifies/corrects existing information, update the existing entry
3. DELETE - The new fact contradicts/invalidates existing information, remove the old entry
4. NONE - The new fact is redundant/duplicate of existing information, skip it

Always respond with valid JSON in the exact format specified."""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        try:
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.1,
                response_format={"type": "json_object"} if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT else None
            )
            
            # Parse the response
            return self._parse_decision_response(response, new_entry, similar_entries)
            
        except Exception as e:
            logger.warning(f"LLM decision failed: {e}, defaulting to ADD")
            return ConflictResolution(
                action=ConflictAction.ADD,
                new_entry=new_entry,
                reason=f"LLM decision failed: {e}"
            )
    
    def _build_decision_prompt(
        self,
        new_entry: MemoryEntry,
        similar_entries: List[Tuple[MemoryEntry, float]]
    ) -> str:
        """Build the prompt for LLM decision."""
        
        existing_memories = []
        for i, (entry, score) in enumerate(similar_entries):
            existing_memories.append({
                "id": str(i),
                "text": entry.lossless_restatement,
                "entry_id": entry.entry_id,
                "memory_type": getattr(entry, 'memory_type', 'factual')
            })
        
        # Detect if new entry is an error/pattern type
        new_text = new_entry.lossless_restatement.lower()
        is_error_pattern = any(kw in new_text for kw in ['fail', 'error', 'avoid', 'syntax error', 'does not work', 'invalid'])
        is_success_pattern = any(kw in new_text for kw in ['successful', 'works', 'use this', 'correct approach'])
        
        pattern_guidance = ""
        if is_error_pattern or is_success_pattern:
            pattern_guidance = """
[IMPORTANT - Pattern Memory]
The new fact appears to be an ERROR PATTERN or SUCCESS PATTERN (learning from experience).
These are CRITICAL for avoiding repeated mistakes. Be VERY conservative with NONE:
- ERROR patterns (failures, what doesn't work) should ALWAYS be added unless an IDENTICAL error pattern already exists
- SUCCESS patterns (what works) should ALWAYS be added unless an IDENTICAL success pattern already exists
- Factual information about data/schema is NOT the same as error/success patterns
- Different error messages or different approaches are DISTINCT, not redundant"""
        
        return f"""Compare the new fact against existing memories and decide the appropriate action.

[New Fact]
{new_entry.lossless_restatement}

[Existing Similar Memories]
{json.dumps(existing_memories, indent=2)}
{pattern_guidance}

[Decision Criteria]
- ADD: New fact contains distinct information not covered by existing memories (DEFAULT for patterns)
- UPDATE: New fact modifies, corrects, or adds details to an existing memory (specify which one to update and provide merged content)
- DELETE: New fact contradicts or invalidates an existing memory (specify which one to delete)
- NONE: New fact is essentially an EXACT DUPLICATE of existing information (use ONLY if text says essentially the same thing)

[IMPORTANT] Only use NONE if the new fact says essentially the SAME thing as an existing memory.
Different types of information (error patterns vs schema facts vs success patterns) are NEVER duplicates.

[Output Format]
Return JSON with exactly this structure:
{{
    "action": "ADD" | "UPDATE" | "DELETE" | "NONE",
    "existing_id": "id of existing memory to update/delete (only for UPDATE/DELETE)",
    "merged_content": "merged text combining old and new info (only for UPDATE)",
    "reason": "brief explanation of your decision"
}}

Respond with only the JSON object, no additional text."""

    def _parse_decision_response(
        self,
        response: str,
        new_entry: MemoryEntry,
        similar_entries: List[Tuple[MemoryEntry, float]]
    ) -> ConflictResolution:
        """Parse LLM response into ConflictResolution."""
        try:
            data = self.llm_client.extract_json(response)
            
            action_str = data.get("action", "ADD").upper()
            action = ConflictAction[action_str] if action_str in ConflictAction.__members__ else ConflictAction.ADD
            
            existing_id = data.get("existing_id")
            merged_content = data.get("merged_content")
            reason = data.get("reason", "")
            
            # Get the actual entry_id from similar entries if UPDATE/DELETE
            actual_entry_id = None
            if existing_id is not None and action in (ConflictAction.UPDATE, ConflictAction.DELETE):
                try:
                    idx = int(existing_id)
                    if 0 <= idx < len(similar_entries):
                        actual_entry_id = similar_entries[idx][0].entry_id
                        logger.debug(f"Resolved existing_id={existing_id} to entry_id={actual_entry_id}")
                except (ValueError, IndexError):
                    logger.warning(f"Could not resolve existing_id={existing_id} to entry_id")
            
            return ConflictResolution(
                action=action,
                new_entry=new_entry,
                existing_entry_id=actual_entry_id,
                merged_content=merged_content,
                reason=reason
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse decision: {e}, defaulting to ADD")
            return ConflictResolution(
                action=ConflictAction.ADD,
                new_entry=new_entry,
                reason=f"Parse error: {e}"
            )
    
    def apply_resolutions(
        self,
        resolutions: List[ConflictResolution]
    ) -> Dict[str, int]:
        """
        Apply the resolved actions to the vector store.
        
        Returns counts of each action taken.
        """
        start_time = time.time()
        counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
        entries_to_add = []
        
        for resolution in resolutions:
            action = resolution.action
            counts[action.value] += 1
            
            if action == ConflictAction.ADD:
                entries_to_add.append(resolution.new_entry)
                logger.debug(f"  ADD: {resolution.new_entry.lossless_restatement[:50]}...")
                
            elif action == ConflictAction.UPDATE:
                if resolution.existing_entry_id and resolution.merged_content:
                    # Delete old entry
                    try:
                        self.vector_store.delete_entry(resolution.existing_entry_id)
                        logger.debug(f"  UPDATE: Deleted old entry {resolution.existing_entry_id}")
                    except Exception as e:
                        logger.error(f"Failed to delete for update: {e}")
                    
                    # Create updated entry (preserving agent_id/user_id from new_entry)
                    updated_entry = MemoryEntry(
                        lossless_restatement=resolution.merged_content,
                        keywords=resolution.new_entry.keywords,
                        timestamp=resolution.new_entry.timestamp,
                        location=resolution.new_entry.location,
                        persons=resolution.new_entry.persons,
                        entities=resolution.new_entry.entities,
                        topic=resolution.new_entry.topic,
                        memory_type=resolution.new_entry.memory_type,
                        scope=resolution.new_entry.scope,
                        source_entity=resolution.new_entry.source_entity,
                        confidence=resolution.new_entry.confidence,
                        agent_id=resolution.new_entry.agent_id,
                        user_id=resolution.new_entry.user_id
                    )
                    entries_to_add.append(updated_entry)
                    logger.debug(f"  UPDATE: Created merged entry: {resolution.merged_content[:50]}...")
                else:
                    # Fallback to ADD if we don't have the info for update
                    entries_to_add.append(resolution.new_entry)
                    logger.debug(f"  UPDATE->ADD (fallback): Missing merge info")
                    
            elif action == ConflictAction.DELETE:
                if resolution.existing_entry_id:
                    try:
                        self.vector_store.delete_entry(resolution.existing_entry_id)
                        logger.debug(f"  DELETE: Removed entry {resolution.existing_entry_id}")
                    except Exception as e:
                        logger.error(f"Failed to delete: {e}")
                # Also add the new entry that triggered the delete
                entries_to_add.append(resolution.new_entry)
                    
            elif action == ConflictAction.NONE:
                logger.debug(f"  NONE: Skipped duplicate: {resolution.new_entry.lossless_restatement[:50]}...")
        
        # Batch add all new entries
        if entries_to_add:
            self.vector_store.add_entries(entries_to_add)
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Applied resolutions: ADD={counts['ADD']}, UPDATE={counts['UPDATE']}, DELETE={counts['DELETE']}, NONE={counts['NONE']}, stored={len(entries_to_add)}",
            duration_ms=duration_ms
        )
        return counts
