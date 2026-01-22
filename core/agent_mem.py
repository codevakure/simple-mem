"""
Multi-Agent Memory - Clean approach using database columns

Simple solution: Add agent_id and user_id columns to the database table,
filter on retrieval. No tags, no prefixes, just clean SQL filtering.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from models.memory_entry import MemoryEntry, Dialogue
from database.vector_store import get_vector_store
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient
from core.memory_builder import MemoryBuilder
from core.hybrid_retriever import HybridRetriever
import config


# =============================================================================
# Shared instances for performance (singleton pattern)
# =============================================================================
_shared_embedding_model: Optional[EmbeddingModel] = None
_shared_llm_client: Optional[LLMClient] = None
_shared_vector_store = None


def _get_shared_components():
    """Get or create shared components (connection pooling effect)."""
    global _shared_embedding_model, _shared_llm_client, _shared_vector_store
    
    if _shared_embedding_model is None:
        _shared_embedding_model = EmbeddingModel()
    
    if _shared_llm_client is None:
        _shared_llm_client = LLMClient()
    
    if _shared_vector_store is None:
        _shared_vector_store = get_vector_store(embedding_model=_shared_embedding_model)
    
    return _shared_embedding_model, _shared_llm_client, _shared_vector_store


class AgentMemory:
    """
    Memory system for a specific agent with optional user filtering.
    
    Uses SimpleMem's core capabilities:
    - De-linearization: LLM extracts atomic facts from conversations
    - Hybrid Retrieval: semantic + lexical + symbolic search
    
    Adds agent/user isolation via database columns.
    
    Performance: Shares DB connection, embedding model, and LLM client
    across all instances (safe for 2000+ users).
    
    Retrieval Modes:
    - Fast (default): Vector search only, ~150-200ms
    - Deep: LLM planning + reflection, ~2-3s (for complex multi-hop queries)
    """
    
    def __init__(
        self, 
        agent_id: str = None, 
        user_id: str = None, 
        clear_db: bool = False,
        enable_deep_analysis: bool = False
    ):
        """
        Initialize memory for a specific agent.
        
        Args:
            agent_id: Unique identifier for this agent (e.g., "sql_agent")
            user_id: Optional user identifier for user-specific memories
            clear_db: If True, clears ALL memories (use carefully!)
            enable_deep_analysis: If True, use LLM planning/reflection (slower but better for complex queries)
        """
        self.agent_id = agent_id
        self.user_id = user_id
        self.enable_deep_analysis = enable_deep_analysis
        
        # Use shared components (not new instances each time!)
        self.embedding_model, self.llm_client, self.vector_store = _get_shared_components()
        
        if clear_db:
            self.vector_store.clear()
        
        # Create wrapped memory builder that tags entries
        self.memory_builder = _AgentMemoryBuilder(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            agent_id=agent_id,
            user_id=user_id
        )
        
        # Create wrapped retriever that filters by agent/user
        self.retriever = _AgentRetriever(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            agent_id=agent_id,
            user_id=user_id,
            enable_deep_analysis=enable_deep_analysis
        )
        
        self._dialogue_counter = 0
    
    def add_dialogue(self, speaker: str, content: str, timestamp: str = None):
        """
        Add a dialogue turn from conversation.
        
        Example:
            memory.add_dialogue("user", "Show me Q4 revenue for Texas Capital")
            memory.add_dialogue("agent", "Based on revenue_table, it's $1M")
            memory.add_dialogue("user", "Wrong! Use sales_data.quarterly_revenue")
            memory.finalize()
        
        SimpleMem's De-linearization will automatically extract corrections.
        """
        self._dialogue_counter += 1
        dialogue = Dialogue(
            dialogue_id=self._dialogue_counter,
            speaker=speaker,
            content=content,
            timestamp=timestamp or datetime.now().isoformat()
        )
        self.memory_builder.add_dialogue(dialogue, auto_process=False)
    
    def finalize(self) -> int:
        """
        Finalize conversation - extracts memories using LLM (blocking).
        
        Call at end of conversation or natural breakpoints.
        Returns number of memories created.
        
        For non-blocking, use finalize_async() instead.
        """
        before = self.vector_store.count_rows()
        self.memory_builder.process_remaining()
        after = self.vector_store.count_rows()
        return after - before
    
    def finalize_async(self) -> None:
        """
        Finalize conversation in background thread (non-blocking).
        
        Agent doesn't wait - memory extraction happens in background.
        Use this when you don't need the count of memories created.
        """
        import threading
        
        def _background_finalize():
            try:
                self.memory_builder.process_remaining()
            except Exception as e:
                print(f"[Background Finalize] Error: {e}")
        
        thread = threading.Thread(target=_background_finalize, daemon=True)
        thread.start()
    
    def get_context(self, query: str, max_results: int = 10) -> List[MemoryEntry]:
        """
        Get relevant memories for a query (filtered to this agent/user).
        
        Uses hybrid retrieval: semantic + lexical + symbolic search.
        """
        return self.retriever.retrieve(query)[:max_results]
    
    def get_context_string(self, query: str, max_results: int = 10) -> str:
        """
        Get formatted context string to inject into agent's prompt.
        
        Example:
            context = memory.get_context_string(user_question)
            prompt = f'''
            RELEVANT CONTEXT FROM MEMORY:
            {context}
            
            USER QUESTION: {user_question}
            '''
        """
        memories = self.get_context(query, max_results)
        
        if not memories:
            return "No relevant memories found."
        
        lines = [f"{i}. {m.lossless_restatement}" for i, m in enumerate(memories, 1)]
        return "\n".join(lines)
    
    def add_knowledge(self, content: str, category: str = "general") -> str:
        """
        Directly add knowledge without conversation flow.
        
        Use to pre-load schemas, rules, preferences.
        """
        entry_id = f"knowledge_{self.agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            lossless_restatement=content,
            keywords=[category, "knowledge"],
            timestamp=datetime.now().isoformat(),
            topic=category,
            agent_id=self.agent_id,
            user_id=self.user_id
        )
        
        self.vector_store.add_entries([entry])
        return entry_id
    
    def get_all_memories(self) -> List[MemoryEntry]:
        """Get all memories for this agent (and user if specified)."""
        return self.vector_store.get_all_entries(
            agent_id=self.agent_id,
            user_id=self.user_id
        )
    
    def get_memory_count(self) -> int:
        """Get count of memories for this agent."""
        return len(self.get_all_memories())
    
    def clear_memories(self) -> int:
        """Clear only this agent's memories."""
        return self.vector_store.delete_by_agent(self.agent_id)


class _AgentMemoryBuilder(MemoryBuilder):
    """Extended MemoryBuilder that adds agent_id/user_id to entries and deduplicates."""
    
    def __init__(self, llm_client, vector_store, agent_id: str, user_id: str = None, 
                 deduplicate: bool = True, similarity_threshold: float = 0.85, **kwargs):
        super().__init__(llm_client, vector_store, **kwargs)
        self.agent_id = agent_id
        self.user_id = user_id
        self.deduplicate = deduplicate
        self.similarity_threshold = similarity_threshold
    
    def _generate_memory_entries(self, dialogues) -> List[MemoryEntry]:
        """Override to add agent_id/user_id and deduplicate similar entries."""
        entries = super()._generate_memory_entries(dialogues)
        
        # Add agent_id and user_id to each entry
        for entry in entries:
            entry.agent_id = self.agent_id
            entry.user_id = self.user_id
        
        # Deduplicate: check if similar content already exists
        if self.deduplicate and entries:
            entries = self._deduplicate_entries(entries)
        
        return entries
    
    def _deduplicate_entries(self, new_entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Remove entries that are too similar to existing ones using embedding similarity.
        
        Dedup checks at AGENT level (not user level) - shared knowledge.
        If john says "Use sales_data" and jane says the same, only first is stored.
        """
        unique_entries = []
        
        for entry in new_entries:
            # Quick semantic search to find similar existing entries
            # Note: agent_id only (no user_id) for shared agent knowledge
            try:
                similar = self.vector_store.semantic_search(
                    entry.lossless_restatement,
                    top_k=3,
                    agent_id=self.agent_id,
                    user_id=None  # Agent-level dedup (shared knowledge)
                )
                
                # Check if any existing entry is too similar using embedding similarity
                is_duplicate = False
                if similar:
                    # Get embeddings for comparison
                    new_embedding = self.vector_store.embedding_model.encode_single(
                        entry.lossless_restatement, is_query=False
                    )
                    
                    for existing in similar:
                        # Get existing entry's embedding via another search (it's in the DB)
                        existing_embedding = self.vector_store.embedding_model.encode_single(
                            existing.lossless_restatement, is_query=False
                        )
                        
                        # Cosine similarity
                        import numpy as np
                        similarity = np.dot(new_embedding, existing_embedding) / (
                            np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
                        )
                        
                        if similarity > self.similarity_threshold:
                            print(f"[Dedup] Skipping duplicate (sim={similarity:.3f}): {entry.lossless_restatement[:60]}...")
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_entries.append(entry)
                    
            except Exception as e:
                # If dedup check fails, keep the entry
                print(f"[Dedup] Check failed: {e}")
                unique_entries.append(entry)
        
        if len(unique_entries) < len(new_entries):
            print(f"[Dedup] Kept {len(unique_entries)}/{len(new_entries)} entries (removed {len(new_entries) - len(unique_entries)} duplicates)")
        
        return unique_entries
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity (Jaccard) - fallback."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class _AgentRetriever(HybridRetriever):
    """
    Fast retriever for agent memory - disables planning/reflection for speed.
    
    SimpleMem's full HybridRetriever does:
    - Planning: LLM decomposes query into sub-queries (~500ms)
    - Reflection: 2 rounds checking completeness (~1000ms each)
    
    For agent memory, we just need fast vector search (~50-100ms).
    """
    
    def __init__(
        self, 
        llm_client, 
        vector_store, 
        agent_id: str, 
        user_id: str = None, 
        enable_deep_analysis: bool = False,
        **kwargs
    ):
        """
        Initialize retriever.
        
        Args:
            enable_deep_analysis: If True, use full LLM planning/reflection.
                                  If False (default), fast vector search only.
        """
        self.enable_deep_analysis = enable_deep_analysis
        
        # Set planning/reflection based on deep_analysis flag
        super().__init__(
            llm_client, 
            vector_store,
            enable_planning=enable_deep_analysis,
            enable_reflection=enable_deep_analysis,
            **kwargs
        )
        self.agent_id = agent_id
        self.user_id = user_id
    
    def retrieve(self, query: str, enable_reflection: bool = None) -> List[MemoryEntry]:
        """
        Retrieve relevant memories.
        
        If deep_analysis is enabled, uses full LLM planning/reflection.
        Otherwise, fast semantic + keyword search only.
        """
        if self.enable_deep_analysis:
            # Use parent's full planning/reflection pipeline
            return super().retrieve(query, enable_reflection=enable_reflection)
        
        # Fast path: just semantic + keyword search, no LLM calls
        semantic_results = self._semantic_search(query)
        
        # Simple keyword extraction (no LLM)
        keywords = self._extract_keywords_simple(query)
        keyword_results = self.vector_store.keyword_search(
            keywords, 
            top_k=self.keyword_top_k,
            agent_id=self.agent_id,
            user_id=self.user_id
        ) if keywords else []
        
        # Merge and dedupe
        seen_ids = set()
        results = []
        for entry in semantic_results + keyword_results:
            if entry.entry_id not in seen_ids:
                seen_ids.add(entry.entry_id)
                results.append(entry)
        
        return results
    
    def _extract_keywords_simple(self, query: str) -> List[str]:
        """Extract keywords without LLM - just split and filter."""
        import re
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'what', 'where', 'when', 'who', 'how', 'why', 'which',
                     'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with', 'from',
                     'and', 'or', 'but', 'not', 'this', 'that', 'it', 'i', 'you',
                     'me', 'my', 'your', 'can', 'could', 'would', 'should', 'do', 'does'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords[:5]  # Top 5 keywords
    
    def _semantic_search(self, query: str) -> List[MemoryEntry]:
        """Override to filter by agent_id/user_id."""
        return self.vector_store.semantic_search(
            query, 
            top_k=self.semantic_top_k,
            agent_id=self.agent_id,
            user_id=self.user_id
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_all_agents() -> List[Dict[str, Any]]:
    """Get list of all agents with memory counts."""
    embedding_model = EmbeddingModel()
    vector_store = get_vector_store(embedding_model=embedding_model)
    return vector_store.get_agents()


def get_all_users(agent_id: str = None) -> List[Dict[str, Any]]:
    """Get list of all users with memory counts."""
    embedding_model = EmbeddingModel()
    vector_store = get_vector_store(embedding_model=embedding_model)
    return vector_store.get_users(agent_id=agent_id)


def delete_agent_memories(agent_id: str) -> int:
    """Delete all memories for an agent."""
    embedding_model = EmbeddingModel()
    vector_store = get_vector_store(embedding_model=embedding_model)
    return vector_store.delete_by_agent(agent_id)


def delete_user_memories(user_id: str, agent_id: str = None) -> int:
    """Delete all memories for a user."""
    embedding_model = EmbeddingModel()
    vector_store = get_vector_store(embedding_model=embedding_model)
    return vector_store.delete_by_user(user_id, agent_id=agent_id)
