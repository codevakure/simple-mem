"""
Answer Generator - Final synthesis from retrieved atomic contexts

Paper Reference: Section 3.3 - Reconstructive Synthesis (Read Path)
Generates answers from the final context C_final synthesized by query-aware retrieval
"""
from typing import List
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
import config


class AnswerGenerator:
    """
    Answer Generator - Reconstructive Synthesis from Atomic Contexts

    Paper Reference: Section 3.3 - Eq. (10)
    Synthesizes final answer from pruned, query-specific context:
    C_final = ⊕_{m ∈ Top-k_dyn(S)} [t_m: Content(m)]

    Features:
    1. Receive query and retrieved atomic entries
    2. Generate answers from disambiguated, self-contained facts
    3. Ensure accuracy through atomic context independence
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_answer(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Generate answer

        Args:
        - query: User question
        - contexts: List of retrieved relevant MemoryEntry

        Returns:
        - Generated answer (concise phrase)
        """
        if not contexts:
            return "No relevant information found"

        # Build context string
        context_str = self._format_contexts(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        # Call LLM to generate answer
        messages = [
            {
                "role": "system",
                "content": "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
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

                # Parse JSON response
                result = self.llm_client.extract_json(response)
                # Return the answer from JSON
                return result.get("answer", response.strip())

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    # Fallback to raw response
                    if 'response' in locals():
                        return response.strip()
                    else:
                        return "Failed to generate answer"

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts to readable text with special handling for pattern memories.
        """
        # Separate pattern memories from factual memories
        patterns = []
        facts = []
        
        for entry in contexts:
            if entry.memory_type == "pattern":
                patterns.append(entry)
            else:
                facts.append(entry)
        
        formatted = []
        
        # Show patterns first if they exist (these guide behavior)
        if patterns:
            formatted.append("=" * 60)
            formatted.append("LEARNED PATTERNS (Success/Error Guidance)")
            formatted.append("=" * 60)
            for i, entry in enumerate(patterns, 1):
                parts = [f"[Pattern {i}] Confidence: {entry.confidence:.1f} | Scope: {entry.scope}"]
                parts.append(f"  Guidance: {entry.lossless_restatement}")
                if entry.keywords:
                    parts.append(f"  Keywords: {', '.join(entry.keywords[:5])}")
                formatted.append("\n".join(parts))
            formatted.append("")
        
        # Then show factual content
        if facts:
            formatted.append("=" * 60)
            formatted.append("FACTUAL CONTEXT")
            formatted.append("=" * 60)
            for i, entry in enumerate(facts, 1):
                parts = [f"[Context {i}]"]
                parts.append(f"Content: {entry.lossless_restatement}")

                if entry.timestamp:
                    parts.append(f"Time: {entry.timestamp}")

                if entry.location:
                    parts.append(f"Location: {entry.location}")

                if entry.persons:
                    parts.append(f"Persons: {', '.join(entry.persons)}")

                if entry.entities:
                    parts.append(f"Related Entities: {', '.join(entry.entities)}")

                if entry.topic:
                    parts.append(f"Topic: {entry.topic}")

                formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """
        Build answer generation prompt with pattern handling
        """
        return f"""
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

CRITICAL INSTRUCTIONS FOR PATTERN MEMORIES:

If you see "LEARNED PATTERNS" section above:
- These are ERROR PATTERNS (what failed) and SUCCESS PATTERNS (what worked)
- ERROR PATTERNS tell you what approaches to AVOID (e.g., "LIMIT clause fails with syntax error")
- SUCCESS PATTERNS tell you what approaches to USE (e.g., "Use information_schema.columns query - this works")
- ALWAYS check patterns FIRST before attempting any action
- If a pattern says an approach fails, DO NOT try that approach
- If a pattern provides a working solution, USE that solution

Example Pattern Usage:
- Pattern says: "Query with LIMIT fails with syntax error - avoid LIMIT clause"
- Your response: Skip any queries using LIMIT, use alternative approach
- Pattern says: "Successfully retrieved schema using SELECT column_name FROM information_schema.columns"
- Your response: Use that exact working query pattern

Requirements:
1. First, check if there are LEARNED PATTERNS that guide your approach
2. Think through the reasoning process considering patterns
3. Provide a CONCISE answer based on context
4. Answer must be based ONLY on the provided context
5. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
6. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process (mention if patterns influenced your approach)",
  "answer": "Concise answer in a short phrase"
}}
```

Example with Pattern:
Question: "What's the schema of memory_entries table?"
Context:
[Pattern 1] Confidence: 0.7 | Scope: universal
  Guidance: Query execution failed with 'syntax error at or near LIMIT' - avoid LIMIT clause in this database
[Pattern 2] Confidence: 0.8 | Scope: universal
  Guidance: Successfully retrieved schema using SELECT column_name, data_type FROM information_schema.columns WHERE table_name='memory_entries'

Output:
```json
{{
  "reasoning": "Pattern indicates LIMIT clause fails and provides working query using information_schema.columns",
  "answer": "Use: SELECT column_name, data_type FROM information_schema.columns WHERE table_name='memory_entries'"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""
