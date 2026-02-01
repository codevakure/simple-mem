"""
SimpleMem API Server - Clean REST API

Endpoints:
  # Batch Mode (existing)
  POST /conversation  - Store conversation (with optional agent_id/user_id)
  POST /query         - Get context for a query (with optional filters)
  GET  /memories      - List memories (with optional filters)
  DELETE /memories    - Delete memories (with filters)
  GET  /agents        - List all agents
  GET  /users         - List all users
  
  # Real-Time Session Mode (new)
  POST /session/turn     - Add a single turn to session buffer
  POST /session/process  - Force process session buffer
  GET  /session/status   - Get session buffer status
  DELETE /session        - Clear session buffer

Run: python server.py
Swagger: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import config
from main import SimpleMemSystem
from core.agent_mem import AgentMemory, get_all_agents, get_all_users, delete_agent_memories, delete_user_memories
from utils.logger import get_logger, estimate_tokens

# Initialize logger for server
logger = get_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def sanitize_for_logging(text: str) -> str:
    """
    Sanitize text for Windows console logging by removing/replacing non-ASCII characters.
    Prevents UnicodeEncodeError on Windows terminals.
    """
    try:
        # Try to encode to ASCII, replace errors
        return text.encode('ascii', errors='replace').decode('ascii')
    except Exception:
        # Fallback: filter to printable ASCII
        return ''.join(c if ord(c) < 128 else '?' for c in text)


# ============================================================================
# Request/Response Models
# ============================================================================

class DialogueTurn(BaseModel):
    speaker: str = Field(..., description="'user' or 'agent'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")


class ConversationInput(BaseModel):
    """Store a conversation with optional agent/user filtering."""
    dialogues: List[DialogueTurn] = Field(..., description="Conversation turns")
    agent_id: Optional[str] = Field(None, description="Agent identifier (optional)")
    user_id: Optional[str] = Field(None, description="User identifier (optional)")
    user_name: Optional[str] = Field(None, description="Human-readable user name (optional)")
    async_processing: bool = Field(True, description="Process in background (non-blocking) - default True")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agent_id": "sql_agent",
            "user_id": "user123",
            "user_name": "John Doe",
            "dialogues": [
                {"speaker": "user", "content": "Show me Q4 revenue"},
                {"speaker": "agent", "content": "Based on revenue_table, it's $1M"},
                {"speaker": "user", "content": "Wrong! Use sales_data.quarterly_revenue"},
                {"speaker": "agent", "content": "Got it, the correct value is $2.4M"}
            ]
        }
    })


class QueryInput(BaseModel):
    """Query memories with optional agent/user filtering."""
    query: str = Field(..., description="Query to search memories")
    agent_id: Optional[str] = Field(None, description="Filter by agent (optional)")
    user_id: Optional[str] = Field(None, description="Filter by user (optional)")
    user_name: Optional[str] = Field(None, description="Human-readable user name (optional, for display)")
    max_results: int = Field(10, description="Maximum results")
    deep_analysis: bool = Field(False, description="Enable LLM planning/reflection (slower but better for complex queries)")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What is Q3 revenue for Texas Capital?",
            "agent_id": "sql_agent",
            "user_name": "John Doe",
            "deep_analysis": False
        }
    })


class KnowledgeInput(BaseModel):
    """Directly add knowledge."""
    content: str = Field(..., description="Knowledge to store")
    category: str = Field("general", description="Category: schema, rule, preference")
    agent_id: Optional[str] = Field(None, description="Agent identifier (optional)")
    user_id: Optional[str] = Field(None, description="User identifier (optional)")
    user_name: Optional[str] = Field(None, description="Human-readable user name (optional)")


class MemoryResponse(BaseModel):
    entry_id: str
    content: str
    keywords: List[str] = []
    timestamp: Optional[str] = None
    # Classification fields for Ranger
    memory_type: Optional[str] = None  # correction, feedback, pattern, preference
    scope: Optional[str] = None  # entity, universal
    source_entity: Optional[str] = None  # provenance for patterns
    confidence: Optional[float] = None  # 1.0=correction/feedback, 0.9=pattern, 0.8=preference
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None  # Human-readable user name
    created_at: Optional[str] = None  # Database creation timestamp


class MemoryResultItem(BaseModel):
    """Individual memory result with classification for Ranger."""
    content: str
    memory_type: Optional[str] = None  # correction, feedback, pattern, preference
    scope: Optional[str] = None  # entity, universal
    source_entity: Optional[str] = None  # provenance for patterns
    confidence: Optional[float] = None  # 1.0=correction/feedback, 0.9=pattern, 0.8=preference
    score: Optional[float] = None  # similarity score


class QueryResponse(BaseModel):
    query: str
    context: str  # Formatted string for LLM consumption
    memories: Optional[List[MemoryResultItem]] = None  # Structured results for Ranger
    memory_count: int
    processing_time_ms: float


# ============================================================================
# Session Manager - Per-Agent/User Buffer Management
# ============================================================================

class SessionManager:
    """
    Manages per-(agent_id, user_id) session buffers for real-time memory.
    
    Each unique (agent_id, user_id) pair gets its own:
    - Dialogue buffer
    - Turn counter
    - Last activity timestamp
    
    Features:
    - Auto-process after N turns (configurable)
    - TTL cleanup for idle sessions
    - Thread-safe operations
    """
    
    def __init__(self, auto_process_turns: int = 4, session_ttl_minutes: int = 30):
        """
        Args:
            auto_process_turns: Process buffer after this many turns (default: 4 = 2 exchanges)
            session_ttl_minutes: Clean up inactive sessions after this time
        """
        self.auto_process_turns = auto_process_turns
        self.session_ttl_minutes = session_ttl_minutes
        
        # sessions[session_key] = { 'memory': AgentMemory, 'last_active': datetime }
        self.sessions: Dict[str, dict] = {}
        self._lock = threading.Lock()
        
        # Background processing executor (max 4 concurrent memory extractions)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memory_processor")
        
        # Track in-progress processing jobs
        self._processing_jobs: Dict[str, dict] = {}  # session_key -> { 'started': datetime, 'future': Future }
        self._processing_lock = threading.Lock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _session_key(self, agent_id: str, user_id: str) -> str:
        return f"{agent_id or 'global'}::{user_id or 'global'}"
    
    def get_or_create_session(self, agent_id: str, user_id: str, user_name: str = None) -> 'AgentMemory':
        """Get existing session or create new one."""
        key = self._session_key(agent_id, user_id)
        
        with self._lock:
            if key not in self.sessions:
                self.sessions[key] = {
                    'memory': AgentMemory(agent_id=agent_id, user_id=user_id, user_name=user_name),
                    'last_active': datetime.now(),
                    'turn_count': 0,
                    'user_name': user_name
                }
            elif user_name and not self.sessions[key].get('user_name'):
                # Update user_name if it wasn't set before
                self.sessions[key]['user_name'] = user_name
                self.sessions[key]['memory'].user_name = user_name
                self.sessions[key]['memory'].memory_builder.user_name = user_name
            
            self.sessions[key]['last_active'] = datetime.now()
            return self.sessions[key]
    
    def add_turn(self, agent_id: str, user_id: str, speaker: str, content: str, 
                 timestamp: str = None, process_immediately: bool = False,
                 async_processing: bool = True, user_name: str = None) -> dict:
        """
        Add a single turn to session buffer.
        
        Args:
            async_processing: If True (default), processing happens in background thread.
                            If False, blocks until processing completes.
        
        Returns status including whether auto-processing was triggered.
        """
        session = self.get_or_create_session(agent_id, user_id, user_name)
        memory = session['memory']
        
        # Log incoming turn with full content
        session_key = self._session_key(agent_id, user_id)
        content_tokens = estimate_tokens(content)
        
        # Sanitize content for logging
        safe_content = sanitize_for_logging(content)
        
        logger.info(f"===============================================================")
        logger.info(f"[INCOMING TURN] Session: {session_key}")
        logger.info(f"  Speaker: {speaker}")
        logger.info(f"  Content tokens: ~{content_tokens}")
        logger.info(f"  Timestamp: {timestamp or 'auto'}")
        logger.debug(f"  +- FULL CONTENT -----------------------------------------------")
        # Log full content in chunks for readability
        for i in range(0, len(safe_content), 500):
            chunk = safe_content[i:i+500]
            logger.debug(f"  | {chunk}")
        logger.debug(f"  +---------------------------------------------------------------")
        
        # Check if already processing this session
        with self._processing_lock:
            is_processing = session_key in self._processing_jobs
        
        if is_processing:
            logger.info(f"  [SKIP] Session {session_key} already processing in background")
            # Still add to buffer for next batch
            memory.add_dialogue(speaker, content, timestamp)
            session['turn_count'] += 1
            return {
                'session_key': session_key,
                'turn_count': session['turn_count'],
                'buffer_size': len(memory.memory_builder.dialogue_buffer),
                'processed': False,
                'processing_in_progress': True,
                'message': 'Session is currently processing previous batch'
            }
        
        # Add dialogue to buffer (no auto-process)
        memory.add_dialogue(speaker, content, timestamp)
        session['turn_count'] += 1
        
        result = {
            'session_key': session_key,
            'turn_count': session['turn_count'],
            'buffer_size': len(memory.memory_builder.dialogue_buffer),
            'processed': False
        }
        
        logger.info(f"  Buffer status: {result['buffer_size']} turns (auto-process at {self.auto_process_turns})")
        
        # Check if we should process
        should_process = process_immediately or (session['turn_count'] >= self.auto_process_turns)
        
        if should_process:
            logger.info(f"====================================================================")
            logger.info(f"[PROCESSING BUFFER] Session: {session_key}")
            logger.info(f"  Turns in buffer: {result['buffer_size']}")
            logger.info(f"  Mode: {'ASYNC (background)' if async_processing else 'SYNC (blocking)'}")
            
            # Log all buffered dialogues before processing
            logger.info(f"  +-- BUFFERED DIALOGUES FOR LLM ------------------------------")
            for i, dlg in enumerate(memory.memory_builder.dialogue_buffer):
                dlg_tokens = estimate_tokens(str(dlg))
                logger.info(f"  | [{i+1}] {dlg.speaker}: {str(dlg)[:100]}... (~{dlg_tokens} tokens)")
            logger.info(f"  +-------------------------------------------------------------")
            
            # Snapshot the buffer for processing (so new turns go to fresh buffer)
            dialogues_to_process = list(memory.memory_builder.dialogue_buffer)
            memory.memory_builder.dialogue_buffer.clear()
            session['turn_count'] = 0  # Reset counter
            
            if async_processing:
                # Process in background thread
                result['processed'] = False
                result['processing_started'] = True
                result['buffer_size'] = 0
                result['message'] = 'Processing started in background'
                
                # Submit to executor
                future = self._executor.submit(
                    self._process_dialogues_async,
                    session_key, memory, dialogues_to_process
                )
                
                # Track the job
                with self._processing_lock:
                    self._processing_jobs[session_key] = {
                        'started': datetime.now(),
                        'future': future,
                        'dialogue_count': len(dialogues_to_process)
                    }
                
                logger.info(f"[ASYNC] Processing submitted to background executor")
            else:
                # Synchronous processing (blocking)
                count = self._process_dialogues_sync(memory, dialogues_to_process)
                result['processed'] = True
                result['memories_created'] = count
                result['buffer_size'] = 0
                
                logger.info(f"[SYNC] Processing complete: {count} memory entries")
            
            logger.info(f"====================================================================")
        
        return result
    
    def _process_dialogues_sync(self, memory: AgentMemory, dialogues: list) -> int:
        """Process dialogues synchronously, return count of memories created."""
        # Temporarily add dialogues back to buffer for processing
        memory.memory_builder.dialogue_buffer.extend(dialogues)
        count = memory.finalize()
        return count
    
    def _process_dialogues_async(self, session_key: str, memory: AgentMemory, dialogues: list):
        """Process dialogues in background thread."""
        try:
            logger.info(f"[ASYNC WORKER] Starting processing for {session_key}")
            logger.info(f"  Dialogues: {len(dialogues)}")
            start_time = time.time()
            
            # Add dialogues to buffer and process
            memory.memory_builder.dialogue_buffer.extend(dialogues)
            count = memory.finalize()
            
            duration = time.time() - start_time
            logger.info(f"[ASYNC WORKER] Complete for {session_key}")
            logger.info(f"  Memories created: {count}")
            logger.info(f"  Duration: {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"[ASYNC WORKER] Error processing {session_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            # Remove from processing jobs
            with self._processing_lock:
                if session_key in self._processing_jobs:
                    del self._processing_jobs[session_key]
    
    def get_processing_status(self, agent_id: str, user_id: str) -> dict:
        """Check if a session has background processing in progress."""
        session_key = self._session_key(agent_id, user_id)
        
        with self._processing_lock:
            if session_key not in self._processing_jobs:
                return {'session_key': session_key, 'processing': False}
            
            job = self._processing_jobs[session_key]
            return {
                'session_key': session_key,
                'processing': True,
                'started': job['started'].isoformat(),
                'dialogue_count': job['dialogue_count'],
                'done': job['future'].done()
            }
    
    def process_session(self, agent_id: str, user_id: str) -> dict:
        """Force process all buffered turns for a session."""
        key = self._session_key(agent_id, user_id)
        
        with self._lock:
            if key not in self.sessions:
                return {'error': 'Session not found', 'session_key': key}
            
            session = self.sessions[key]
        
        memory = session['memory']
        buffer_size = len(memory.memory_builder.dialogue_buffer)
        
        if buffer_size == 0:
            return {'session_key': key, 'message': 'Buffer empty, nothing to process'}
        
        count = memory.finalize()
        session['turn_count'] = 0
        
        return {
            'session_key': key,
            'turns_processed': buffer_size,
            'memories_created': count
        }
    
    def get_session_status(self, agent_id: str, user_id: str) -> dict:
        """Get status of a session buffer."""
        key = self._session_key(agent_id, user_id)
        
        with self._lock:
            if key not in self.sessions:
                return {'exists': False, 'session_key': key}
            
            session = self.sessions[key]
            memory = session['memory']
            
            return {
                'exists': True,
                'session_key': key,
                'agent_id': agent_id,
                'user_id': user_id,
                'user_name': session.get('user_name'),
                'turn_count': session['turn_count'],
                'buffer_size': len(memory.memory_builder.dialogue_buffer),
                'last_active': session['last_active'].isoformat(),
                'auto_process_at': self.auto_process_turns
            }
    
    def clear_session(self, agent_id: str, user_id: str) -> dict:
        """Clear session buffer without processing."""
        key = self._session_key(agent_id, user_id)
        
        with self._lock:
            if key in self.sessions:
                del self.sessions[key]
                return {'cleared': True, 'session_key': key}
            return {'cleared': False, 'session_key': key, 'message': 'Session not found'}
    
    def list_sessions(self) -> List[dict]:
        """List all active sessions."""
        with self._lock:
            return [
                {
                    'session_key': key,
                    'turn_count': session['turn_count'],
                    'buffer_size': len(session['memory'].memory_builder.dialogue_buffer),
                    'last_active': session['last_active'].isoformat()
                }
                for key, session in self.sessions.items()
            ]
    
    def _start_cleanup_thread(self):
        """Start background thread to clean up stale sessions."""
        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_stale_sessions()
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def _cleanup_stale_sessions(self):
        """Remove sessions that have been idle too long."""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(minutes=self.session_ttl_minutes)
        
        with self._lock:
            stale_keys = [
                key for key, session in self.sessions.items()
                if session['last_active'] < cutoff
            ]
            
            for key in stale_keys:
                # Process remaining buffer before cleanup
                session = self.sessions[key]
                try:
                    if len(session['memory'].memory_builder.dialogue_buffer) > 0:
                        session['memory'].finalize()
                except Exception as e:
                    print(f"[Session Cleanup] Error finalizing {key}: {e}")
                
                del self.sessions[key]
                print(f"[Session Cleanup] Removed stale session: {key}")


# Global session manager
session_manager: Optional[SessionManager] = None


# ============================================================================
# App Setup
# ============================================================================

simplemem: Optional[SimpleMemSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global simplemem, session_manager
    
    # Get settings from config
    window_size = getattr(config, 'WINDOW_SIZE', 12)
    session_ttl = getattr(config, 'SESSION_TTL_MINUTES', 30)
    
    print("\n" + "=" * 50)
    print("SimpleMem API Starting...")
    print(f"  Vector Store: {getattr(config, 'VECTOR_STORE', 'lancedb')}")
    print(f"  Window Size: {window_size} turns ({window_size // 2} exchanges)")
    print(f"  Session TTL: {session_ttl} minutes")
    print("=" * 50 + "\n")
    simplemem = SimpleMemSystem()
    session_manager = SessionManager(auto_process_turns=window_size, session_ttl_minutes=session_ttl)
    yield
    print("SimpleMem API Shutting down...")


app = FastAPI(
    title="SimpleMem API",
    description="Memory system for AI agents. Store conversations, query context.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    return {"name": "SimpleMem API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["Info"])
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ----------------------------------------------------------------------------
# POST /conversation - Store conversation
# ----------------------------------------------------------------------------
@app.post("/conversation", tags=["Memory"])
async def store_conversation(input: ConversationInput):
    """
    Store a conversation and extract memories.
    
    - No agent_id/user_id: Stored globally
    - agent_id only: Stored for that agent
    - user_id only: Stored for that user  
    - Both: Stored for specific agent AND user
    
    Set async_processing=true for non-blocking (returns immediately).
    """
    start = time.time()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RAW REQUEST LOGGING - See full conversation structure
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("[RAW REQUEST] POST /conversation")
    logger.info(f"  agent_id: {input.agent_id}")
    logger.info(f"  user_id: {input.user_id}")
    logger.info(f"  user_name: {input.user_name}")
    logger.info(f"  async_processing: {input.async_processing}")
    logger.info(f"  dialogue_count: {len(input.dialogues)}")
    
    # Log each dialogue turn with structure analysis
    logger.info(f"  +-- DIALOGUES ------------------------------------------------")
    for i, d in enumerate(input.dialogues):
        has_tool = "[Tool" in d.content
        has_result = "[Tool Result" in d.content or "[Result" in d.content
        content_preview = sanitize_for_logging(d.content[:150])
        logger.info(f"  | [{i+1}] {d.speaker}: {content_preview}...")
        logger.info(f"  |      tokens: ~{estimate_tokens(d.content)}, has_tool: {has_tool}, has_result: {has_result}")
    logger.info(f"  +------------------------------------------------------------")
    logger.info("=" * 70)
    
    memory = AgentMemory(agent_id=input.agent_id, user_id=input.user_id, user_name=input.user_name)
    
    for d in input.dialogues:
        memory.add_dialogue(d.speaker, d.content, d.timestamp)
    
    if input.async_processing:
        memory.finalize_async()
        return {
            "status": "queued",
            "agent_id": input.agent_id,
            "user_id": input.user_id,
            "dialogues": len(input.dialogues),
            "message": "Processing in background"
        }
    else:
        count = memory.finalize()
        elapsed = (time.time() - start) * 1000
        return {
            "status": "processed",
            "agent_id": input.agent_id,
            "user_id": input.user_id,
            "dialogues": len(input.dialogues),
            "memories_created": count,
            "processing_time_ms": round(elapsed, 2)
        }


# ----------------------------------------------------------------------------
# POST /conversations/batch - Batch process multiple conversations
# ----------------------------------------------------------------------------
class BatchConversationItem(BaseModel):
    """Single conversation in a batch."""
    dialogues: List[DialogueTurn]
    agent_id: Optional[str] = None
    user_id: Optional[str] = None


class BatchConversationInput(BaseModel):
    """Batch of conversations to process."""
    conversations: List[BatchConversationItem] = Field(..., description="List of conversations")
    async_processing: bool = Field(True, description="Process all in background")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "conversations": [
                {
                    "agent_id": "sql_agent",
                    "user_id": "user1",
                    "dialogues": [
                        {"speaker": "user", "content": "Query 1"},
                        {"speaker": "agent", "content": "Response 1"}
                    ]
                },
                {
                    "agent_id": "sql_agent", 
                    "user_id": "user2",
                    "dialogues": [
                        {"speaker": "user", "content": "Query 2"},
                        {"speaker": "agent", "content": "Response 2"}
                    ]
                }
            ],
            "async_processing": True
        }
    })


@app.post("/conversations/batch", tags=["Memory"])
async def store_conversations_batch(input: BatchConversationInput):
    """
    Batch process multiple conversations at once.
    
    Useful for bulk imports or processing conversations from multiple users.
    Default is async (non-blocking) for performance.
    """
    import threading
    
    results = []
    
    def process_one(conv):
        memory = AgentMemory(agent_id=conv.agent_id, user_id=conv.user_id)
        for d in conv.dialogues:
            memory.add_dialogue(d.speaker, d.content, d.timestamp)
        return memory.finalize()
    
    if input.async_processing:
        # Process all in background threads
        def background_batch():
            for conv in input.conversations:
                try:
                    process_one(conv)
                except Exception as e:
                    print(f"[Batch] Error processing conversation: {e}")
        
        thread = threading.Thread(target=background_batch, daemon=True)
        thread.start()
        
        return {
            "status": "queued",
            "conversations": len(input.conversations),
            "message": "All conversations processing in background"
        }
    else:
        # Process synchronously
        start = time.time()
        total_memories = 0
        for conv in input.conversations:
            count = process_one(conv)
            total_memories += count
            results.append({
                "agent_id": conv.agent_id,
                "user_id": conv.user_id,
                "memories_created": count
            })
        elapsed = (time.time() - start) * 1000
        
        return {
            "status": "processed",
            "conversations": len(input.conversations),
            "total_memories_created": total_memories,
            "details": results,
            "processing_time_ms": round(elapsed, 2)
        }


# ----------------------------------------------------------------------------
# POST /query - Get context for a query
# ----------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["Memory"])
async def query_memories(input: QueryInput):
    """
    Get relevant context for a query.
    
    **Retrieval Modes:**
    - `deep_analysis=false` (default): Fast vector search (~150-200ms)
    - `deep_analysis=true`: LLM planning + reflection (~2-3s, better for complex queries)
    
    **Filtering:**
    - No filters: Searches ALL memories
    - agent_id only: Searches that agent's memories
    - user_id only: Searches that user's memories
    - Both: Searches specific agent AND user memories
    
    **Response includes:**
    - `context`: Formatted string for LLM consumption
    - `memories`: Structured list with classification (memory_type, scope, confidence)
    
    **Memory Classification (for Ranger):**
    - `memory_type`: 'factual' (tool result), 'correction' (user corrected), 'pattern' (derived)
    - `scope`: 'entity' (specific to one entity), 'universal' (applies broadly)
    - `confidence`: 1.0 (user correction), 0.8 (tool result), 0.6 (pattern)
    """
    start = time.time()
    
    # Log incoming query
    logger.info("=" * 70)
    logger.info("[QUERY] POST /query - Memory retrieval request")
    logger.info(f"  agent_id: {input.agent_id}")
    logger.info(f"  user_id: {input.user_id}")
    logger.info(f"  query: {input.query[:200]}{'...' if len(input.query) > 200 else ''}")
    logger.info(f"  max_results: {input.max_results}")
    logger.info(f"  deep_analysis: {input.deep_analysis}")
    
    memory = AgentMemory(
        agent_id=input.agent_id, 
        user_id=input.user_id,
        enable_deep_analysis=input.deep_analysis
    )
    context = memory.get_context_string(input.query, input.max_results)
    memories = memory.get_context(input.query, input.max_results)
    
    # Build structured results for Ranger
    memory_items = [
        MemoryResultItem(
            content=m.lossless_restatement,
            memory_type=m.memory_type,
            scope=m.scope,
            source_entity=m.source_entity,
            confidence=m.confidence
        )
        for m in memories
    ]
    
    elapsed = (time.time() - start) * 1000
    
    # Log response
    logger.info(f"[QUERY] Response: {len(memory_items)} memories found in {elapsed:.0f}ms")
    for i, m in enumerate(memory_items[:5], 1):
        logger.info(f"  [{i}] [{m.memory_type}, conf={m.confidence}] {m.content[:80]}...")
    if len(memory_items) > 5:
        logger.info(f"  ... and {len(memory_items) - 5} more")
    logger.info("=" * 70)
    
    return QueryResponse(
        query=input.query,
        context=context,
        memories=memory_items,
        memory_count=len(memories),
        processing_time_ms=round(elapsed, 2)
    )


# ----------------------------------------------------------------------------
# POST /knowledge - Add knowledge directly
# ----------------------------------------------------------------------------
@app.post("/knowledge", tags=["Memory"])
async def add_knowledge(input: KnowledgeInput):
    """
    Directly add knowledge without conversation flow.
    
    Use for: schemas, rules, preferences.
    """
    memory = AgentMemory(agent_id=input.agent_id, user_id=input.user_id)
    entry_id = memory.add_knowledge(input.content, input.category)
    
    return {
        "status": "stored",
        "entry_id": entry_id,
        "agent_id": input.agent_id,
        "user_id": input.user_id
    }


# ----------------------------------------------------------------------------
# GET /memories - List memories with filters and pagination
# ----------------------------------------------------------------------------
@app.get("/memories", tags=["Memory"])
async def get_memories(
    agent_id: str = Query(None, description="Filter by agent"),
    user_id: str = Query(None, description="Filter by user"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    search: str = Query(None, description="SEMANTIC search using vector similarity (not text search)"),
    scope: str = Query(None, description="Filter by scope: entity or universal"),
    memory_type: str = Query(None, description="Filter by type: correction, feedback, pattern, or preference"),
    min_confidence: float = Query(None, ge=0, le=1, description="Minimum confidence score")
):
    """
    Get memories with optional filters and pagination.
    
    - agent_id/user_id: Scope to specific agent/user
    - page/limit: Server-side pagination
    - search: SEMANTIC search using vector embeddings (cosine similarity)
    - scope: Filter by entity/universal
    - memory_type: Filter by correction/feedback/pattern/preference
    - min_confidence: Minimum confidence threshold
    """
    # Get filtered and paginated memories from vector store
    result = simplemem.vector_store.get_entries_paginated(
        agent_id=agent_id,
        user_id=user_id,
        page=page,
        limit=limit,
        search=search,
        scope=scope,
        memory_type=memory_type,
        min_confidence=min_confidence
    )
    
    memories_response = [
        MemoryResponse(
            entry_id=m.entry_id,
            content=m.lossless_restatement,
            keywords=m.keywords or [],
            timestamp=m.timestamp,
            memory_type=getattr(m, 'memory_type', None),
            scope=getattr(m, 'scope', None),
            source_entity=getattr(m, 'source_entity', None),
            confidence=getattr(m, 'confidence', None),
            agent_id=getattr(m, 'agent_id', None),
            user_id=getattr(m, 'user_id', None),
            user_name=getattr(m, 'user_name', None),
            created_at=getattr(m, 'created_at', None)
        )
        for m in result['memories']
    ]
    
    return {
        "memories": memories_response,
        "total": result['total'],
        "page": result['page'],
        "limit": result['limit'],
        "totalPages": result['totalPages'],
        "hasNextPage": result['hasNextPage'],
        "hasPrevPage": result['hasPrevPage']
    }


# ----------------------------------------------------------------------------
# DELETE /memories - Delete memories with filters
# ----------------------------------------------------------------------------
@app.delete("/memories", tags=["Memory"])
async def delete_memories(
    agent_id: str = Query(None, description="Delete by agent"),
    user_id: str = Query(None, description="Delete by user"),
    all: bool = Query(False, description="Delete ALL memories (dangerous!)")
):
    """
    Delete memories with filters.
    
    - agent_id only: Delete that agent's memories
    - user_id only: Delete that user's memories
    - Both: Delete specific agent AND user memories
    - all=true: Delete everything (requires explicit flag)
    """
    if all:
        simplemem.vector_store.clear()
        return {"status": "cleared", "message": "All memories deleted"}
    
    if not agent_id and not user_id:
        raise HTTPException(400, "Specify agent_id, user_id, or all=true")
    
    if agent_id and not user_id:
        count = delete_agent_memories(agent_id)
    else:
        count = delete_user_memories(user_id, agent_id)
    
    return {
        "status": "deleted",
        "agent_id": agent_id,
        "user_id": user_id,
        "count": count
    }


# ----------------------------------------------------------------------------
# GET /agents - List all agents
# ----------------------------------------------------------------------------
@app.get("/agents", tags=["Admin"])
async def list_agents():
    """Get list of all agents with memory counts."""
    return get_all_agents()


# ----------------------------------------------------------------------------
# GET /users - List all users
# ----------------------------------------------------------------------------
@app.get("/users", tags=["Admin"])
async def list_users(agent_id: str = Query(None, description="Filter by agent")):
    """Get list of all users with memory counts."""
    return get_all_users(agent_id)


# ============================================================================
# Real-Time Session Endpoints
# ============================================================================

class TurnInput(BaseModel):
    """Single turn to add to session buffer."""
    speaker: str = Field(..., description="'user' or 'agent'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field(None, description="Human-readable user name")
    process_now: bool = Field(False, description="Force immediate processing after this turn")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agent_id": "sql_agent",
            "user_id": "user123",
            "user_name": "John Doe",
            "speaker": "user",
            "content": "Show me Q4 revenue",
            "process_now": False
        }
    })


class SessionKey(BaseModel):
    """Session identifier."""
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    user_name: Optional[str] = Field(None, description="Human-readable user name")


@app.post("/session/turn", tags=["Real-Time Session"])
async def add_session_turn(input: TurnInput):
    """
    Add a single turn to the session buffer (real-time mode).
    
    **How it works:**
    - Each (agent_id, user_id) pair has its own buffer
    - Buffer auto-processes after 4 turns (2 user+agent exchanges)
    - Set `process_now=true` to force immediate processing
    - Idle sessions auto-cleanup after 30 minutes
    
    **Example flow:**
    ```
    POST /session/turn {speaker: "user", content: "Get Q4 revenue"}     → buffer=1
    POST /session/turn {speaker: "agent", content: "Revenue is $1M"}    → buffer=2
    POST /session/turn {speaker: "user", content: "Show in billions"}   → buffer=3
    POST /session/turn {speaker: "agent", content: "OK, noted"}         → buffer=4 → AUTO-PROCESS!
    ```
    """
    start = time.time()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RAW REQUEST LOGGING - See exactly what Ranger sends
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("[RAW REQUEST] POST /session/turn")
    logger.info(f"  agent_id: {input.agent_id}")
    logger.info(f"  user_id: {input.user_id}")
    logger.info(f"  speaker: {input.speaker}")
    logger.info(f"  timestamp: {input.timestamp}")
    logger.info(f"  process_now: {input.process_now}")
    logger.info(f"  content_length: {len(input.content)} chars")
    logger.info(f"  content_tokens: ~{estimate_tokens(input.content)}")
    
    # Log content structure analysis
    has_tool_call = "[Tool Call:" in input.content or "[Tool:" in input.content
    has_tool_result = "[Tool Result:" in input.content
    has_reasoning = "[Reasoning]" in input.content or "[Thinking]" in input.content
    has_agent_response = "[Agent Response]" in input.content
    
    logger.info(f"  +-- CONTENT STRUCTURE --------------------------------------")
    logger.info(f"  | Has Tool Calls:    {has_tool_call}")
    logger.info(f"  | Has Tool Results:  {has_tool_result}")
    logger.info(f"  | Has Reasoning:     {has_reasoning}")
    logger.info(f"  | Has Agent Response:{has_agent_response}")
    logger.info(f"  +------------------------------------------------------------")
    
    # Log full content with clear boundaries
    safe_content = sanitize_for_logging(input.content)
    logger.debug(f"  +-- FULL RAW CONTENT --------------------------------------")
    for i, line in enumerate(safe_content.split('\n')):
        if i < 100:  # Limit to 100 lines for readability
            logger.debug(f"  | {line[:200]}")
        elif i == 100:
            logger.debug(f"  | ... (truncated, {len(safe_content.split(chr(10)))} total lines)")
            break
    logger.debug(f"  +------------------------------------------------------------")
    logger.info("=" * 70)
    
    result = session_manager.add_turn(
        agent_id=input.agent_id,
        user_id=input.user_id,
        user_name=input.user_name,
        speaker=input.speaker,
        content=input.content,
        timestamp=input.timestamp,
        process_immediately=input.process_now
    )
    
    result['processing_time_ms'] = round((time.time() - start) * 1000, 2)
    return result


@app.post("/session/process", tags=["Real-Time Session"])
async def process_session(input: SessionKey):
    """
    Force process all buffered turns for a session.
    
    Use this at the end of a conversation or when you want to ensure
    all pending turns are converted to memories.
    """
    start = time.time()
    result = session_manager.process_session(input.agent_id, input.user_id)
    result['processing_time_ms'] = round((time.time() - start) * 1000, 2)
    return result


@app.get("/session/status", tags=["Real-Time Session"])
async def get_session_status(
    agent_id: str = Query(None, description="Agent identifier"),
    user_id: str = Query(None, description="User identifier")
):
    """Get status of a session buffer."""
    return session_manager.get_session_status(agent_id, user_id)


@app.delete("/session", tags=["Real-Time Session"])
async def clear_session(
    agent_id: str = Query(None, description="Agent identifier"),
    user_id: str = Query(None, description="User identifier")
):
    """Clear a session buffer without processing the pending turns."""
    return session_manager.clear_session(agent_id, user_id)


@app.get("/sessions", tags=["Real-Time Session"])
async def list_sessions():
    """List all active sessions with buffer status."""
    return {
        "sessions": session_manager.list_sessions(),
        "total": len(session_manager.sessions)
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="Port to run on (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    print("=" * 50)
    print("  SimpleMem API")
    print(f"  Swagger: http://localhost:{args.port}/docs")
    print("=" * 50)
    
    uvicorn.run("server:app", host=args.host, port=args.port, reload=True)
