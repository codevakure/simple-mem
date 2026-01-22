"""
Vector Store - Structured Multi-View Indexing Implementation (Section 3.2)

Paper Reference: Section 3.2 - Structured Indexing
Implements the three structured indexing dimensions:
- Semantic Layer: Dense vectors v_k ∈ ℝ^d (embedding-based similarity)
- Lexical Layer: Full-text search with Tantivy FTS (LanceDB) or tsvector (pgvector)
- Symbolic Layer: Metadata R_k = {(key, val)} (structured filtering via SQL)

Supports multiple backends based on config.VECTOR_STORE:
- 'lancedb': Local LanceDB with Tantivy FTS (default)
- 'pgvector': PostgreSQL with pgvector extension
"""
from typing import List, Optional, Dict, Any
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
import config
import os


def get_vector_store(
    db_path: str = None,
    embedding_model: EmbeddingModel = None,
    table_name: str = None,
    **kwargs
):
    """
    Factory function to get the appropriate vector store based on config.
    
    Returns:
        VectorStore instance (LanceDB or PgVector based on config)
    """
    vector_store_type = getattr(config, 'VECTOR_STORE', 'lancedb').lower()
    
    if vector_store_type == 'pgvector':
        return PgVectorStore(
            embedding_model=embedding_model,
            table_name=table_name,
            **kwargs
        )
    else:
        return LanceDBVectorStore(
            db_path=db_path,
            embedding_model=embedding_model,
            table_name=table_name,
            **kwargs
        )


# Alias for backward compatibility
class VectorStore:
    """
    Backward-compatible VectorStore class.
    Automatically selects the right backend based on config.VECTOR_STORE.
    """
    def __new__(cls, *args, **kwargs):
        vector_store_type = getattr(config, 'VECTOR_STORE', 'lancedb').lower()
        
        if vector_store_type == 'pgvector':
            return PgVectorStore(*args, **kwargs)
        else:
            return LanceDBVectorStore(*args, **kwargs)


class LanceDBVectorStore:
    """
    LanceDB Backend - Structured Multi-View Indexing

    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers:
    1. Semantic Layer: Dense embedding vectors for conceptual similarity
    2. Lexical Layer: Full-text search via Tantivy FTS index
    3. Symbolic Layer: SQL-based metadata filtering with DataFusion
    """

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        storage_options: Optional[Dict[str, Any]] = None
    ):
        import lancedb
        import pyarrow as pa
        
        self.lancedb = lancedb
        self.pa = pa
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.table = None
        self._fts_initialized = False

        # Detect if using cloud storage (GCS, S3, Azure)
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self):
        """Initialize table schema and FTS index."""
        schema = self.pa.schema([
            self.pa.field("entry_id", self.pa.string()),
            self.pa.field("lossless_restatement", self.pa.string()),
            self.pa.field("keywords", self.pa.list_(self.pa.string())),
            self.pa.field("timestamp", self.pa.string()),
            self.pa.field("location", self.pa.string()),
            self.pa.field("persons", self.pa.list_(self.pa.string())),
            self.pa.field("entities", self.pa.list_(self.pa.string())),
            self.pa.field("topic", self.pa.string()),
            self.pa.field("vector", self.pa.list_(self.pa.float32(), self.embedding_model.dimension))
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def _init_fts_index(self):
        """Initialize Full-Text Search index on lossless_restatement column."""
        if self._fts_initialized:
            return

        try:
            if self._is_cloud_storage:
                # Use native FTS for cloud storage (Tantivy only works with local filesystem)
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=False,
                    replace=True
                )
                print("FTS index created (native mode for cloud storage)")
            else:
                # Use Tantivy FTS for local storage (better performance)
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True
                )
                print("FTS index created (Tantivy mode)")
            self._fts_initialized = True
        except Exception as e:
            print(f"FTS index creation skipped: {e}")

    def _results_to_entries(self, results: List[dict]) -> List[MemoryEntry]:
        """Convert LanceDB results to MemoryEntry objects."""
        entries = []
        for r in results:
            try:
                entries.append(MemoryEntry(
                    entry_id=r["entry_id"],
                    lossless_restatement=r["lossless_restatement"],
                    keywords=list(r.get("keywords") or []),
                    timestamp=r.get("timestamp") or None,
                    location=r.get("location") or None,
                    persons=list(r.get("persons") or []),
                    entities=list(r.get("entities") or []),
                    topic=r.get("topic") or None
                ))
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
                continue
        return entries

    def add_entries(self, entries: List[MemoryEntry]):
        """Batch add memory entries."""
        if not entries:
            return

        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        data = []
        for entry, vector in zip(entries, vectors):
            data.append({
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons,
                "entities": entry.entities,
                "topic": entry.topic or "",
                "vector": vector.tolist()
            })

        self.table.add(data)
        print(f"Added {len(entries)} memory entries")

        # Initialize FTS index after first data insertion
        if not self._fts_initialized:
            self._init_fts_index()

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        Semantic Layer Search - Dense vector similarity.

        Paper Reference: Section 3.1
        Retrieves based on v_k = E_dense(S_k) where S_k is the lossless restatement.
        """
        try:
            if self.table.count_rows() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = self.table.search(query_vector.tolist()).limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """
        Lexical Layer Search - Full-text search via Tantivy FTS.

        Paper Reference: Section 3.1
        Retrieves based on BM25 text matching using LanceDB native FTS.
        """
        try:
            if not keywords or self.table.count_rows() == 0:
                return []

            # LanceDB auto-detects string input as FTS query when FTS index exists
            query = " ".join(keywords)
            results = self.table.search(query).limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - SQL-based metadata filtering.

        Paper Reference: Section 3.1
        Retrieves based on R_k = {(key, val)} for structured constraints.
        Uses DataFusion SQL expressions with array_has_any for list columns.
        """
        try:
            if self.table.count_rows() == 0:
                return []

            if not any([persons, timestamp_range, location, entities]):
                return []

            conditions = []

            if persons:
                values = ", ".join([f"'{p}'" for p in persons])
                conditions.append(f"array_has_any(persons, make_array({values}))")

            if location:
                safe_location = location.replace("'", "''")
                conditions.append(f"location LIKE '%{safe_location}%'")

            if entities:
                values = ", ".join([f"'{e}'" for e in entities])
                conditions.append(f"array_has_any(entities, make_array({values}))")

            if timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append(f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'")

            where_clause = " AND ".join(conditions)
            query = self.table.search().where(where_clause, prefilter=True)

            if top_k:
                query = query.limit(top_k)

            results = query.to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during structured search: {e}")
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        results = self.table.to_arrow().to_pylist()
        return self._results_to_entries(results)

    def optimize(self):
        """Optimize table after bulk insertions for better query performance."""
        self.table.optimize()
        print("Table optimized")

    def clear(self):
        """Clear all data and reinitialize table."""
        self.db.drop_table(self.table_name)
        self._fts_initialized = False
        self._init_table()
        print("Database cleared")


class PgVectorStore:
    """
    PostgreSQL + pgvector Backend - Structured Multi-View Indexing

    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers:
    1. Semantic Layer: Dense embedding vectors with pgvector for cosine similarity
    2. Lexical Layer: Full-text search via PostgreSQL tsvector/tsquery
    3. Symbolic Layer: SQL-based metadata filtering with PostgreSQL arrays
    
    Uses connection pooling for concurrent access (2000+ users).
    """
    
    # Class-level connection pool (shared across all instances)
    _pool = None
    _pool_lock = None

    def __init__(
        self,
        db_path: str = None,  # Not used, for API compatibility
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        pool_min: int = 5,
        pool_max: int = 50,
        use_hnsw: bool = True,  # HNSW is faster than IVFFlat, no training needed
        **kwargs
    ):
        import psycopg2
        from psycopg2 import pool
        from psycopg2.extras import execute_values
        from pgvector.psycopg2 import register_vector
        import threading
        
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.use_hnsw = use_hnsw
        self._execute_values = execute_values  # Store for batch inserts
        self.database_url = getattr(config, 'DATABASE_URL', None)
        
        if not self.database_url:
            host = getattr(config, 'POSTGRES_HOST', 'localhost')
            port = getattr(config, 'POSTGRES_PORT', 5432)
            user = getattr(config, 'POSTGRES_USER', 'simplemem')
            password = getattr(config, 'POSTGRES_PASSWORD', 'simplemem_dev_password')
            db = getattr(config, 'POSTGRES_DB', 'simplemem_db')
            self.database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        
        # Initialize class-level pool lock if needed
        if PgVectorStore._pool_lock is None:
            PgVectorStore._pool_lock = threading.Lock()
        
        # Create connection pool (thread-safe, shared across instances)
        with PgVectorStore._pool_lock:
            if PgVectorStore._pool is None:
                PgVectorStore._pool = pool.ThreadedConnectionPool(
                    minconn=pool_min,
                    maxconn=pool_max,
                    dsn=self.database_url
                )
                print(f"Created connection pool: min={pool_min}, max={pool_max}")
        
        # Store register_vector for new connections
        self._register_vector = register_vector
        
        # Get a connection to initialize and register vector type
        conn = self._get_conn()
        try:
            register_vector(conn)
            self._init_table_with_conn(conn)
        finally:
            self._put_conn(conn)
        
        print(f"PgVectorStore ready: {self.table_name} (pooled, {'HNSW' if self.use_hnsw else 'IVFFlat'})")
    
    def _get_conn(self):
        """Get connection from pool with vector type registered."""
        conn = PgVectorStore._pool.getconn()
        conn.autocommit = False
        # Ensure vector type is registered for this connection
        try:
            self._register_vector(conn)
        except:
            pass  # Already registered
        return conn
    
    def _put_conn(self, conn):
        """Return connection to pool."""
        try:
            conn.rollback()  # Clear any uncommitted state
        except:
            pass
        PgVectorStore._pool.putconn(conn)
    
    def _init_table_with_conn(self, conn):
        """Initialize table schema with pgvector and indexes."""
        with conn.cursor() as cur:
            # Ensure pgvector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table if not exists
            dimension = self.embedding_model.dimension
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    entry_id VARCHAR(255) UNIQUE NOT NULL,
                    lossless_restatement TEXT NOT NULL,
                    keywords TEXT[] DEFAULT '{{}}',
                    timestamp VARCHAR(255),
                    location VARCHAR(512),
                    persons TEXT[] DEFAULT '{{}}',
                    entities TEXT[] DEFAULT '{{}}',
                    topic VARCHAR(512),
                    agent_id VARCHAR(255),
                    user_id VARCHAR(255),
                    vector vector({dimension}),
                    search_vector tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', lossless_restatement)
                    ) STORED,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add columns if table already exists (migration)
            cur.execute(f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name = '{self.table_name}' AND column_name = 'agent_id') THEN
                        ALTER TABLE {self.table_name} ADD COLUMN agent_id VARCHAR(255);
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name = '{self.table_name}' AND column_name = 'user_id') THEN
                        ALTER TABLE {self.table_name} ADD COLUMN user_id VARCHAR(255);
                    END IF;
                END $$;
            """)
            
            # Create indexes for efficient search
            # HNSW: Faster queries, no training needed, better recall
            # IVFFlat: Faster builds, needs more tuning (lists, probes)
            if self.use_hnsw:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_vector 
                    ON {self.table_name} 
                    USING hnsw (vector vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            else:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_vector 
                    ON {self.table_name} 
                    USING ivfflat (vector vector_cosine_ops)
                    WITH (lists = 100)
                """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_search 
                ON {self.table_name} 
                USING gin(search_vector)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_persons 
                ON {self.table_name} 
                USING gin(persons)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_entities 
                ON {self.table_name} 
                USING gin(entities)
            """)
            
            # Indexes for agent_id and user_id filtering
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_agent_id 
                ON {self.table_name} (agent_id)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_user_id 
                ON {self.table_name} (user_id)
            """)
            
            # Composite index for filtered vector searches (multi-tenant optimization)
            # Speeds up queries like: WHERE agent_id = X ORDER BY vector <=> query
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_agent_user 
                ON {self.table_name} (agent_id, user_id)
            """)
            
            conn.commit()
            print(f"Initialized pgvector table: {self.table_name}")

    def _results_to_entries(self, results: List[tuple], include_ids: bool = False) -> List[MemoryEntry]:
        """Convert PostgreSQL results to MemoryEntry objects.
        
        Args:
            results: Query results
            include_ids: If True, expects agent_id, user_id at positions 8, 9
        """
        entries = []
        for r in results:
            try:
                entry = MemoryEntry(
                    entry_id=r[0],
                    lossless_restatement=r[1],
                    keywords=list(r[2]) if r[2] else [],
                    timestamp=r[3] or None,
                    location=r[4] or None,
                    persons=list(r[5]) if r[5] else [],
                    entities=list(r[6]) if r[6] else [],
                    topic=r[7] or None,
                    agent_id=r[8] if include_ids and len(r) > 8 else None,
                    user_id=r[9] if include_ids and len(r) > 9 else None
                )
                entries.append(entry)
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
                continue
        return entries

    def _check_semantic_duplicate(
        self, 
        vector: list, 
        agent_id: str = None, 
        user_id: str = None,
        similarity_threshold: float = 0.95
    ) -> Optional[str]:
        """
        Check if a semantically similar entry already exists.
        
        Returns entry_id of existing duplicate if found, None otherwise.
        Cosine similarity > threshold means content is essentially the same.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Cosine distance: 0 = identical, 2 = opposite
                # Similarity = 1 - distance, so threshold 0.95 = distance < 0.05
                distance_threshold = 1 - similarity_threshold
                
                # Build query with proper parameter ordering
                if agent_id and user_id:
                    cur.execute(f"""
                        SELECT entry_id, 1 - (vector <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE agent_id = %s AND user_id = %s
                        AND (vector <=> %s::vector) < %s
                        ORDER BY vector <=> %s::vector
                        LIMIT 1
                    """, [vector, agent_id, user_id, vector, distance_threshold, vector])
                elif agent_id:
                    cur.execute(f"""
                        SELECT entry_id, 1 - (vector <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE agent_id = %s
                        AND (vector <=> %s::vector) < %s
                        ORDER BY vector <=> %s::vector
                        LIMIT 1
                    """, [vector, agent_id, vector, distance_threshold, vector])
                elif user_id:
                    cur.execute(f"""
                        SELECT entry_id, 1 - (vector <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE user_id = %s
                        AND (vector <=> %s::vector) < %s
                        ORDER BY vector <=> %s::vector
                        LIMIT 1
                    """, [vector, user_id, vector, distance_threshold, vector])
                else:
                    cur.execute(f"""
                        SELECT entry_id, 1 - (vector <=> %s::vector) as similarity
                        FROM {self.table_name}
                        WHERE (vector <=> %s::vector) < %s
                        ORDER BY vector <=> %s::vector
                        LIMIT 1
                    """, [vector, vector, distance_threshold, vector])
                
                result = cur.fetchone()
                if result:
                    return result[0]  # Return existing entry_id
                return None
        finally:
            self._put_conn(conn)

    def add_entries(
        self, 
        entries: List[MemoryEntry], 
        deduplicate: bool = True,
        similarity_threshold: float = 0.95
    ):
        """
        Batch add memory entries with optional semantic deduplication.
        
        Args:
            entries: List of MemoryEntry objects
            deduplicate: If True, skip entries that are semantically similar to existing ones
            similarity_threshold: Cosine similarity threshold (0.95 = 95% similar)
        """
        if not entries:
            return

        # Batch encode all documents at once (like LanceDB)
        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        # Filter out semantic duplicates if enabled
        entries_to_add = []
        vectors_to_add = []
        skipped_count = 0
        
        if deduplicate:
            for entry, vector in zip(entries, vectors):
                existing_id = self._check_semantic_duplicate(
                    vector.tolist(),
                    entry.agent_id,
                    entry.user_id,
                    similarity_threshold
                )
                if existing_id:
                    skipped_count += 1
                else:
                    entries_to_add.append(entry)
                    vectors_to_add.append(vector)
        else:
            entries_to_add = entries
            vectors_to_add = vectors
        
        if not entries_to_add:
            if skipped_count > 0:
                print(f"Skipped {skipped_count} duplicate entries (semantic similarity > {similarity_threshold})")
            return

        # Prepare batch data
        data = [
            (
                entry.entry_id,
                entry.lossless_restatement,
                entry.keywords,
                entry.timestamp or "",
                entry.location or "",
                entry.persons,
                entry.entities,
                entry.topic or "",
                entry.agent_id,
                entry.user_id,
                vector.tolist()
            )
            for entry, vector in zip(entries_to_add, vectors_to_add)
        ]

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Use execute_values for 10-100x faster batch inserts
                self._execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} 
                    (entry_id, lossless_restatement, keywords, timestamp, location, 
                     persons, entities, topic, agent_id, user_id, vector)
                    VALUES %s
                    ON CONFLICT (entry_id) DO UPDATE SET
                        lossless_restatement = EXCLUDED.lossless_restatement,
                        keywords = EXCLUDED.keywords,
                        timestamp = EXCLUDED.timestamp,
                        location = EXCLUDED.location,
                        persons = EXCLUDED.persons,
                        entities = EXCLUDED.entities,
                        topic = EXCLUDED.topic,
                        agent_id = EXCLUDED.agent_id,
                        user_id = EXCLUDED.user_id,
                        vector = EXCLUDED.vector
                    """,
                    data,
                    page_size=100  # Batch size for network efficiency
                )
                conn.commit()
            msg = f"Added {len(entries_to_add)} memory entries"
            if skipped_count > 0:
                msg += f" (skipped {skipped_count} duplicates)"
            print(msg)
        finally:
            self._put_conn(conn)

    def semantic_search(self, query: str, top_k: int = 5, 
                        agent_id: str = None, user_id: str = None) -> List[MemoryEntry]:
        """
        Semantic Layer Search - Dense vector similarity with pgvector.
        Uses cosine distance for similarity matching.
        
        Note: Like LanceDB, returns top_k results without similarity threshold.
        
        Args:
            query: Search query
            top_k: Number of results
            agent_id: Filter by agent (optional)
            user_id: Filter by user (optional)
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                if cur.fetchone()[0] == 0:
                    return []
                
                # Set index-specific parameters for better recall
                if self.use_hnsw:
                    # HNSW: ef_search controls accuracy/speed tradeoff (default 40)
                    cur.execute("SET hnsw.ef_search = 100")
                else:
                    # IVFFlat: probes controls how many lists to search (default 1)
                    cur.execute("SET ivfflat.probes = 10")

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            
            # Build WHERE clause for filtering (like LanceDB prefilter)
            conditions = []
            where_params = []
            
            if agent_id:
                conditions.append("agent_id = %s")
                where_params.append(agent_id)
            if user_id:
                conditions.append("user_id = %s")
                where_params.append(user_id)
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            # Params order: WHERE params first, then vector for ORDER BY, then LIMIT
            params = where_params + [query_vector.tolist(), top_k]
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT entry_id, lossless_restatement, keywords, timestamp, 
                           location, persons, entities, topic, agent_id, user_id
                    FROM {self.table_name}
                    {where_clause}
                    ORDER BY vector <=> %s::vector
                    LIMIT %s
                """, params)
                
                results = cur.fetchall()
                return self._results_to_entries(results, include_ids=True)

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []
        finally:
            self._put_conn(conn)

    def keyword_search(self, keywords: List[str], top_k: int = 3,
                       agent_id: str = None, user_id: str = None) -> List[MemoryEntry]:
        """
        Lexical Layer Search - Full-text search via PostgreSQL tsvector/tsquery.
        
        Uses GIN index for fast full-text matching (similar to LanceDB's Tantivy FTS).
        
        Args:
            keywords: List of keywords to search
            top_k: Number of results
            agent_id: Filter by agent (optional)
            user_id: Filter by user (optional)
        """
        if not keywords:
            return []
            
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                if cur.fetchone()[0] == 0:
                    return []

                # Build tsquery from keywords (OR search like LanceDB)
                query_str = " | ".join(keywords)
            
            # Build additional filters
            conditions = ["search_vector @@ plainto_tsquery('english', %s)"]
            params = [query_str, query_str]  # For rank and WHERE
            
            if agent_id:
                conditions.append("agent_id = %s")
                params.append(agent_id)
            if user_id:
                conditions.append("user_id = %s")
                params.append(user_id)
            
            where_clause = " AND ".join(conditions)
            params.append(top_k)
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT entry_id, lossless_restatement, keywords, timestamp, 
                           location, persons, entities, topic, agent_id, user_id,
                           ts_rank(search_vector, plainto_tsquery('english', %s)) as rank
                    FROM {self.table_name}
                    WHERE {where_clause}
                    ORDER BY rank DESC
                    LIMIT %s
                """, params)
                
                results = cur.fetchall()
                # Remove the rank column from results (last column)
                return self._results_to_entries([r[:10] for r in results], include_ids=True)

        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []
        finally:
            self._put_conn(conn)

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - SQL-based metadata filtering with PostgreSQL arrays.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                if cur.fetchone()[0] == 0:
                    return []

            if not any([persons, timestamp_range, location, entities]):
                return []

            conditions = []
            params = []

            if persons:
                conditions.append("persons && %s")
                params.append(persons)

            if location:
                conditions.append("location ILIKE %s")
                params.append(f"%{location}%")

            if entities:
                conditions.append("entities && %s")
                params.append(entities)

            if timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append("timestamp >= %s AND timestamp <= %s")
                params.extend([start_time, end_time])

            where_clause = " AND ".join(conditions)
            limit_clause = f"LIMIT {top_k}" if top_k else ""
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT entry_id, lossless_restatement, keywords, timestamp, 
                           location, persons, entities, topic
                    FROM {self.table_name}
                    WHERE {where_clause}
                    {limit_clause}
                """, params)
                
                results = cur.fetchall()
                return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during structured search: {e}")
            return []
        finally:
            self._put_conn(conn)

    def get_all_entries(self, agent_id: str = None, user_id: str = None) -> List[MemoryEntry]:
        """Get all memory entries, optionally filtered by agent_id/user_id."""
        conditions = []
        params = []
        
        if agent_id:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT entry_id, lossless_restatement, keywords, timestamp, 
                           location, persons, entities, topic, agent_id, user_id
                    FROM {self.table_name}
                    {where_clause}
                """, params)
                results = cur.fetchall()
                return self._results_to_entries(results, include_ids=True)
        finally:
            self._put_conn(conn)

    def count_rows(self) -> int:
        """Get total row count."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
        finally:
            self._put_conn(conn)

    def optimize(self):
        """Reindex and analyze table for better performance."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"REINDEX TABLE {self.table_name}")
                cur.execute(f"ANALYZE {self.table_name}")
                conn.commit()
            print("Table optimized (reindexed and analyzed)")
        finally:
            self._put_conn(conn)

    def clear(self):
        """Clear all data from table."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
                conn.commit()
            print("Database cleared")
        finally:
            self._put_conn(conn)

    def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of all agents with memory counts."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT agent_id, COUNT(*) as count
                    FROM {self.table_name}
                    WHERE agent_id IS NOT NULL
                    GROUP BY agent_id
                    ORDER BY count DESC
                """)
                results = cur.fetchall()
                return [{"agent_id": r[0], "memory_count": r[1]} for r in results]
        finally:
            self._put_conn(conn)

    def get_users(self, agent_id: str = None) -> List[Dict[str, Any]]:
        """Get list of all users with memory counts, optionally filtered by agent."""
        conditions = ["user_id IS NOT NULL"]
        params = []
        
        if agent_id:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        
        where_clause = " AND ".join(conditions)
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT user_id, COUNT(*) as count
                    FROM {self.table_name}
                    WHERE {where_clause}
                    GROUP BY user_id
                    ORDER BY count DESC
                """, params)
                results = cur.fetchall()
                return [{"user_id": r[0], "memory_count": r[1]} for r in results]
        finally:
            self._put_conn(conn)

    def delete_by_agent(self, agent_id: str) -> int:
        """Delete all memories for a specific agent."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE agent_id = %s
                """, (agent_id,))
                count = cur.rowcount
                conn.commit()
            return count
        finally:
            self._put_conn(conn)

    def delete_by_user(self, user_id: str, agent_id: str = None) -> int:
        """Delete all memories for a specific user, optionally within an agent."""
        conditions = ["user_id = %s"]
        params = [user_id]
        
        if agent_id:
            conditions.append("agent_id = %s")
            params.append(agent_id)
        
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE {' AND '.join(conditions)}
                """, params)
                count = cur.rowcount
                conn.commit()
            return count
        finally:
            self._put_conn(conn)

    def __del__(self):
        """Cleanup - pool is shared so we don't close it here."""
        pass  # Pool is class-level, managed separately
    
    @classmethod
    def close_pool(cls):
        """Close the connection pool (call on shutdown)."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            print("Connection pool closed")
