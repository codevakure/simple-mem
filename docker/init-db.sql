-- SimpleMem PostgreSQL Initialization Script
-- This script runs automatically when the PostgreSQL container starts for the first time
-- NOTE: The Python PgVectorStore._init_table() will also create these if missing

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create memory_entries table
-- The Python code will auto-create this table if it doesn't exist,
-- but having it here ensures the schema is ready immediately
CREATE TABLE IF NOT EXISTS memory_entries (
    id SERIAL PRIMARY KEY,
    entry_id VARCHAR(255) UNIQUE NOT NULL,
    lossless_restatement TEXT NOT NULL,
    keywords TEXT[] DEFAULT '{}',
    timestamp VARCHAR(255),
    location VARCHAR(512),
    persons TEXT[] DEFAULT '{}',
    entities TEXT[] DEFAULT '{}',
    topic VARCHAR(512),
    vector vector(1024),  -- Titan V2 with 1024 dimensions
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', lossless_restatement)
    ) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for fast retrieval
CREATE INDEX IF NOT EXISTS idx_memory_entries_vector 
    ON memory_entries USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_memory_entries_search 
    ON memory_entries USING gin(search_vector);

CREATE INDEX IF NOT EXISTS idx_memory_entries_persons 
    ON memory_entries USING gin(persons);

CREATE INDEX IF NOT EXISTS idx_memory_entries_entities 
    ON memory_entries USING gin(entities);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'SimpleMem database initialized successfully!';
    RAISE NOTICE 'pgvector extension enabled';
    RAISE NOTICE 'memory_entries table created with vector(1024) column';
END $$;
