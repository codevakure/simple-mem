"""Check database contents"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.vector_store import VectorStore

vs = VectorStore()
count = vs.count_rows()
print(f"Total entries in database: {count}")

if count > 0:
    entries = vs.get_all_entries()
    print("\nStored entries:")
    for e in entries[:10]:
        print(f"  Agent: {e.agent_id}")
        print(f"  Content: {e.lossless_restatement[:100]}...")
        print()
