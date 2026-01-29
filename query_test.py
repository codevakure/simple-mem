"""Quick test to query SimpleMem memories"""
import requests

queries = [
    # Entity-specific queries (should return Loan #12345 specific info)
    "What is the payment schedule for John Smith loan?",
    "Loan 12345 details",
    
    # Universal rule queries (should return the 30-year loan rule)
    "What payment types are valid for 30-year loans?",
    "Can I use same day payment for a 30 year loan?",
]

for q in queries:
    print(f"\n=== Query: {q} ===")
    resp = requests.post("http://localhost:8001/query", json={
        "agent_id": "loan_agent",
        "query": q,
        "top_k": 3
    })
    data = resp.json()
    if data.get("context"):
        print(f"  Memory count: {data.get('memory_count', 0)}")
        print(f"  Context:\n{data['context']}")
    else:
        print("  No context found")
