from security_orchestrator.rag.retriever import build_local_retriever

retriever = build_local_retriever()

print("Knowledge base loaded:", len(retriever.documents), "documents")

query = "incident response cybersecurity risk management detection recovery"

print("\nQuery:", query)

results = retriever.retrieve(query, top_k=3)

print("\nRetrieved documents:", len(results))

for item in results:
    print("\nTitle:", item["title"])
    print("Source:", item["source"])
    print("Score:", round(item["score"], 4))
    print("Matched terms:", item["matched_terms"])
    print("URL:", item["source_url"])
