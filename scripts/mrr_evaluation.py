def mean_reciprocal_rank(retrieved_results, relevant_docs):
    """
    retrieved_results: dict
        key   -> query id
        value -> list of retrieved document ids (ranked)

    relevant_docs: dict
        key   -> query id
        value -> list or set of relevant document ids
    """

    reciprocal_ranks = []

    for query_id in retrieved_results:
        retrieved_list = retrieved_results[query_id]
        relevant_set = set(relevant_docs[query_id])

        rank_found = 0  # 0 means no relevant document found

        for rank, doc_id in enumerate(retrieved_list, start=1):
            if doc_id in relevant_set:
                rank_found = rank
                break

        if rank_found != 0:
            reciprocal_ranks.append(1 / rank_found)
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr

if __name__ == "__main__":
    retrieved_results = {
        "q1": ["d3", "d7", "d1", "d9"],
        "q2": ["d2", "d5", "d8"],
        "q3": ["d10", "d4", "d6"]
    }

    relevant_docs = {
        "q1": ["d1"],
        "q2": ["d5"],
        "q3": ["d6"]
    }

    mrr = mean_reciprocal_rank(retrieved_results, relevant_docs)
    print("MRR Score:", mrr)

