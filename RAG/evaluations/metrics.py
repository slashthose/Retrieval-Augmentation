from math import log2


def recall_at_k(retrieved, relevant):
    retrieved = set(retrieved)
    relevant = set(relevant)

    if len(relevant) == 0:
        return 0

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(retrieved, relevant):

    if len(retrieved) == 0:
        return 0

    retrieved = set(retrieved)
    relevant = set(relevant)

    return len(retrieved & relevant) / len(retrieved)


def hit_rate_at_k(retrieved, relevant):

    return int(
        len(set(retrieved) & set(relevant)) > 0
    )


def reciprocal_rank(retrieved, relevant):

    for i, chunk in enumerate(retrieved):

        if chunk in relevant:

            return 1 / (i + 1)

    return 0


def ndcg_at_k(retrieved, relevant):

    dcg = 0

    for i, chunk in enumerate(retrieved):

        if chunk in relevant:

            dcg += 1 / log2(i + 2)

    ideal = sum(
        1 / log2(i + 2)
        for i in range(min(len(relevant), len(retrieved)))
    )

    if ideal == 0:

        return 0

    return dcg / ideal