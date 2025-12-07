import numpy as np


# 计算一个向量与多个向量的余弦相似度
def cosine_similarity(from_vec, to_vecs):
    from_vec = np.array(from_vec, dtype=float)
    to_vecs = np.array(to_vecs, dtype=float)
    norm1 = np.linalg.norm(from_vec)
    similarities = []
    for to_vec in to_vecs:
        to_vec = np.array(to_vec, dtype=float)
        norm_vec = np.linalg.norm(to_vec)
        if norm1 == 0 or norm_vec == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(from_vec, to_vec) / (norm1 * norm_vec))
        similarities.append(similarity)
    return np.array(similarities)


def mmr_select(query_vector, doc_vectors, k=3, lambda_mult=0.5):
    quer_similarities = cosine_similarity(query_vector, doc_vectors)
    # 选择相关性最高的文档的下标, 组成列表
    selected = [int(np.argmax(quer_similarities))]

    while len(selected) < k:
        # 存放每个候选文档的mmr分数
        mmr_scores = []
        for i in range(len(doc_vectors)):
            if i not in selected:
                # 相关性,i与候选文档的相关性
                relevance = quer_similarities[i]
                selected_vecs = doc_vectors[selected]  # S 结果集
                sims = cosine_similarity(
                    doc_vectors[i], selected_vecs
                )  # i 与 S 结果集的相似度
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * np.max(sims)
                mmr_scores.append((i, mmr_score))

        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)

    return selected


# 相关性所占权重 0-1
lambda_mult = 1
k = 3
doc_vectors = np.array([[9, 2], [2, 9], [7, 8], [1, 3], [6, 1]])
query_vector = [4, 2]


selected = mmr_select(query_vector, doc_vectors, k=k, lambda_mult=lambda_mult)
print(f"lambda_mult={lambda_mult:.1f},最终文档{[int(s) for s in selected]}")
