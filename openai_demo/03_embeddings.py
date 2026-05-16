#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
OpenAI API - Embeddings（向量嵌入）
================================================================================

接口：client.embeddings.create
用途：将文本转换为向量表示，用于语义搜索、聚类、分类、RAG等场景

字段说明：
--------------------------------------------------------------------------------
1. model: 模型名
   - 示例：text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
   - 必填
   - text-embedding-3-small: 1536维，性价比高
   - text-embedding-3-large: 3072维，效果更好
   - text-embedding-ada-002: 旧版模型

2. input: 输入文本
   - 示例："这是一段文本" 或 ["文本1", "文本2"]
   - 必填
   - 最多 2048 个输入
   - 单个文本最大 8191 tokens

3. encoding_format: 编码格式
   - 示例："float", "base64"
   - 默认："float"

4. dimensions: 输出维度
   - 示例：256, 512, 1536
   - 仅 text-embedding-3 系列支持
   - 可降低维度以减少存储成本

5. user: 用户标识
   - 示例："user_123"
================================================================================
"""

from config import create_client, MODELS
import numpy as np

client = create_client()


def single_embedding():
    """
    单文本嵌入示例
    """
    response = client.embeddings.create(
        model=MODELS["embedding"]["small"],
        input="这是一段需要转换为向量的文本"
    )
    
    vector = response.data[0].embedding
    print(f"向量维度: {len(vector)}")
    print(f"前10个值: {vector[:10]}")


def batch_embeddings():
    """
    批量文本嵌入示例
    """
    texts = [
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
        "自然语言处理处理人类语言",
        "计算机视觉让机器理解图像"
    ]
    
    response = client.embeddings.create(
        model=MODELS["embedding"]["small"],
        input=texts
    )
    
    for i, item in enumerate(response.data):
        print(f"文本 {i+1} 向量维度: {len(item.embedding)}")


def custom_dimensions():
    """
    自定义维度示例（仅 text-embedding-3 系列）
    """
    response = client.embeddings.create(
        model=MODELS["embedding"]["small"],
        input="降低向量维度可以减少存储成本",
        dimensions=256  # 自定义维度
    )
    
    vector = response.data[0].embedding
    print(f"自定义维度: {len(vector)}")


def cosine_similarity(vec1, vec2):
    """
    计算余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search():
    """
    语义搜索示例
    """
    # 文档库
    documents = [
        "Python是一种流行的编程语言",
        "机器学习可以用于数据预测",
        "苹果是一种水果",
        "北京是中国的首都",
        "深度学习使用多层神经网络"
    ]
    
    query = "编程语言有哪些？"
    
    # 获取查询向量
    query_response = client.embeddings.create(
        model=MODELS["embedding"]["small"],
        input=query
    )
    query_vector = query_response.data[0].embedding
    
    # 获取文档向量
    doc_response = client.embeddings.create(
        model=MODELS["embedding"]["small"],
        input=documents
    )
    doc_vectors = [item.embedding for item in doc_response.data]
    
    # 计算相似度并排序
    similarities = [
        (documents[i], cosine_similarity(query_vector, doc_vectors[i]))
        for i in range(len(documents))
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"查询: {query}")
    print("\n最相关的文档:")
    for doc, score in similarities[:3]:
        print(f"  相似度 {score:.4f}: {doc}")


def clustering_demo():
    """
    文本聚类示例
    """
    from sklearn.cluster import KMeans
    
    texts = [
        "苹果手机很好用",
        "安卓手机性价比高",
        "香蕉是一种水果",
        "橙子富含维生素C",
        "机器学习很有趣",
        "深度学习是AI的核心"
    ]
    
    # 获取嵌入
    response = client.embeddings.create(
        model=MODELS["embedding"]["small"],
        input=texts
    )
    vectors = np.array([item.embedding for item in response.data])
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(vectors)
    
    # 按聚类分组
    clusters = {}
    for text, label in zip(texts, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)
    
    print("聚类结果:")
    for label, items in clusters.items():
        print(f"\n类别 {label + 1}:")
        for item in items:
            print(f"  - {item}")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 单文本嵌入")
    print("=" * 60)
    single_embedding()
    
    print("\n" + "=" * 60)
    print("2. 批量文本嵌入")
    print("=" * 60)
    batch_embeddings()
    
    print("\n" + "=" * 60)
    print("3. 自定义维度")
    print("=" * 60)
    custom_dimensions()
    
    print("\n" + "=" * 60)
    print("4. 语义搜索")
    print("=" * 60)
    semantic_search()
    
    print("\n" + "=" * 60)
    print("5. 文本聚类")
    print("=" * 60)
    clustering_demo()