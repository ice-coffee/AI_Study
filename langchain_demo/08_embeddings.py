#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - Embeddings（向量嵌入）
================================================================================

模块：langchain_openai, langchain_community.embeddings
用途：将文本转换为向量表示，用于语义搜索、相似度计算等

核心概念：
--------------------------------------------------------------------------------
1. Embeddings: 嵌入模型接口
   - embed_documents(texts): 批量嵌入文档
   - embed_query(text): 嵌入单个查询
   - 返回浮点数列表 (向量)

2. 向量维度:
   - OpenAI text-embedding-3-small: 1536 维
   - OpenAI text-embedding-3-large: 3072 维
   - 不同模型维度不同

3. 用途:
   - 语义搜索
   - 文本聚类
   - 相似度计算
   - 推荐系统
   - RAG 应用

常用嵌入模型：
--------------------------------------------------------------------------------
- OpenAIEmbeddings: OpenAI 的嵌入模型
- HuggingFaceEmbeddings: HuggingFace 开源模型
- CohereEmbeddings: Cohere 嵌入模型
================================================================================
"""

import os
import numpy as np
from langchain_openai import OpenAIEmbeddings


def basic_embedding():
    """
    基础嵌入示例
    """
    # 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 嵌入单个文本
    text = "Python 是一种流行的编程语言"
    vector = embeddings.embed_query(text)
    
    print(f"文本: {text}")
    print(f"向量维度: {len(vector)}")
    print(f"向量前5个元素: {vector[:5]}")


def batch_embedding():
    """
    批量嵌入文档
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 批量嵌入
    texts = [
        "Python 是一种编程语言",
        "JavaScript 主要用于网页开发",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络",
    ]
    
    vectors = embeddings.embed_documents(texts)
    
    print(f"文档数量: {len(texts)}")
    print(f"向量数量: {len(vectors)}")
    print(f"每个向量维度: {len(vectors[0])}")


def cosine_similarity():
    """
    计算余弦相似度
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 嵌入多个句子
    sentences = [
        "我喜欢吃苹果",
        "苹果是一种水果",
        "我喜欢编程",
        "Python 是一种编程语言",
    ]
    
    vectors = embeddings.embed_documents(sentences)
    
    # 定义余弦相似度函数
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 计算相似度矩阵
    print("相似度矩阵:")
    print("-" * 50)
    
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            sim = cosine_sim(vectors[i], vectors[j])
            if i <= j:  # 只打印上三角
                print(f"'{s1[:10]}...' vs '{s2[:10]}...': {sim:.4f}")


def semantic_search():
    """
    语义搜索示例
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 文档库
    documents = [
        "Python 是一种高级编程语言，以其简洁的语法著称",
        "JavaScript 是 Web 开发的核心技术之一",
        "机器学习可以让计算机从数据中学习",
        "深度学习是机器学习的一个子领域",
        "自然语言处理专注于计算机与人类语言的交互",
        "计算机视觉使机器能够理解图像和视频",
    ]
    
    # 嵌入所有文档
    doc_vectors = embeddings.embed_documents(documents)
    
    # 查询
    query = "什么是 AI？"
    query_vector = embeddings.embed_query(query)
    
    # 计算相似度
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarities = [
        (doc, cosine_sim(query_vector, doc_vec))
        for doc, doc_vec in zip(documents, doc_vectors)
    ]
    
    # 排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"查询: {query}")
    print("\n搜索结果 (按相似度排序):")
    for i, (doc, sim) in enumerate(similarities, 1):
        print(f"{i}. [相似度: {sim:.4f}] {doc[:40]}...")


def embedding_with_different_models():
    """
    使用不同嵌入模型
    """
    # text-embedding-3-small
    small_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # text-embedding-3-large
    large_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    text = "Hello, World!"
    
    small_vector = small_embeddings.embed_query(text)
    large_vector = large_embeddings.embed_query(text)
    
    print(f"文本: {text}")
    print(f"text-embedding-3-small 维度: {len(small_vector)}")
    print(f"text-embedding-3-large 维度: {len(large_vector)}")


def clustering_example():
    """
    简单的文本聚类示例
    """
    from collections import defaultdict
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 不同类别的句子
    sentences = [
        # 编程相关
        "Python 是一种编程语言",
        "JavaScript 用于网页开发",
        "Go 语言支持并发编程",
        # 水果相关
        "苹果是一种水果",
        "香蕉富含钾元素",
        "橙子含有丰富的维生素C",
        # 运动相关
        "足球是一项团队运动",
        "篮球需要运球技巧",
        "游泳是一项有氧运动",
    ]
    
    # 嵌入所有句子
    vectors = embeddings.embed_documents(sentences)
    
    # 计算中心点
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 简单聚类：找每个句子的最近邻
    print("句子相似度分组:")
    for i, (sent, vec) in enumerate(zip(sentences, vectors)):
        # 找最相似的其他句子
        similarities = [
            (j, cosine_sim(vec, vectors[j]))
            for j in range(len(vectors)) if j != i
        ]
        best_match = max(similarities, key=lambda x: x[1])
        print(f"'{sent[:15]}...' -> 最相似: '{sentences[best_match[0]][:15]}...' ({best_match[1]:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础嵌入")
    print("=" * 60)
    basic_embedding()
    
    print("\n" + "=" * 60)
    print("2. 批量嵌入")
    print("=" * 60)
    batch_embedding()
    
    print("\n" + "=" * 60)
    print("3. 余弦相似度")
    print("=" * 60)
    cosine_similarity()
    
    print("\n" + "=" * 60)
    print("4. 语义搜索")
    print("=" * 60)
    semantic_search()
    
    print("\n" + "=" * 60)
    print("5. 不同嵌入模型")
    print("=" * 60)
    embedding_with_different_models()
    
    print("\n" + "=" * 60)
    print("6. 文本聚类")
    print("=" * 60)
    clustering_example()
