#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
LangChain API - RAG（检索增强生成）
================================================================================

模块：langchain_community.vectorstores, langchain_community.document_loaders
用途：结合外部知识库增强 LLM 的回答能力

RAG 流程：
--------------------------------------------------------------------------------
1. 文档加载 (Document Loading)
   - 从各种来源加载文档（PDF、网页、数据库等）
   - Document: page_content + metadata

2. 文档分割 (Document Splitting)
   - 将长文档切分成小块
   - 常用：RecursiveCharacterTextSplitter

3. 向量化 (Embedding)
   - 将文本转换为向量表示
   - 常用：OpenAIEmbeddings

4. 向量存储 (Vector Store)
   - 存储文档向量
   - 支持：FAISS, Chroma, Pinecone 等

5. 检索 (Retrieval)
   - 根据查询向量检索相似文档
   - 支持相似度搜索、MMR 等

6. 生成 (Generation)
   - 将检索结果作为上下文
   - 结合用户问题生成回答

核心组件：
--------------------------------------------------------------------------------
- DocumentLoaders: 文档加载器
- TextSplitters: 文本分割器
- Embeddings: 嵌入模型
- VectorStore: 向量存储
- Retriever: 检索器
================================================================================
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def create_sample_documents():
    """
    创建示例文档
    """
    documents = [
        Document(
            page_content="Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。"
                         "Python的设计哲学强调代码的可读性和简洁性。",
            metadata={"source": "python_intro", "topic": "Python"}
        ),
        Document(
            page_content="Python支持多种编程范式，包括面向对象编程、函数式编程和过程式编程。"
                         "Python有一个大型标准库，提供了丰富的功能。",
            metadata={"source": "python_features", "topic": "Python"}
        ),
        Document(
            page_content="JavaScript是一种动态编程语言，主要用于网页开发。"
                         "它可以在浏览器中运行，也可以通过Node.js在服务器端运行。",
            metadata={"source": "js_intro", "topic": "JavaScript"}
        ),
        Document(
            page_content="Go语言（也称Golang）由Google开发，是一种静态类型、编译型语言。"
                         "Go语言以并发编程和高性能著称。",
            metadata={"source": "go_intro", "topic": "Go"}
        ),
        Document(
            page_content="Rust是一种系统编程语言，专注于安全、并发和性能。"
                         "Rust通过所有权系统实现内存安全，无需垃圾回收器。",
            metadata={"source": "rust_intro", "topic": "Rust"}
        ),
    ]
    return documents


def basic_rag():
    """
    基础 RAG 示例
    """
    # 初始化组件
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    # 创建文档
    documents = create_sample_documents()
    
    # 创建向量存储
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 定义 RAG 提示词
    prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文：
{context}

问题：{question}

回答：
""")
    
    # 构建格式化文档的函数
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 构建 RAG 链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 提问
    question = "Python是什么时候创建的？"
    result = rag_chain.invoke(question)
    print(f"问题: {question}")
    print(f"回答: {result}")


def rag_with_text_splitter():
    """
    使用文本分割器的 RAG
    """
    from langchain_community.document_loaders import TextLoader
    
    # 创建一个长文本
    long_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行
    通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。
    
    机器学习是人工智能的一个子领域，它使计算机系统能够从数据中学习和改进，而无需显式编程。
    深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。
    
    自然语言处理（NLP）是人工智能的另一个重要领域，专注于计算机与人类语言之间的交互。
    NLP 应用包括机器翻译、情感分析、聊天机器人和文本摘要。
    
    计算机视觉是人工智能的领域之一，使计算机能够从图像或视频中获取信息。
    应用包括图像识别、目标检测和自动驾驶汽车。
    """
    
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )
    
    # 分割文本
    splits = text_splitter.create_documents([long_text])
    
    print(f"分割后的文档数量: {len(splits)}")
    for i, split in enumerate(splits[:3]):
        print(f"\n文档 {i+1}: {split.page_content[:50]}...")
    
    # 创建向量存储
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 相似度搜索
    query = "什么是深度学习？"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"\n查询: {query}")
    print("检索结果:")
    for doc in results:
        print(f"  - {doc.page_content[:80]}...")


def rag_with_score():
    """
    带相似度分数的 RAG
    """
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    documents = create_sample_documents()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 带分数的相似度搜索
    query = "Python有哪些特点？"
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    
    print(f"查询: {query}")
    print("\n检索结果（带分数）:")
    for doc, score in results_with_scores:
        print(f"  分数: {score:.4f}")
        print(f"  内容: {doc.page_content[:60]}...")
        print()


def mmr_retrieval():
    """
    MMR（最大边际相关性）检索 - 提高结果多样性
    """
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    documents = create_sample_documents()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # MMR 检索
    query = "编程语言"
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=10  # 候选文档数量
    )
    
    print(f"查询: {query}")
    print("\nMMR 检索结果（多样性更好）:")
    for i, doc in enumerate(mmr_results):
        print(f"{i+1}. {doc.metadata.get('topic', 'Unknown')}: {doc.page_content[:50]}...")


def retriever_as_runnable():
    """
    将检索器作为 Runnable 使用
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    embeddings = OpenAIEmbeddings(
        api_key=os.environ.get("ONE_API_KEY"),
        base_url=os.environ.get("BASE_URL")
    )
    
    documents = create_sample_documents()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    
    # 使用 retriever | format_docs 的方式
    def format_docs(docs):
        return "\n\n".join(f"[{doc.metadata.get('topic')}]: {doc.page_content}" for doc in docs)
    
    prompt = ChatPromptTemplate.from_template("""
根据以下参考资料回答问题：

{context}

问题：{question}
""")
    
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = chain.invoke("哪种语言适合并发编程？")
    print("回答:", result)


if __name__ == "__main__":
    print("=" * 60)
    print("1. 基础 RAG")
    print("=" * 60)
    basic_rag()
    
    print("\n" + "=" * 60)
    print("2. 文本分割")
    print("=" * 60)
    rag_with_text_splitter()
    
    print("\n" + "=" * 60)
    print("3. 带分数的检索")
    print("=" * 60)
    rag_with_score()
    
    print("\n" + "=" * 60)
    print("4. MMR 检索")
    print("=" * 60)
    mmr_retrieval()
    
    print("\n" + "=" * 60)
    print("5. Retriever 作为 Runnable")
    print("=" * 60)
    retriever_as_runnable()
