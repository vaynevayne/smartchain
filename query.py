import os
from typing import Optional, List
import chromadb
import logging
from sentence_transformers import SentenceTransformer
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = "./chroma_db"

query = "李胜强的外号叫什么?"
logger.info(f"用户查询:{query}")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
os.environ["HF-ENDPOINT"] = "https://hf-mirror.com"

DEFAULT_COLLECTION_NAME = "rag"


def query_rag(query: str, n_results):
    # 1.把查询条件转向量

    embedding = SentenceTransformer(DEFAULT_MODEL_NAME).encode([query])[0].tolist()
    client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)
    # 2.检索上下文
    related_chunks = collection.query(
        query_embeddings=[embedding], n_results=n_results
    ).get("documents")
    if not related_chunks or not related_chunks[0]:
        raise ValueError("未查询到内容")
    # print(f"上下文:{related_chunks}")
    context = "\n".join(related_chunks[0])
    # 3.拼接提示词
    prompt = f"""
    你是一个AI助手
    用户提问:{query}
    已知信息:{context}
    """
    result = ChatDeepSeek(model="deepseek-chat", temperature=0).invoke(
        [{"role": "user", "content": prompt}]
    )
    # print(f"llm:result,{result}")
    return result.content


try:
    answer = query_rag(query, n_results=3)
    print(f"答案:{answer}")
except Exception as e:
    logging.error(f"错误:{e}")
    raise
