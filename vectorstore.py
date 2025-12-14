import chromadb
from typing import Optional
import pprint
import logging
import os
import hashlib
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 默认的集合名称
DEFAULT_COLLECTION_NAME = "rag"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
os.environ["HF-ENDPOINT"] = "https://hf-mirror.com"
DEFAULT_DB_PATH = "./chroma_db"
# 定义向量模型的全局变量
_model: Optional[SentenceTransformer] = None
# 定义全局的chromadb 客户端变量
_client: Optional[chromadb.PersistentClient] = None


def get_model():
    global _model
    if _model is None:
        print("正在加载模型进行测试...")
        _model = SentenceTransformer(DEFAULT_MODEL_NAME)
        print("模型加载完成")
    return _model


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        print("正在加载模型进行测试...")
        _client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
        print("模型加载完成")
    return _client


def save_text_to_db(text_or_doc, collection_name: str = DEFAULT_COLLECTION_NAME):
    try:
        if hasattr(text_or_doc, "page_content"):
            text = text_or_doc.page_content
            meta = getattr(text_or_doc, "metadata", {}) or {"source": "document"}
        else:
            text = str(text_or_doc)
            meta = {"source": "document"}
        if not text or not text.strip():
            logging.warning("尝试保存空文件,已跳过")
            return ""
        model = get_model()
        client = get_client()
        collection = client.get_or_create_collection(collection_name)
        text_id = hashlib.md5(text.encode("utf-8")).hexdigest()
        existing = collection.get(ids=[text_id])
        if existing and existing.get("ids"):
            logging.info(f"此文本已存在,跳过保存,{text}")
            return text_id
        embedding = model.encode([text])[0].tolist()
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[text_id],
            metadatas=[meta],
        )
        return text_id

    except Exception as e:
        logging.error(f"保存到数据库失败,{str(e)}")
        raise
