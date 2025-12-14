from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import chromadb
from typing import Optional
import pprint
import logging
from vectorstore import save_text_to_db

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
chunk_size = 250
chunk_overlap = 30
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on, return_each_line=True
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)


def read_text(file_path):
    try:
        with open(file_path) as file:
            content = file.read()
        return content
    except FileNotFoundError:
        logger.error(f"文本文件不存在: {file_path}")
        raise
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        raise


def main():
    """
    # 1. 读文件
    """
    content = read_text("README.md")
    # MD splits
    md_header_splits = markdown_splitter.split_text(content)
    # pprint.pprint(md_header_splits)
    splits = text_splitter.split_documents(md_header_splits)
    success_count = 0
    for idx, doc in enumerate(splits, 1):
        try:
            logging.info(f"正保存第{idx}个")
            save_text_to_db(doc)
            success_count += 1
        except Exception as e:
            logger.error(f"保存第{idx}块失败:{str(e)}")
    logging.info(f"完成向量化并入库:success:{success_count}/{len(splits)}")


main()
