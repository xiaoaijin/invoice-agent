"""LangChain 模型工厂：统一走 DashScope OpenAI 兼容模式。

一个 SDK 覆盖视觉（qwen-vl-max）、对话（qwen-max）、嵌入（text-embedding-v3）。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from invoice_agent.config import get_settings, require_api_key


@lru_cache
def get_vision_model() -> ChatOpenAI:
    """识票用多模态模型。温度 0 保证可复现。"""
    s = get_settings()
    return ChatOpenAI(
        model=s.model_vision,
        api_key=require_api_key(),
        base_url=s.llm_base_url,
        temperature=0,
    )


@lru_cache
def get_chat_model() -> ChatOpenAI:
    """汇总回答/问答用对话模型。"""
    s = get_settings()
    return ChatOpenAI(
        model=s.model_chat,
        api_key=require_api_key(),
        base_url=s.llm_base_url,
        temperature=0.2,
    )


@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=require_api_key(),
        base_url=s.llm_base_url,
    )
