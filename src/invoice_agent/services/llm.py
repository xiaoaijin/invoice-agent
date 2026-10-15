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
    """汇总回答/问答用对话模型。

    流式输出（UI 逐 token 展示）+ 显式关闭思考模式（qwen3 系列默认开启会拖慢首字
    到分钟级）+ 请求超时兜底，防止单次调用挂死整个 run。
    """
    s = get_settings()
    return ChatOpenAI(
        model=s.model_chat,
        api_key=require_api_key(),
        base_url=s.llm_base_url,
        temperature=0.2,
        streaming=True,
        timeout=60.0,
        max_retries=1,
        extra_body={"enable_thinking": False},
    )


@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=require_api_key(),
        base_url=s.llm_base_url,
        # DashScope 兼容端点只接受字符串数组，不接受 langchain 默认的 token 数组
        check_embedding_ctx_length=False,
        chunk_size=10,  # DashScope 嵌入接口单批上限 20 条
    )
