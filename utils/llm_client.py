#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 及多 LLM API 统一调用客户端模块 (Unified LLM Client)

支持 DeepSeek API (`deepseek-chat`, `deepseek-reasoner`)，
同时兼容 OpenAI、智谱 (Zhipu)、硅基流动 (SiliconFlow) 等 OpenAI 兼容接口。
"""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List, Union

class QuantLLMClient:
    """量化系统通用大模型接口客户端"""
    
    # 常用 Provider 预设配置
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "default_model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "env_key": "OPENAI_API_KEY"
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "default_model": "glm-4-flash",
            "env_key": "ZHIPU_API_KEY"
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "deepseek-ai/DeepSeek-V3",
            "env_key": "SILICONFLOW_API_KEY"
        }
    }

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60
    ):
        """
        初始化 LLM 客户端
        :param provider: 目标服务商 ("deepseek", "openai", "zhipu", "siliconflow", 或 "custom")
        :param api_key: 可选，如未提供则自动从环境变量获取
        :param base_url: 可选，如未提供则使用预设 base_url
        :param model: 可选，如未提供则使用默认模型
        :param timeout: 请求超时时间（秒）
        """
        self.provider = provider.lower()
        preset = self.PROVIDERS.get(self.provider, {})

        # 获取 API Key
        env_var = preset.get("env_key", "LLM_API_KEY")
        self.api_key = api_key or os.getenv(env_var) or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # 获取 Base URL
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or preset.get("base_url", "https://api.deepseek.com")).rstrip('/')

        # 获取 Model
        self.model = model or os.getenv("LLM_MODEL") or preset.get("default_model", "deepseek-chat")
        
        self.timeout = timeout

        if not self.api_key:
            print(f"[Notice] API Key for {self.provider} not found in environment. Please set {env_var} or DEEPSEEK_API_KEY in your .env file.")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        json_output: bool = False,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        调用 Chat Completions 接口
        :param messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
        :param model: 覆盖默认模型
        :param temperature: 温度参数
        :param max_tokens: 最大生成 token 数
        :param json_output: 是否强制输出 JSON 格式
        :return: 包含 response content 和解析结果的字典
        """
        target_model = model or self.model
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if json_output:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                result = {
                    "raw_content": content,
                    "model": target_model,
                    "usage": data.get("usage", {})
                }
                if json_output:
                    try:
                        result["parsed_json"] = json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"[Warning] JSON parse failed: {e}")
                        result["parsed_json"] = None
                return result
            else:
                print(f"[Error] API Status [{response.status_code}]: {response.text}")
                return None
        except Exception as e:
            print(f"[Error] Request failed: {e}")
            return None

    def analyze_sentiment(self, text: str, system_prompt: str, json_output: bool = True) -> Optional[Dict[str, Any]]:
        """简易情绪/新闻分析便捷方法"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return self.chat_completion(messages, json_output=json_output)


def get_deepseek_client(api_key: Optional[str] = None, model: str = "deepseek-chat") -> QuantLLMClient:
    """获取 DeepSeek 专属客户端实例"""
    return QuantLLMClient(provider="deepseek", api_key=api_key, model=model)


if __name__ == "__main__":
    client = get_deepseek_client()
    print(f"DeepSeek Client initialized! Base URL: {client.base_url}, Default Model: {client.model}")
