from dataclasses import dataclass


@dataclass(frozen=True)
class Deepseek:
    label = "deepseek-ai/deepseek-v4-flash"
    temp = 1
    top_p = 0.95
    max_tokens = 16384
    extra_body = {
        "chat_template_kwargs": {"thinking": True},
        "reasoning_effort": "high",
    }
