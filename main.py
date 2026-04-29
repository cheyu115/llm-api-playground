import os
from functools import partial

from openai import OpenAI


def get_input(prompt: str = "> ") -> str:
    """重複提示直到使用者輸入非空行（去除前後空白後）"""
    while True:
        try:
            text = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            exit(0)

        if text.strip():
            return text


def main():
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    configured_completion = partial(
        client.chat.completions.create,
        model="deepseek-ai/deepseek-v4-pro",
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"thinking": True},
            "reasoning_effort": "max",
        },
        stream=True,
    )

    while True:
        prompt = get_input()

        completion = configured_completion(
            messages=[{"role": "user", "content": prompt}]
        )

        thinking_active = False

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            if chunk.choices:
                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    thinking_active = True
                    print(reasoning_content, end="", flush=True)
                if delta.content is not None:
                    if thinking_active:
                        print("\n-----思考結束-----")
                        thinking_active = False
                    print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n-----回答結束-----")


if __name__ == "__main__":
    main()
