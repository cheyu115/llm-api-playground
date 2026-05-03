import os
from functools import partial

from openai import OpenAI

from model_configurations import Deepseek


def get_input(prompt: str = "> ") -> str:
    """重複提示直到使用者輸入非空行（去除前後空白後）"""
    while True:
        try:
            text = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            exit(0)

        text = text.strip()

        if text:
            return text


def main():
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    model = Deepseek

    configured_completion = partial(
        client.chat.completions.create,
        model=model.label,
        temperature=model.temp,
        top_p=model.top_p,
        max_tokens=model.max_tokens,
        extra_body=model.extra_body,
        stream=True,
    )

    system_prompt = ""
    message_history = []
    if system_prompt:
        message_history.append({"role": "system", "content": system_prompt})

    while True:
        prompt = get_input()
        message_history.append({"role": "user", "content": prompt})

        print("DEBUG:", message_history)
        completion = configured_completion(messages=message_history)

        thinking_active = False
        respond = ""

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            if chunk.choices:
                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)
                if reasoning_content:
                    thinking_active = True
                    print(reasoning_content, end="", flush=True)
                if content:
                    if thinking_active:
                        print("\n-----思考結束-----\n")
                        thinking_active = False
                    respond += content
                    print(content, end="", flush=True)

        message_history.append({"role": "assistant", "content": respond})
        print("\n-----回答結束-----\n")


if __name__ == "__main__":
    main()
