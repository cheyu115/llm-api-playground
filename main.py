import os
from functools import partial

from openai import OpenAI


def main():
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    create_completion = partial(
        client.chat.completions.create,
        model="deepseek-ai/deepseek-v4-pro",
        temperature=1,
        top_p=0.95,
        max_tokens=65536,
        extra_body={"chat_template_kwargs": {"thinking": True}},
        stream=True,
    )

    content = input("-> ")

    completion = create_completion(messages=[{"role": "user", "content": content}])

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    main()
