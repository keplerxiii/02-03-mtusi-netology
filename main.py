import logging
import os

from dotenv import load_dotenv

from robust_llm import RobustLLMClient, setup_logging


def main() -> None:
    load_dotenv()
    setup_logging(logging.INFO)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        logging.warning("Нет ключей в .env — будет ответ «Сервис временно недоступен»")

    client = RobustLLMClient()
    r = client.chat("Скажи одно слово: ОК")
    print("Ответ:", r.text)
    print("Провайдер:", r.provider, "модель:", r.model)
    print(
        "Сессия: prompt=",
        client.session_usage.prompt_tokens,
        "completion=",
        client.session_usage.completion_tokens,
        "cost_usd~=",
        round(client.session_usage.estimated_cost_usd, 6),
    )


if __name__ == "__main__":
    main()
