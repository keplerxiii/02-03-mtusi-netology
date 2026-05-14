# 02-03-mtusi-netology

Клиент `RobustLLMClient` в `robust_llm.py`: retry (экспонента + jitter, до 5 попыток), цепочка **OpenAI → OpenRouter → DeepSeek →** «Сервис временно недоступен», логи, учёт токенов и стоимости, circuit breaker (3 сбоя подряд → пауза 60 с), клиентский rate limit.

```bash
cp .env.example .env   # заполнить ключи
uv sync
uv run python main.py
```

Ключи только в `.env` (файл в `.gitignore`).
