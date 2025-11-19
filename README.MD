A simple Telegram bot that i wrote to combat spam in my lovely Tesla chat.
At first, I tested it with a local LLM like Llama3, but then GPT-4o was released, which is faster, better, and cheaper
since Llama requires quite a relatively beefy hosting setup.

7-1-2025:
- Added structured output to the model.

8-1-2025:
- Switched to DeepSeek Chat model (cheaper and comparable quality).
- Added Docker support for easy deployment.

## How to run (Docker)

1.  Create `.env` file with your keys:
    ```
    TELEGRAM_API_TOKEN=your_token
    DEEPSEEK_API_KEY=your_key
    ```

2.  Run with Docker Compose:
    ```bash
    touch spam_log.txt safe_users.json
    docker compose up -d --build
    ```

## Deployment notes
The `safe_users.json` file (whitelist) is in `.gitignore`. When deploying to a new server, you need to copy it manually or create a new one:
```bash
echo "{}" > safe_users.json
```
