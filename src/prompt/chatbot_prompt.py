from pathlib import Path

file_path = Path(__file__).parents[0] / Path("chatbot_prompt.md")
with file_path.open("r", encoding='utf-8') as f:
    CHATBOT_SYSTEM_PROMPT = f.read()
