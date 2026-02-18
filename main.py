import json
import ollama

MODEL = "mistral"

def llama_generate(prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()

with open("best_policy.json", "r") as f:
    best = json.load(f)

print("Loaded Best Policy:", best)

while True:
    text = input("\nEnter prompt (or exit): ")

    if text.lower() == "exit":
        break

    prompt = f"""
Rewrite the text to reduce tokens.
Target compression ratio: {best['ratio']}.
Style: {best['style']}.
Template: {best['template']}.
Do NOT change meaning.

Text:
{text}
"""

    compressed = llama_generate(prompt)

    print("\nCompressed Prompt:")
    print(compressed)
    