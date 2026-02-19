import optuna
import numpy as np
import json
from sklearn.metrics import accuracy_score
from dataset import small_dataset
import ollama

MODEL = "mistral"
LAMBDA = 0.1
N_TRIALS = 3



def llama_generate(prompt):

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()


def compress_prompt(text, ratio, style, template):

    prompt = f"""
Rewrite the text to reduce tokens.
Target compression ratio: {ratio}.
Style: {style}.
Template: {template}.
Do NOT change meaning.

Text:
{text}
"""

    return llama_generate(prompt)



def classify(text):

    prompt = f"""
Classify into one of:
World, Sports, Business, Sci/Tech.

Text:
{text}

Answer only label.
"""

    return llama_generate(prompt)



print("Computing baseline...")

baseline_preds = []

for item in small_dataset:
    pred = classify(item["text"])
    baseline_preds.append(pred)

true_labels = [item["label"] for item in small_dataset]
baseline_acc = accuracy_score(true_labels, baseline_preds)

print("Baseline Accuracy:", baseline_acc)



def objective(trial):

    ratio = trial.suggest_float("ratio", 0.6, 0.9)
    style = trial.suggest_categorical(
        "style",
        ["concise summary", "entity focused", "remove redundancy"]
    )
    template = trial.suggest_categorical(
        "template",
        ["plain", "compact"]
    )

    preds = []
    compression_scores = []

    for item in small_dataset:

        original = item["text"]

        compressed = compress_prompt(original, ratio, style, template)
        pred = classify(compressed)

        preds.append(pred)

        compression_scores.append(
            len(compressed.split()) / len(original.split())
        )

    acc = accuracy_score(true_labels, preds)
    avg_compression = np.mean(compression_scores)

    score = acc - LAMBDA * avg_compression

    return score



print("Running Bayesian Optimization...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

print("Best Parameters Found:")
print(study.best_params)

with open("best_policy.json", "w") as f:
    json.dump(study.best_params, f)

print("Best policy saved.")

