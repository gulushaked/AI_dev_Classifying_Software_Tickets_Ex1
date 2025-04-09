import os
from openai import AzureOpenAI
from gt import correct_category
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# TODO: Edge cases of LLM returns other outputs than split/no split
# TODO: use getenv to read .env file
# TODO: use multithreading for issuing requests
# --- Get environment variables contents ---
MODEL = 'gpt-35-16k'
OPENAI_API_VERSION = '2023-12-01-preview'
MODEL_4o = 'gpt-4o-mini'
OPENAI_API_VERSION_4o = '2024-08-01-preview'

azure_openai_api_key = os.environ["CLASS_AZURE_KEY"]
azure_openai_endpoint = os.environ["SUBSCRIPTION_OPENAI_ENDPOINT"]
azure_openai_endpoint_4o = os.environ["SUBSCRIPTION_OPENAI_ENDPOINT_4o"]

# --- Initialize Azure OpenAI clients for two models ---
client_model1 = AzureOpenAI(
    api_key=azure_openai_api_key,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=azure_openai_endpoint
)

client_model2 = AzureOpenAI(
    api_key=azure_openai_api_key,
    api_version=OPENAI_API_VERSION_4o,
    azure_endpoint=azure_openai_endpoint_4o
)
# ---Test Conn--
def test_client_connection(client, model_name):
    try:
        print(f"Testing connection to model: {model_name} ...")
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a simple echo bot."},
                {"role": "user", "content": "Hello"}
            ]
        )
        test_output = response.choices[0].message.content
        print(f"✅ Connection to {model_name} successful. Response: {test_output}")
    except Exception as e:
        print(f"❌ Connection to {model_name} failed: {e}")


test_client_connection(client_model1, "gpt-35-16k")
test_client_connection(client_model2, "gpt-4o-mini")
# --- Static Configuration ---

SYSTEM_MESSAGE = (
    """ You are an expert software ticket classifier.
        Your job is to read a bug ticket and decide if it contains one issue or multiple distinct issues."""
)

PROMPT_TEMPLATE = """You are a bug triage expert. Your task is to decide if a user-reported software bug ticket contains one issue or multiple separate issues.

Definitions:
- "no-split" = the ticket clearly describes one distinct issue.
- "split" = the ticket contains more than one distinct problem.
- "unknown" = the ticket is unclear or it's not possible to confidently decide.

Examples:
1. "The interface layout breaks when resizing the window." -> no-split
2. "Reports show incorrect calculations, and the interface layout breaks when resizing." -> split
3. "Passowrd is visible to everyone and username too. -> no-split

Now analyze the following ticket and respond with only: "split", "no-split".

Ticket:
"{ticket}"
"""

# Each tuple = (model_name, temperature, associated_client)
MODEL_TEMP_CONFIGS = [
    (MODEL, 0.0, client_model1),
    (MODEL_4o, 0.0, client_model2),
    (MODEL_4o, 0.9, client_model2)
]

# --- File Readers ---

def load_tickets(path):
    tickets = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                num, text = line.strip().split(':', 1)
                tickets[int(num)] = text.strip()
    return tickets

def load_ground_truth(path):
    truth = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                num, label = line.strip().split(':', 1)
                truth[int(num)] = label.strip().lower()
    return truth

"""
# --- LLM Output Cleaning ---
def normalize_response(resp):
    resp = resp.lower().strip()
    if "split" in resp and "no-split" not in resp:
        return "split"
    elif "no-split" in resp:
        return "no-split"
    elif resp in ["split", "no-split", "unknown"]:
        return resp
    else:
        return "unknown"

"""

# Collect results in a dict like {ticket_num: [(model, temp, label, response), ...]}
def generate_split_predictions(tickets):
    results = {}

    for ticket_num, text in tickets.items():
        model_results = []
        for model, temp, client in MODEL_TEMP_CONFIGS:
            prompt = PROMPT_TEMPLATE.format(ticket=text)
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ]
                )
                llm_raw = response.choices[0].message.content.strip()
                normalized = llm_raw
            except Exception as e:
                llm_raw = f"Error: {str(e)}"
                normalized = "unknown"

            model_results.append((model, temp, normalized, llm_raw))
        results[ticket_num] = model_results
    return results

# Save results to split.txt in required format
def write_split_results_to_file(results, output_path="split.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for ticket_num in sorted(results.keys()):
            f.write(f"{ticket_num}:\n")
            for model, temp, split_label, llm_response in results[ticket_num]:
                f.write(f"{model}, {temp}: {split_label}\n")
                f.write(f"LLM response: {llm_response}\n\n")  # blank line between responses

# --- Run Program ---
#tickets = load_tickets("tkts_1.txt")

# Step 1: Generate predictions
#split_results = generate_split_predictions(tickets)

# Step 2: Save to file in required format
#write_split_results_to_file(split_results, "split.txt")
print(" split.txt has been created.")

############################
#######--PART_2#############
############################

CATEGORY_SYSTEM_MESSAGE = (
    "You are an expert in software bug categorization. "
    "You must assign each ticket to one of the following categories: "
    "\"interface\", \"lacking feature\", \"logic defect\", \"data\", "
    "\"security and access control\", \"configuration\", \"stability\", \"performance\". "
    "Respond with only the category name. Do not explain your reasoning."
)

CATEGORY_PROMPT_TEMPLATE = """Classify the following software bug ticket into one of the categories:
- interface
- lacking feature
- logic defect
- data
- security and access control
- configuration
- stability
- performance

Respond with only the category name.

Ticket:
"{ticket}"
"""
MODEL_TEMP_CATEGORY_CONFIGS = [
    (MODEL, 0.0, client_model1),
    (MODEL, 0.5, client_model1),
    (MODEL, 0.9, client_model1),
    (MODEL_4o, 0.0, client_model2),
    (MODEL_4o, 0.5, client_model2),
    (MODEL_4o, 0.9, client_model2)
]

def generate_category_predictions(tickets):
    results = {}

    for ticket_num, text in tickets.items():
        model_results = []
        for model, temp, client in MODEL_TEMP_CATEGORY_CONFIGS:
            prompt = CATEGORY_PROMPT_TEMPLATE.format(ticket=text)
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": CATEGORY_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ]
                )
                llm_raw = response.choices[0].message.content.strip()
                normalized = llm_raw
                print(llm_raw) #DEBUG
            except Exception as e:
                llm_raw = f"Error: {str(e)}"
                normalized = "unknown"

            model_results.append((model, temp, normalized, llm_raw))
        results[ticket_num] = model_results
    return results

def write_category_results_to_file(results, output_path="categories.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        for ticket_num in sorted(results.keys()):
            f.write(f"{ticket_num}:\n")
            for model, temp, category, llm_response in results[ticket_num]:
                f.write(f"{model}, {temp}: {category}\n")
                f.write(f"LLM response: {llm_response}\n\n")

tickets = load_tickets("tkts_2.txt")
category_results = generate_category_predictions(tickets)
write_category_results_to_file(category_results, "categories.txt")
print("categories.txt has been created.")

##############################
#######--PART_3--#############
##############################

def parse_categories_file(path="categories.txt"):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_ticket = None
    for line in lines:
        line = line.strip()
        if not line:
            continue  # skip blank lines
        if re.match(r"^\d+:$", line):
            current_ticket = int(line[:-1])
            results[current_ticket] = []
        elif current_ticket is not None and ", " in line and ":" in line:
            try:
                model_temp, category = line.split(":")
                model, temp = model_temp.split(", ")
                category = category.strip().lower()
                results[current_ticket].append((model, float(temp), category))
            except ValueError:
                continue  # bad line format
    return results

def analyze_categories(results):
    stats_per_model = defaultdict(lambda: [0, 0])  # model: [correct, total]
    stats_per_temp = defaultdict(lambda: [0, 0])   # temperature: [correct, total]
    majority_correct = 0
    total_tickets = 0

    ticket_stats = {}

    for ticket_num, predictions in results.items():
        total_tickets += 1
        correct_count = 0
        category_counts = defaultdict(int)

        for model, temp, category in predictions:
            stats_per_temp[temp][1] += 1
            stats_per_model[model][1] += 1

            is_correct = correct_category(ticket_num, category)
            if is_correct:
                correct_count += 1
                stats_per_temp[temp][0] += 1
                stats_per_model[model][0] += 1

            if category != "unknown":
                category_counts[category] += 1

        majority = max(category_counts.values()) if category_counts else 0
        majority_label = "correct" if majority >= 4 else "incorrect"
        if majority_label == "correct":
            majority_correct += 1

        ticket_stats[ticket_num] = {
            "correct_count": correct_count,
            "majority": majority_label
        }

    return ticket_stats, stats_per_temp, stats_per_model, majority_correct, total_tickets

def write_statistics_to_file(ticket_stats, stats_per_temp, stats_per_model, majority_correct, total_tickets, output_path="statistics.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        # Write per-ticket stats
        for ticket_num in sorted(ticket_stats.keys()):
            f.write(f"{ticket_num}:\n")
            f.write(f"number correct: {ticket_stats[ticket_num]['correct_count']}\n")
            f.write(f"majority voting: {ticket_stats[ticket_num]['majority']}\n\n")

        # Summary
        f.write("summary\n")
        for temp in sorted(stats_per_temp.keys()):
            correct, total = stats_per_temp[temp]
            pct = (correct / total) * 100 if total > 0 else 0
            f.write(f"percent correct temperature = {temp}: {pct:.2f}\n")

        for model in sorted(stats_per_model.keys()):
            correct, total = stats_per_model[model]
            pct = (correct / total) * 100 if total > 0 else 0
            f.write(f"percent correct model = {model}: {pct:.2f}\n")

        # Totals
        total_correct = sum(c for c, _ in stats_per_model.values())
        total_predictions = sum(t for _, t in stats_per_model.values())
        overall_pct = (total_correct / total_predictions) * 100 if total_predictions > 0 else 0
        majority_pct = (majority_correct / total_tickets) * 100 if total_tickets > 0 else 0

        f.write(f"percent correct total: {overall_pct:.2f}\n")
        f.write(f"percent correct majority voting: {majority_pct:.2f}\n")

results = parse_categories_file("categories.txt")
ticket_stats, temp_stats, model_stats, majority_correct, total_tickets = analyze_categories(results)
write_statistics_to_file(ticket_stats, temp_stats, model_stats, majority_correct, total_tickets, "statistics.txt")
print("✅ statistics.txt created.")


