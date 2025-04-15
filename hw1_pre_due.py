import os
from dotenv import load_dotenv
import sys
from openai import AzureOpenAI
from gt import correct_category
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from utils import *

# TODO: Change to non multithreading
# TODO: Remove back to tkts_1/2 with no test
# TODO: Fix logic in correct category in part3
# TODO: Wrapp getenv with exception handling for avoiding failure when running as a script
# TODO: Prompt script runner with name of files were created and their path
# --- Get environment variables contents ---
MODEL = 'gpt-35-16k'
OPENAI_API_VERSION = '2023-12-01-preview'
MODEL_4o = 'gpt-4o-mini'
OPENAI_API_VERSION_4o = '2024-08-01-preview'



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


# --- File Readers ---

def load_tickets(path):
    tickets = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                num, text = line.strip().split(':', 1)
                tickets[int(num)] = text.strip()
    return tickets

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



# Collect results in a dict like {ticket_num: [(model, temp, label, response), ...]}
def generate_split_predictions(tickets, model_configs):
    results = {}

    for ticket_num, text in tickets.items():
        model_results = []
        for model, temp, client in model_configs:
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
                normalized = normalize_response(llm_raw)
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

############################
#######--PART_2#############
############################
ALLOWED_CATEGORIES = {
        "interface",
        "lacking feature",
        "logic defect",
        "data",
        "security and access control",
        "configuration",
        "stability",
        "performance"
    }

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

def normalize_category_response(resp):
    resp = resp.lower().strip()
    for cat in ALLOWED_CATEGORIES:
        if cat in resp:
            return cat
    return "unknown"


def generate_category_predictions_threaded(tickets, model_temp_category_configs, max_workers=10):
    results_lock = threading.Lock()
    results = defaultdict(list)

    def call_model(ticket_num, ticket_text, model, temp, client):
        """
        Function to run in each thread: sends the request and returns the result.
        """
        prompt = CATEGORY_PROMPT_TEMPLATE.format(ticket=ticket_text)
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
            normalized = normalize_category_response(llm_raw)
        except Exception as e:
            llm_raw = f"Error: {str(e)}"
            normalized = "unknown"

        return ticket_num, model, temp, normalized, llm_raw

    # Launch all requests in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ticket_num, ticket_text in tickets.items():
            for model, temp, client in model_temp_category_configs:
                futures.append(
                    executor.submit(call_model, ticket_num, ticket_text, model, temp, client)
                )

        # As each one completes, add it to results
        for future in as_completed(futures):
            ticket_num, model, temp, normalized, llm_raw = future.result()
            with results_lock:
                results[ticket_num].append((model, temp, normalized, llm_raw))

    return results


def generate_category_predictions(tickets, model_temp_category_configs):
    results = {}

    for ticket_num, text in tickets.items():
        model_results = []
        for model, temp, client in model_temp_category_configs:
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

        if ticket_num == 6:
            print(category_counts)
        majority = max(category_counts.values()) if category_counts else 0
        majority_label = "correct" if majority >= 4 else "incorrect"
        if majority_label == "correct":
            majority_correct += 1

        ticket_stats[ticket_num] = {
            "correct_count": correct_count,
            "majority": majority_label
        }
        if ticket_num == 6:
            print(ticket_stats)
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

def main():

    # Load environment variables from a file called .env
    load_dotenv('environment_variables.env')

    # Now use getenv to safely retrieve values
    azure_openai_api_key = os.getenv("CLASS_AZURE_KEY")
    azure_openai_endpoint = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT")
    azure_openai_endpoint_4o = os.getenv("SUBSCRIPTION_OPENAI_ENDPOINT_4o")

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

    test_client_connection(client_model1, "gpt-35-16k")
    test_client_connection(client_model2, "gpt-4o-mini")

    # Part 1:
    ## Each tuple = (model_name, temperature, associated_client)
    MODEL_TEMP_CONFIGS = [
        (MODEL, 0.0, client_model1),
        (MODEL_4o, 0.0, client_model2),
        (MODEL_4o, 0.9, client_model2)
    ]

    tickets = load_tickets('1test.txt')
    split_results = generate_split_predictions(tickets, MODEL_TEMP_CONFIGS)
    write_split_results_to_file(split_results, "split.txt")
    print(" split.txt has been created.")

    # Part 2:
    MODEL_TEMP_CATEGORY_CONFIGS = [
        (MODEL, 0.0, client_model1),
        (MODEL, 0.5, client_model1),
        (MODEL, 0.9, client_model1),
        (MODEL_4o, 0.0, client_model2),
        (MODEL_4o, 0.5, client_model2),
        (MODEL_4o, 0.9, client_model2)
    ]

    tickets = load_tickets("2test.txt")
    start_time = time.time()
    category_results = generate_category_predictions_threaded(tickets,MODEL_TEMP_CATEGORY_CONFIGS
                                                              ,max_workers=10)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️ Time taken for Part 2 (multithreaded): {elapsed_time:.2f} seconds")
    write_category_results_to_file(category_results, "categories.txt")
    print("categories.txt has been created.")

    # Part 3:
    results = parse_categories_file("categories.txt")
    ticket_stats, temp_stats, model_stats, majority_correct, total_tickets = analyze_categories(results)
    write_statistics_to_file(ticket_stats, temp_stats, model_stats, majority_correct, total_tickets, "statistics.txt")
    print("✅ statistics.txt created.")

if __name__ == '__main__':
    main()
