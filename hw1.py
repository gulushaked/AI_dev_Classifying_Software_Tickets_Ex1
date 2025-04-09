import os
from openai import AzureOpenAI

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
