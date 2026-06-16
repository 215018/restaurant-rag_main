# This script implements the RAG chain for the restaurant menu chatbot.
# RAG means Retrieval-Augmented Generation:
# 1. Retrieve matching menu items from retriever.py.
# 2. Check direct availability questions like "Do you have pizza?"
# 3. Send only valid menu items to the LLM.
# 4. Generate a friendly customer answer in English.

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retriever import describe_spice_safety, retrieve_filtered_items


# Go up from src/rag_chain.py to the main restaurant-rag folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the .env file that stores GROQ_API_KEY.
ENV_PATH = PROJECT_ROOT / ".env"

# Load the Groq API key from the .env file.
load_dotenv(ENV_PATH)


def format_items_for_prompt(filtered_items, intent):
    # Convert filtered menu items into a short text block for the LLM prompt.
    item_blocks = []

    for item in filtered_items:
        # Create a spice safety note based on the customer's taste request.
        spice_note = describe_spice_safety(item, intent["taste"])

        # Keep this block short to avoid Groq token limit errors.
        block = f"""
Name: {item.get("name")}
Price: {item.get("price")} euros
Description: {item.get("description_de")}
Category: {item.get("category")}
Meal type: {item.get("meal_type")}
Allergens: {item.get("allergen_names")}
Spice note: {spice_note}
"""
        item_blocks.append(block)

    # Join all menu item blocks into one context string.
    return "\n".join(item_blocks)


def is_availability_question(query):
    # Detect only direct availability questions.
    # We do NOT include "I want", "I need", or "show me"
    # because those are usually recommendation/preference questions.
    text = query.lower().strip()

    availability_phrases = [
        "do you have",
        "do we have",
        "is there",
        "are there",
        "have you got",
    ]

    return any(text.startswith(phrase) for phrase in availability_phrases)


def extract_requested_item(query):
    # Extract the requested item from direct availability questions.
    # Example: "Do you have pizza?" becomes "pizza".
    text = query.lower()

    # Remove punctuation.
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove common availability question words.
    removable_words = {
        "do",
        "you",
        "we",
        "have",
        "has",
        "got",
        "any",
        "is",
        "there",
        "are",
        "a",
        "an",
        "the",
        "please",
        "menu",
        "item",
        "items",
        "dish",
        "dishes",
        "food",
        "option",
        "options",
    }

    words = text.split()

    cleaned_words = []

    for word in words:
        if word not in removable_words:
            cleaned_words.append(word)

    if not cleaned_words:
        return None

    return " ".join(cleaned_words)


def get_keyword_aliases(keyword):
    # Add English/German aliases for common menu words.
    # This helps "salad" match "salat", "beer" match "bier", etc.
    alias_map = {
        "chicken": ["chicken", "hähnchen", "haehnchen"],
        "fish": ["fish", "fisch", "maschli"],
        "lamb": ["lamb", "lamm"],
        "duck": ["duck", "ente"],
        "shrimp": ["shrimp", "prawn", "garnelen", "jheenga"],
        "salad": ["salad", "salat"],
        "soup": ["soup", "suppe", "shorba"],
        "beer": ["beer", "bier", "kingfisher", "hefeweizen"],
        "rice": ["rice", "reis", "biryani"],
        "bread": ["bread", "naan", "roti", "paratha"],
        "egg": ["egg", "ei"],
        "eggs": ["egg", "eggs", "ei"],
    }

    return alias_map.get(keyword, [keyword])


def requested_item_exists_in_results(requested_item, filtered_items):
    # Check whether the requested item actually appears in the retrieved menu items.
    # This prevents wrong answers like:
    # pizza -> naan
    # burger -> paneer
    # beef -> lamb
    if not requested_item:
        return True

    requested_words = requested_item.split()

    for item in filtered_items:
        # General item text checks the main menu fields.
        item_text = " ".join(
            [
                str(item.get("name", "")),
                str(item.get("description_de", "")),
                str(item.get("category", "")),
                str(item.get("meal_type", "")),
            ]
        ).lower()

        # Strong item text is stricter.
        # This is used for egg so we do not treat "egg" as a dish
        # just because it appears as an allergen somewhere.
        strong_item_text = " ".join(
            [
                str(item.get("name", "")),
                str(item.get("category", "")),
                str(item.get("meal_type", "")),
            ]
        ).lower()

        word_matches = []

        for word in requested_words:
            aliases = get_keyword_aliases(word)

            if word in ["egg", "eggs", "ei"]:
                word_matches.append(any(alias in strong_item_text for alias in aliases))
            else:
                word_matches.append(any(alias in item_text for alias in aliases))

        if all(word_matches):
            return True

    return False


def build_not_found_message(requested_item):
    # Create a clear message when the requested item is not found in the menu.
    return (
        f"I could not find {requested_item} on the menu. "
        "Please ask about another dish, drink, allergy, or food preference from the Redi menu."
    )


def build_prompt(query, intent, filtered_items):
    # Build the final prompt for the LLM using:
    # 1. Customer question
    # 2. Extracted intent
    # 3. Filtered menu items
    menu_context = format_items_for_prompt(filtered_items, intent)

    prompt = f"""
You are a helpful AI waiter for Redi restaurant.

Answer the customer in clear English.

Use only the menu items provided below.
Do not invent dishes, prices, ingredients, or allergens.
Do not suggest unrelated alternatives unless the customer asks for alternatives.

Important rules:
- The menu items below already passed structured filters.
- If max_price is 12, prices like 11.9 or 11.90 are within budget.
- Do not reject an item unless it violates the extracted intent.
- Do not reject milk_lactose unless the customer asks to avoid milk, dairy, or lactose.
- Do not make claims about kitchen cross-contamination unless it is explicitly provided.
- For drinks, do not mention spice notes unless the menu explicitly says something about spice.
- If allergens are "not_listed", say allergen information is not clearly listed and the customer should confirm with the restaurant.
- If a German description is provided, translate or summarize it in English.
- If the user asks for mild food, use the spice note carefully.

Customer question:
{query}

Extracted customer intent:
{intent}

Menu items:
{menu_context}

Write a friendly answer with:
- best matching options
- price
- short English description
- allergen information
- spice warning only if needed
- final safety note for allergies
"""
    return prompt


def generate_llm_answer(query):
    # Retrieve matching menu items using retriever.py.
    intent, filtered_items = retrieve_filtered_items(query, top_k=100)

    # Automatic availability check.
    # This only applies to direct questions like:
    # "Do you have pizza?"
    # "Is there sushi?"
    # It does NOT apply to:
    # "I want mild chicken under 12 euros"
    # "Show me vegetarian lunch menu items"
    if is_availability_question(query):
        requested_item = extract_requested_item(query)

        if requested_item and not requested_item_exists_in_results(requested_item, filtered_items):
            return build_not_found_message(requested_item)

    # Egg special case:
    # The restaurant menu does not contain egg-based dishes.
    # This prevents the LLM from suggesting unrelated alternatives.
    if intent.get("require_egg"):
        return (
            "I could not find any egg-based dishes on the menu. "
            "Please confirm with the restaurant if you are looking for egg-based items."
        )

    # Send only the best 5 items to the LLM to avoid token limit errors.
    filtered_items = filtered_items[:5]

    # If no matching items were found, return a safe fallback message.
    if not filtered_items:
        return (
            "I could not find that item on the menu. "
            "Please ask about another dish, drink, allergy, or food preference from the Redi menu."
        )

    # Build the prompt using the customer query and filtered menu items.
    prompt = build_prompt(query, intent, filtered_items)

    # Create the Groq LLM client.
    # Groq is the API provider, and llama-3.1-8b-instant is the model.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    # Send the prompt to the LLM.
    response = llm.invoke(prompt)

    # Return the generated answer text.
    return response.content


def main():
    # Test question.
    query = "Do you have pizza?"

    # Generate and print the final customer-friendly answer.
    answer = generate_llm_answer(query)
    print(answer)


if __name__ == "__main__":
    main()