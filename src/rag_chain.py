# This script implements the RAG chain for the restaurant menu chatbot.
# RAG means Retrieval-Augmented Generation:
# 1. Retrieve matching menu items from retriever.py.
# 2. Send only the best few menu items to the LLM.
# 3. Generate a friendly customer answer in English.

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

# Format the filtered menu items into a text block for the LLM prompt.
def format_items_for_prompt(filtered_items, intent):
    # Store formatted menu item text blocks here.
    item_blocks = []

    # Convert each menu item dictionary into a short text block for the LLM.
    for item in filtered_items:
        # Create a spice safety note based on the customer taste request.
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

    # Join all item blocks into one prompt context.
    return "\n".join(item_blocks)

# Build the final prompt for the LLM using the customer query, extracted intent, and filtered menu items.
def build_prompt(query, intent, filtered_items):
    # Convert filtered menu items into text for the LLM.
    menu_context = format_items_for_prompt(filtered_items, intent)

    # Create the final instruction prompt for Groq/Llama.
    prompt = f"""
You are a helpful AI waiter for Redi restaurant.

Answer the customer in clear English.

Use only the menu items provided below.
Do not invent dishes, prices, ingredients, or allergens.

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

# Retrieve matching menu items. Then build the prompt and generate the final customer-friendly answer using the Groq LLM.
def generate_llm_answer(query):
    # Retrieve matching menu items.
    # top_k=40 keeps retrieval useful but avoids sending too much context.
    intent, filtered_items = retrieve_filtered_items(query, top_k=100)

    # Send only the best 5 items to the LLM to avoid Groq token limit errors.
    filtered_items = filtered_items[:5]

    # If no matching items were found, return a safe fallback message.
    if not filtered_items:
        return (
            "I could not find a matching menu item. "
            "Please contact the restaurant directly, especially for allergy-sensitive requests."
        )

    # Build the prompt using the customer query and filtered menu items.
    prompt = build_prompt(query, intent, filtered_items)

    # Create the Groq LLM client.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    # Send the prompt to the LLM.
    response = llm.invoke(prompt)

    # Return the generated answer text.
    return response.content

# Main function to test the RAG chain with an example customer query.
def main():
    # Example question for testing.
    query = "I want to eat Fish"

    # Generate and print the final customer-friendly answer.
    answer = generate_llm_answer(query)
    print(answer)


if __name__ == "__main__":
    main()