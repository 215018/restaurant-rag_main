# This script handles retrieval and structured filtering for the restaurant RAG system.
# It uses CSV first because CSV has structured fields like price, allergens, and meal type.
# If CSV has no matching item, it falls back to PDF chunks only for non-strict menu questions.

from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embedding_model


# Go up from src/retriever.py to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder where ChromaDB stores the vector database.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def load_vector_store():
    # Load the same Hugging Face embedding model used when creating ChromaDB.
    embedding_model = get_embedding_model()

    # Load the existing Chroma vector database from disk.
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )

    return vector_store


def expand_query_for_retrieval(query):
    # Add related English and German words to improve retrieval.
    # Example: "beer" also searches for "bier", "kingfisher", and "hefeweizen".
    text = query.lower()
    extra_words = []

    if "beer" in text or "bier" in text:
        extra_words.append("bier beer indisches bier kingfisher hefeweizen")

    if "fish" in text or "fisch" in text:
        extra_words.append("fish fisch fish curry fischgerichte")

    if "lamb" in text or "lamp" in text or "lamm" in text:
        extra_words.append("lamb lamm lamb curry lammgerichte")

    if "shrimp" in text or "prawn" in text or "garnelen" in text:
        extra_words.append("shrimp prawn garnelen")

    if "egg" in text or "ei" in text:
        extra_words.append("egg ei egg dishes")

    if "salad" in text or "salat" in text:
        extra_words.append(
            "salad salat gemischter salat chicken tikka salat paneer tikka salat salat der saison"
        )

    if "soup" in text or "suppe" in text or "shorba" in text:
        extra_words.append("soup suppe shorba")

    if "naan" in text or "roti" in text or "bread" in text:
        extra_words.append("naan roti paratha bread")

    if "rice" in text or "reis" in text or "biryani" in text:
        extra_words.append("rice reis biryani")

    if "lassi" in text:
        extra_words.append("lassi mango lassi vegan lassi")

    if "dessert" in text or "sweet" in text or "sweets" in text:
        extra_words.append("dessert sweet mango cream gulab")

    if not extra_words:
        return query

    return query + " " + " ".join(extra_words)


def retrieve_relevant_chunks(query, top_k=100):
    # Expand the query before searching ChromaDB.
    expanded_query = expand_query_for_retrieval(query)

    # Load ChromaDB.
    vector_store = load_vector_store()

    # Search for the most relevant menu chunks.
    results = vector_store.similarity_search(
        expanded_query,
        k=top_k,
    )

    return results


def parse_bool(value):
    # Convert text like "true" or "false" into a Python boolean.
    text = str(value).strip().lower()
    return text == "true"


def parse_price(value):
    # Convert price text into a float.
    # Example: "11,90" becomes 11.90.
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def document_to_item(doc):
    # Convert one CSV LangChain Document into a Python dictionary.
    # Example document text:
    # name: Butter Chicken
    # price: 11.9
    # allergen_names: milk_lactose
    item = {}

    for line in doc.page_content.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            item[key.strip()] = value.strip()

    return item


def extract_matching_pdf_line(page_content, query):
    # Find the most relevant line from a PDF chunk.
    query_text = query.lower()

    if "salad" in query_text or "salat" in query_text:
        keywords = [
            "gemischter salat",
            "paneer tikka salat",
            "chicken tikka salat",
            "salat der saison",
            "salat",
        ]
    elif "soup" in query_text or "suppe" in query_text or "shorba" in query_text:
        keywords = ["suppe", "soup", "shorba"]
    elif "dessert" in query_text or "sweet" in query_text or "sweets" in query_text:
        keywords = ["dessert", "mango cream", "gulab"]
    else:
        keywords = query_text.split()

    lines = []

    for line in page_content.splitlines():
        clean_line = line.strip()

        if clean_line:
            lines.append(clean_line)

    for keyword in keywords:
        for line in lines:
            if keyword in line.lower():
                return line

    return page_content[:800]


def clean_pdf_item_name(name):
    # Clean item names extracted from PDF text.
    name = name.strip()

    # Remove menu numbers like "5. Gemischter Salat".
    if ". " in name:
        first_part, rest = name.split(". ", 1)

        if first_part.isdigit():
            name = rest.strip()

    # Remove price at the end.
    # Example: "Gemischter Salat 6,00" becomes "Gemischter Salat".
    parts = name.split()

    if parts:
        last_part = parts[-1]

        if "," in last_part and last_part.replace(",", "").isdigit():
            name = " ".join(parts[:-1])

    return name.strip()


def extract_pdf_price(text):
    # Extract price from a PDF line.
    # Example: "Gemischter Salat 6,00" becomes "6.00".
    parts = text.strip().split()

    if not parts:
        return "not listed"

    last_part = parts[-1]

    if "," in last_part and last_part.replace(",", "").isdigit():
        return last_part.replace(",", ".")

    return "not listed"


def translate_pdf_description(text):
    # Create simple English descriptions for PDF fallback items.
    lower_text = text.lower()

    if "gemischter salat" in lower_text:
        return "Mixed salad."

    if "paneer tikka salat" in lower_text:
        return "Salad served with paneer tikka."

    if "chicken tikka salat" in lower_text:
        return "Salad served with chicken tikka."

    if "salat der saison" in lower_text:
        return "Seasonal salad."

    if "zu jedem hauptgericht" in lower_text:
        return "A seasonal salad is served as a side dish with each main course when dining in."

    if "suppe" in lower_text or "shorba" in lower_text:
        return "Soup item from the menu."

    if "mango cream" in lower_text:
        return "Mango cream dessert."

    if "gulab" in lower_text:
        return "Sweet Indian dessert."

    return (
        "This item was found in the PDF menu, but the full English description is not clearly available. "
        "Please confirm details with the restaurant."
    )


def is_clean_pdf_item_name(name):
    # Reject messy PDF chunks that contain many menu items together.
    if not name:
        return False

    if "\n" in name:
        return False

    if len(name) > 80:
        return False

    digit_groups = 0

    for word in name.split():
        clean_word = word.replace(".", "").replace(",", "")

        if clean_word.isdigit():
            digit_groups += 1

    if digit_groups > 2:
        return False

    return True


def pdf_document_to_item(doc, query):
    # Convert one PDF chunk into a menu-item-like dictionary.
    matched_text = extract_matching_pdf_line(doc.page_content, query)

    name = clean_pdf_item_name(matched_text[:80])
    price = extract_pdf_price(matched_text)
    english_description = translate_pdf_description(matched_text)

    return {
        "name": name,
        "price": price,
        "description_de": english_description,
        "category": "pdf_menu",
        "meal_type": "unknown",
        "allergen_names": "not_listed",
        "source_type": "pdf_menu",
    }


def extract_user_intent(query):
    # Extract structured meaning from the customer question.
    text = query.lower()

    intent = {
        "max_price": None,
        "require_chicken": False,
        "require_fish": False,
        "require_lamb": False,
        "require_duck": False,
        "require_shrimp": False,
        "require_paneer": False,
        "require_egg": False,
        "require_salad": False,
        "require_soup": False,
        "require_bread": False,
        "require_rice": False,
        "require_lassi": False,
        "require_dessert": False,
        "require_vegetarian": False,
        "require_vegan": False,
        "require_lunch_menu": False,
        "require_beer": False,
        "meal_type": None,
        "exclude_drinks": False,
        "avoid_allergens": [],
        "taste": None,
    }

    # Extract budget.
    words = text.replace("€", " euros").split()

    for index, word in enumerate(words):
        if word in ["under", "below", "less", "max", "maximum"]:
            for next_word in words[index + 1:index + 4]:
                try:
                    intent["max_price"] = float(next_word)
                    break
                except ValueError:
                    continue

    # Detect allergy-related words.
    allergy_signals = ["no", "avoid", "allergic", "allergy", "without"]

    # Extract food preferences.
    if "chicken" in text:
        intent["require_chicken"] = True

    if "fish" in text or "fisch" in text:
        intent["require_fish"] = True

    if "lamb" in text or "lamp" in text or "lamm" in text:
        intent["require_lamb"] = True

    if "duck" in text or "ente" in text:
        intent["require_duck"] = True

    if "shrimp" in text or "prawn" in text or "garnelen" in text:
        intent["require_shrimp"] = True

    if "paneer" in text:
        intent["require_paneer"] = True

    # Egg can mean two different things:
    # 1. "Do you have egg dishes?" means the user wants egg.
    # 2. "I am allergic to egg" means the user wants to avoid egg.
    # So we only set require_egg when the sentence is not an allergy/avoidance request.
    if ("egg" in text or "ei" in text) and not any(signal in text for signal in allergy_signals):
        intent["require_egg"] = True

    if "salad" in text or "salat" in text:
        intent["require_salad"] = True

    if "soup" in text or "suppe" in text or "shorba" in text:
        intent["require_soup"] = True

    if "naan" in text or "roti" in text or "bread" in text:
        intent["require_bread"] = True

    if "rice" in text or "reis" in text or "biryani" in text:
        intent["require_rice"] = True

    if "lassi" in text:
        intent["require_lassi"] = True

    if "dessert" in text or "sweet" in text or "sweets" in text:
        intent["require_dessert"] = True

    # Extract dietary preferences.
    if "vegetarian" in text or "veg" in text:
        intent["require_vegetarian"] = True

    if "vegan" in text:
        intent["require_vegan"] = True

    if "lunch" in text or "mittag" in text or "mittagsmenü" in text:
        intent["require_lunch_menu"] = True

    # If the user asks to eat food, remove drinks unless the user asks for drinks.
    if "eat" in text or "food" in text or "dish" in text or "meal" in text:
        intent["exclude_drinks"] = True

    # Detect drink questions.
    if (
        "drink" in text
        or "drinks" in text
        or "lassi" in text
        or "beer" in text
        or "bier" in text
        or "kingfisher" in text
        or "hefeweizen" in text
        or "alcohol" in text
    ):
        intent["meal_type"] = "drink"

    # Detect beer specifically.
    if "beer" in text or "bier" in text or "kingfisher" in text or "hefeweizen" in text:
        intent["require_beer"] = True

    # Detect starter/main meal type.
    if "starter" in text or "appetizer" in text or "pakora" in text:
        intent["meal_type"] = "starter"

    if "main" in text or "main dish" in text or "meal" in text:
        intent["meal_type"] = "main"

    # Map allergy words to normalized allergen names.
    allergen_keywords = {
        "nuts": "nuts",
        "nut": "nuts",
        "milk": "milk_lactose",
        "dairy": "milk_lactose",
        "lactose": "milk_lactose",
        "gluten": "gluten",
        "fish": "fish",
        "egg": "egg",
        "soy": "soy",
        "mustard": "mustard",
    }

    for keyword, allergen_name in allergen_keywords.items():
        if keyword in text and any(signal in text for signal in allergy_signals):
            if allergen_name not in intent["avoid_allergens"]:
                intent["avoid_allergens"].append(allergen_name)

    # Extract spice preference.
    if "mild" in text or "not spicy" in text:
        intent["taste"] = "mild"

    if "spicy" in text or "hot" in text:
        intent["taste"] = "spicy"

    return intent


def item_contains_avoided_allergen(item, avoid_allergens):
    # Check if a menu item contains an allergen the user wants to avoid.
    allergen_names = str(item.get("allergen_names", "")).lower()

    item_text = " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description_de", "")),
            str(item.get("category", "")),
        ]
    ).lower()

    allergen_keyword_map = {
        "milk_lactose": [
            "milk",
            "lactose",
            "butter",
            "cream",
            "paneer",
            "cheese",
            "yogurt",
            "sahne",
            "käse",
            "kaese",
            "milch",
            "lassi",
        ],
        "nuts": [
            "nuts",
            "nut",
            "cashew",
            "cashewn",
            "almond",
            "mandel",
            "nüsse",
            "nuesse",
        ],
        "gluten": [
            "gluten",
            "wheat",
            "weizen",
            "naan",
            "roti",
            "paratha",
            "noodles",
            "nudeln",
        ],
        "egg": ["egg", "ei"],
        "soy": ["soy", "soja"],
        "fish": ["fish", "fisch"],
        "mustard": ["mustard", "senf"],
    }

    for allergen in avoid_allergens:
        allergen = allergen.lower()

        if allergen in allergen_names:
            return True

        for keyword in allergen_keyword_map.get(allergen, []):
            if keyword in item_text:
                return True

    return False


def item_matches_keywords(item, keywords):
    # Check if an item contains any keyword in important menu fields.
    item_text = " ".join(
        [
            str(item.get("name", "")),
            str(item.get("description_de", "")),
            str(item.get("category", "")),
            str(item.get("meal_type", "")),
            str(item.get("allergen_names", "")),
        ]
    ).lower()

    return any(keyword in item_text for keyword in keywords)


def item_is_beer(item):
    # Special helper for beer questions.
    return item_matches_keywords(item, ["beer", "bier", "kingfisher", "hefeweizen"])


def filter_menu_items(
    documents,
    max_price=None,
    require_chicken=False,
    require_fish=False,
    require_lamb=False,
    require_duck=False,
    require_shrimp=False,
    require_paneer=False,
    require_egg=False,
    require_salad=False,
    require_soup=False,
    require_bread=False,
    require_rice=False,
    require_lassi=False,
    require_dessert=False,
    require_vegetarian=False,
    require_vegan=False,
    require_lunch_menu=False,
    require_beer=False,
    meal_type=None,
    exclude_drinks=False,
    avoid_allergens=None,
):
    # Apply structured filters to CSV menu items.
    filtered = []
    seen_names = set()

    if avoid_allergens is None:
        avoid_allergens = []

    for doc in documents:
        # Use CSV first because it has structured fields.
        if doc.metadata.get("source_type") != "csv_menu":
            continue

        item = document_to_item(doc)
        price = parse_price(item.get("price"))

        # Price filter.
        if max_price is not None:
            if price is None or price > max_price:
                continue

        # Ingredient filters.
        if require_chicken and not parse_bool(item.get("contains_chicken")):
            continue

        if require_fish and not parse_bool(item.get("contains_fish")):
            continue

        if require_lamb and not parse_bool(item.get("contains_lamb")):
            continue

        if require_duck and not parse_bool(item.get("contains_duck")):
            continue

        if require_shrimp and not parse_bool(item.get("contains_shrimp")):
            continue

        if require_paneer and not parse_bool(item.get("contains_paneer")):
            continue

        # Egg is not a main ingredient column in the CSV.
        # So if the user asks for egg dishes, we only keep items that mention egg/ei.
        # If your restaurant has no egg dishes, this will correctly return no items.
        if require_egg and not item_matches_keywords(item, ["egg", "ei"]):
            continue

        # Category keyword filters.
        if require_salad and not item_matches_keywords(item, ["salad", "salat"]):
            continue

        if require_soup and not item_matches_keywords(item, ["soup", "suppe", "shorba"]):
            continue

        if require_bread and not item_matches_keywords(item, ["naan", "roti", "paratha", "bread"]):
            continue

        if require_rice and not item_matches_keywords(item, ["rice", "reis", "biryani"]):
            continue

        if require_lassi and not item_matches_keywords(item, ["lassi"]):
            continue

        if require_dessert and not item_matches_keywords(
            item, ["dessert", "sweet", "mango cream", "gulab"]
        ):
            continue

        # Dietary filters.
        if require_vegetarian and not parse_bool(item.get("is_vegetarian")):
            continue

        if require_vegan and not parse_bool(item.get("is_vegan_possible")):
            continue

        if require_lunch_menu and not parse_bool(item.get("is_lunch_menu")):
            continue

        # Meal type filter.
        if meal_type is not None:
            if item.get("meal_type", "").lower() != meal_type:
                continue

        # If user wants food, remove drinks.
        if exclude_drinks:
            if item.get("meal_type", "").lower() == "drink":
                continue

        # Beer filter.
        if require_beer and not item_is_beer(item):
            continue

        # Allergy safety filter.
        if item_contains_avoided_allergen(item, avoid_allergens):
            continue

        # Remove duplicate item names.
        name = item.get("name")

        if name in seen_names:
            continue

        seen_names.add(name)
        filtered.append(item)

    # Keep prompt small for the LLM.
    return filtered[:5]


def describe_spice_safety(item, taste):
    # Create a warning if the user asks for mild food but the description sounds spicy.
    description = item.get("description_de", "").lower()
    meal_type = item.get("meal_type", "").lower()

    if meal_type == "drink":
        return ""

    spicy_words = ["scharf", "würzig", "pikant", "chili", "spicy", "würziger"]

    if taste == "mild" and any(word in description for word in spicy_words):
        return "This may not be mild because the description suggests it is spicy or strongly seasoned."

    if taste == "mild":
        return "The menu does not clearly list spice level, so please confirm if you need it very mild."

    return ""


def retrieve_filtered_items(query, top_k=100):
    # Main function used by rag_chain.py.
    # It retrieves menu chunks, extracts intent, filters CSV rows,
    # and falls back to PDF only when the request is not strict.
    intent = extract_user_intent(query)

    retrieved_docs = retrieve_relevant_chunks(query, top_k=top_k)

    filtered_items = filter_menu_items(
        retrieved_docs,
        max_price=intent["max_price"],
        require_chicken=intent["require_chicken"],
        require_fish=intent["require_fish"],
        require_lamb=intent["require_lamb"],
        require_duck=intent["require_duck"],
        require_shrimp=intent["require_shrimp"],
        require_paneer=intent["require_paneer"],
        require_egg=intent["require_egg"],
        require_salad=intent["require_salad"],
        require_soup=intent["require_soup"],
        require_bread=intent["require_bread"],
        require_rice=intent["require_rice"],
        require_lassi=intent["require_lassi"],
        require_dessert=intent["require_dessert"],
        require_vegetarian=intent["require_vegetarian"],
        require_vegan=intent["require_vegan"],
        require_lunch_menu=intent["require_lunch_menu"],
        require_beer=intent["require_beer"],
        meal_type=intent["meal_type"],
        exclude_drinks=intent["exclude_drinks"],
        avoid_allergens=intent["avoid_allergens"],
    )

    # If CSV found matching items, return them.
    if filtered_items:
        return intent, filtered_items

    # If the user asked for strict filters, do not use PDF fallback.
    # This prevents wrong answers like showing chicken for "egg dishes".
    strict_filter_used = (
        intent["max_price"] is not None
        or intent["avoid_allergens"]
        or intent["require_vegetarian"]
        or intent["require_vegan"]
        or intent["require_lunch_menu"]
        or intent["require_chicken"]
        or intent["require_fish"]
        or intent["require_lamb"]
        or intent["require_duck"]
        or intent["require_shrimp"]
        or intent["require_paneer"]
        or intent["require_egg"]
        or intent["require_beer"]
    )

    if strict_filter_used:
        return intent, []

    # PDF fallback is only for non-strict category questions.
    # Example: "Do you have any salad?"
    pdf_items = []
    seen_pdf_names = set()

    for doc in retrieved_docs:
        if doc.metadata.get("source_type") == "pdf_menu":
            pdf_item = pdf_document_to_item(doc, query)
            name = pdf_item.get("name", "")

            if not is_clean_pdf_item_name(name):
                continue

            if intent["require_salad"]:
                lower_name = name.lower()

                if "salat" not in lower_name and "salad" not in lower_name:
                    continue

            if name in seen_pdf_names:
                continue

            seen_pdf_names.add(name)
            pdf_items.append(pdf_item)

    return intent, pdf_items[:3]


def main():
    # Test this file directly.
    query = "Do you have egg dishes?"

    intent, filtered_items = retrieve_filtered_items(query, top_k=100)

    print(f"User query: {query}")
    print("Extracted intent:")
    print(intent)
    print("=" * 50)

    print("Filtered menu items:")
    print("=" * 50)

    if not filtered_items:
        print("No matching menu items found.")

    for item in filtered_items:
        spice_note = describe_spice_safety(item, intent["taste"])

        print(f"Name: {item.get('name')}")
        print(f"Price: {item.get('price')} euros")
        print(f"Description: {item.get('description_de')}")
        print(f"Allergens: {item.get('allergen_names')}")

        if spice_note:
            print(f"Spice note: {spice_note}")

        print("-" * 50)


if __name__ == "__main__":
    main()