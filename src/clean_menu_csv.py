# This script cleans the raw menu CSV for the restaurant RAG system.
# It normalizes boolean fields, fixes missing descriptions and allergens, and applies rules to identify drinks.
# Import libraries and define file paths.
from pathlib import Path
import pandas as pd

# Go up from src/clean_menu_csv.py to the main restaurant-rag folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw CSV file. This is the input file.
PROJECT_RAW_CSV = PROJECT_ROOT / "data" / "swagat_menu_raw.csv"

# Final cleaned CSV used by the RAG system. This is the output file.
OUTPUT_CSV = PROJECT_ROOT / "data" / "swagat_menu.csv"

# Allergen code mapping from the restaurant menu legend.
ALLERGEN_MAP = {
    "a": "gluten",
    "b": "crustaceans",
    "c": "egg",
    "d": "fish",
    "e": "soy",
    "f": "milk_lactose",
    "g": "nuts",
    "h": "celery",
    "i": "mustard",
    "j": "sesame",
    "k": "sulfur_dioxide_sulfites",
    "l": "lupine",
    "m": "molluscs",
}
# This mapping can be expanded if the menu uses more allergen codes in the future.

# Function to normalize boolean values into "true", "false", or "unknown".
def normalize_bool(value):
    # Convert the value to lowercase text.
    text = str(value).strip().lower()

    if text in ["true", "1", "yes"]:
        return "true"

    if text in ["false", "0", "no"]:
        return "false"

    return "unknown"

# Function to normalize allergen codes into a clean, sorted, comma-separated string.
def normalize_allergen_codes(value):
    # Convert allergen codes into clean lowercase text.
    text = str(value).strip().lower()

    # If the menu does not list allergens, mark this clearly.
    if not text:
        return "not_listed"

    # Some PDF/CSV values may use dots instead of commas, like "f.g".
    text = text.replace(".", ",")

    # Split values like "f,g" into ["f", "g"].
    codes = [code.strip() for code in text.split(",") if code.strip()]

    # Remove duplicate codes and sort them.
    unique_codes = sorted(set(codes))

    return ",".join(unique_codes)

# Function to convert allergen codes into readable English names using the ALLERGEN_MAP.
def get_allergen_names(allergen_codes):
    # If no allergens are listed, keep this clear for the chatbot.
    if allergen_codes == "not_listed":
        return "not_listed"

    codes = [code.strip() for code in allergen_codes.split(",")]

    names = []

    # Convert each code into a readable English allergen name.
    for code in codes:
        names.append(ALLERGEN_MAP.get(code, f"unknown_code_{code}"))

    return ", ".join(sorted(set(names)))

# Function to determine if a menu item is a drink based on its name, category, and meal type.
def is_drink(name, category, meal_type):
    text = f"{name} {category} {meal_type}".lower()

    drink_keywords = [
        "getränke",
        "drink",
        "lassi",
        "bier",
        "beer",
        "hefeweizen",
        "kingfisher",
        "cola",
        "fanta",
        "sprite",
        "wasser",
        "saft",
        "juice",
        "chai",
        "tee",
    ]

    return any(keyword in text for keyword in drink_keywords)

# Function to build a comprehensive search text field for each menu item, combining all relevant information.
def build_search_text(row):
    return (
        f"Name: {row['name']} | "
        f"Category: {row['category']} | "
        f"Description: {row['description_de']} | "
        f"Price: {row['price']} euros | "
        f"Meal type: {row['meal_type']} | "
        f"Lunch menu: {row['is_lunch_menu']} | "
        f"Vegetarian: {row['is_vegetarian']} | "
        f"Vegan possible: {row['is_vegan_possible']} | "
        f"Allergen codes: {row['allergen_codes']} | "
        f"Allergen names: {row['allergen_names']}"
    )

# Main function to clean each row of the menu CSV. It applies all normalization and cleaning rules.
def clean_row(row):
    name = str(row["name"]).strip()
    category = str(row["category"]).strip()
    meal_type = str(row["meal_type"]).strip()

    row["name"] = name
    row["category"] = category
    row["meal_type"] = meal_type

    boolean_columns = [
        "is_lunch_menu",
        "is_vegetarian",
        "is_vegan_possible",
        "contains_chicken",
        "contains_lamb",
        "contains_fish",
        "contains_shrimp",
        "contains_duck",
        "contains_paneer",
    ]

    for column in boolean_columns:
        row[column] = normalize_bool(row[column])

    if not str(row["description_de"]).strip():
        row["description_de"] = f"No detailed description is listed for {name}."

    # Normalize allergen codes and create readable names.
    row["allergen_codes"] = normalize_allergen_codes(row["allergen_codes"])
    row["allergen_names"] = get_allergen_names(row["allergen_codes"])

    # Fix all drink rows.
    if is_drink(name, category, meal_type):
        row["category"] = "Getränke"
        row["meal_type"] = "drink"

        row["contains_chicken"] = "false"
        row["contains_lamb"] = "false"
        row["contains_fish"] = "false"
        row["contains_shrimp"] = "false"
        row["contains_duck"] = "false"
        row["contains_paneer"] = "false"

        row["is_vegetarian"] = "true"

    # Kingfisher is beer, not fish.
    if "kingfisher" in name.lower():
        row["category"] = "Getränke"
        row["meal_type"] = "drink"
        row["contains_fish"] = "false"
        row["is_vegetarian"] = "true"
        row["is_vegan_possible"] = "true"
        row["allergen_codes"] = "not_listed"

    # Normal lassi contains milk/lactose.
    if "lassi" in name.lower() and "vegan" not in name.lower():
        row["category"] = "Getränke"
        row["meal_type"] = "drink"
        row["is_vegetarian"] = "true"
        row["is_vegan_possible"] = "false"

        if row["allergen_codes"] == "not_listed":
            row["allergen_codes"] = "f"

    # Vegan lassi does not contain dairy according to the menu.
    if "lassi" in name.lower() and "vegan" in name.lower():
        row["category"] = "Getränke"
        row["meal_type"] = "drink"
        row["is_vegetarian"] = "true"
        row["is_vegan_possible"] = "true"
        row["allergen_codes"] = "not_listed"

        if "No detailed description" in row["description_de"]:
            row["description_de"] = "Mit Kokosmilch"

    # Recalculate allergen names after special fixes.
    row["allergen_codes"] = normalize_allergen_codes(row["allergen_codes"])
    row["allergen_names"] = get_allergen_names(row["allergen_codes"])

    # Rebuild search text after cleaning.
    row["search_text"] = build_search_text(row)

    return row

# Main function to read the raw CSV, clean it, and save the cleaned version for the RAG system.
def main():
    if not PROJECT_RAW_CSV.exists():
        raise FileNotFoundError(
            f"Could not find raw CSV file: {PROJECT_RAW_CSV}"
        )

    menu_df = pd.read_csv(PROJECT_RAW_CSV, keep_default_na=False)

    cleaned_df = menu_df.apply(clean_row, axis=1)

    cleaned_df.to_csv(OUTPUT_CSV, index=False)

    print("Menu CSV cleaned successfully")
    print(f"Input CSV: {PROJECT_RAW_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Total rows: {len(cleaned_df)}")

# Run the main function when this script is executed.
if __name__ == "__main__":
    main()