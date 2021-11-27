"""Meme module used to generate memes locally.

Create a meme using dog or cat pictures, along
with normally generated (human) quotes, or machine
learning generated (gpt-2 model) quotes. Optionally,
create memes with custom text and authors. Saves file
to a local directory."""
import os
import random
import argparse

from QuoteModel.quote_engine import Ingestor, QuoteModel
from MemeEngine.meme_engine import MemeEngine


def generate_meme(
        path: str = "./tmp",
        animal_type: str = "dog",
        body: str = "",
        author: str = "",
        quote_type: str = "normal",
) -> str:
    """Generate a random meme using custom or ingested quotes.

    Uses a random image of a dog or a cat, and either a random
    quote / author, or a user-defined quote. Saves image to a
    local directory.

    Args:
        path (str, optional): Path to save meme to.
        animal_type (str, optional): Type of animal (dog or cat).
            Defaults to 'dog'.
        body (str, optional): Quote string. Defaults to "".
        author (str, optional): Author string. Defaults to "".
        quote_type (str, optional): Type of quote (normal or gpt2).
            Defaults to 'normal'.

    Raises:
        Exception: Raise exception if body is provided but
            author is not provided.

    Returns:
        str: Path to meme generated.
    """
    imgs = []
    for root, _, files in os.walk(f"./_data/photos/{animal_type}/"):
        imgs = [os.path.join(root, name) for name in files]

    img = random.choice(imgs)

    if not body:
        if animal_type == "dog":
            animal = "Dog"
        elif animal_type == "cat":
            animal = "Cat"
        else:
            raise Exception("Animal type not supported. Use dog or cat.")
        quote_files = [
            f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesTXT.txt",
            f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesDOCX.docx",
            f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesPDF.pdf",
            f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesCSV.csv",
        ]
        quotes = []
        for quote_file in quote_files:
            quotes.extend(Ingestor.parse(quote_file))

        quote = random.choice(quotes)
    else:
        if not author:
            raise Exception("Author Required if Body is Used")
        quote = QuoteModel(body, author)

    return MemeEngine(path).make_meme(img, quote.body, quote.author)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Create a meme using a photo, a quote, and an author."
    )
    PARSER.add_argument(
        "--path",
        type=str,
        default="",
        help="path to image file"
    )
    PARSER.add_argument(
        "--animal",
        type=str,
        default="dog",
        help="animal used for meme (dog or cat)"
    )
    PARSER.add_argument(
        "--type",
        type=str,
        default="normal",
        help="type of quote to use (normal or gpt2)",
    )
    PARSER.add_argument(
        "--body",
        type=str,
        default="",
        help="quote to add to image"
    )
    PARSER.add_argument(
        "--author",
        type=str,
        default="",
        help="author of quote to add to image"
    )

    ARGS = PARSER.parse_args()
    print(
        generate_meme(
            ARGS.path,
            ARGS.animal,
            ARGS.body,
            ARGS.author,
            ARGS.type
        )
    )
