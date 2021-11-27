"""Flask script to create html interface for meme generator.

Ingests all quotes and images for each animal type.

Uses Flask to create an interactive web interface for
generating memes."""
import random
import os
from typing import List
import requests
from flask import Flask, render_template, request

from QuoteModel.quote_engine import Ingestor, QuoteModel
from MemeEngine.meme_engine import MemeEngine

APP = Flask(__name__)

MEME = MemeEngine("./static")


def get_quotes(animal: str, quote_type: str) -> List[QuoteModel]:
    """Ingest all quotes for a given animal and quote type.

    Animals can be either dog or cat, and quote types can
    be normal (human generated animal quotes) or gpt-2 quotes
    which were generated using the gpt-2 transformer model.

    Args:
        animal (str): Animal type (dog or cat).
        quote_type (str): Quote type (normal or gpt2).

    Returns:
        List[QuoteModel]: List of `QuoteModel` objects.
    """
    paths = [
        f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesTXT.txt",
        f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesDOCX.docx",
        f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesPDF.pdf",
        f"./_data/{animal}Quotes/{quote_type}/{animal}QuotesCSV.csv",
    ]

    quotes = []
    for file_path in paths:
        quotes.extend(Ingestor.parse(file_path))

    return quotes


def get_images(image_path: str) -> List[str]:
    """Get a list of image file paths in a directory.

    Args:
        image_path (str): Directory path with images.

    Returns:
        List[str]: List of image file paths.
    """
    imgs = []
    for root, _, files in os.walk(image_path):
        imgs = [os.path.join(root, name) for name in files]
    return imgs


def load_data(animal="dog", quote_type="normal") -> List[QuoteModel]:
    """Load all quotes and images for a type of animal and quote
        type.

    Args:
        animal (str, optional): Animal type. Either 'cat' or 'dog.
            Defaults to 'dog'.
        quote_type (str, optional): Quote type. Either 'normal' or
            'gpt2'. Defaults to 'normal'.

    Returns:
        List[QuoteModel]: List of `QuoteModel` objects.
    """
    quote_dict = {
        "dog": {
            "normal": get_quotes("Dog", "normal"),
            "gpt2": get_quotes("Dog", "gpt2"),
        },
        "cat": {
            "normal": get_quotes("Cat", "normal"),
            "gpt2": get_quotes("Cat", "gpt2"),
        },
    }

    dog_images_path = "./_data/photos/dog/"
    cat_images_path = "./_data/photos/cat/"

    image_dict = {
        "dog": get_images(dog_images_path),
        "cat": get_images(cat_images_path),
    }

    imgs = image_dict[animal]
    quotes = quote_dict[animal][quote_type]

    return imgs, quotes


@APP.route("/", methods=["GET", "POST"])
def meme_rand():
    """Generate a random meme of a dog or a cat.

    Use either normal or machine learning generated quotes.
    """
    use_cat = False
    use_gpt2 = False
    if request.method == "POST":
        use_cat = request.form.get("use cat")
        use_gpt2 = request.form.get("use machine learning quotes")
    animal = "dog"
    quote_type = "normal"
    if use_cat:
        animal = "cat"
    if use_gpt2:
        quote_type = "gpt2"
    imgs, quotes = load_data(animal, quote_type)
    img = random.choice(imgs)
    quote = random.choice(quotes)

    path = MEME.make_meme(img, quote.body, quote.author)
    return render_template("meme.html", path=path)


@APP.route("/create", methods=["GET"])
def meme_form():
    """User input for meme information."""
    return render_template("meme_form.html")


@APP.route("/create", methods=["POST"])
def meme_post():
    """Create a user defined meme."""
    image_url = request.form["image_url"]
    img_file = f"./static/{random.randint(0,100000)}.png"

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(img_file, "wb") as out_file:
            out_file.write(response.content)

    body = request.form["body"]
    author = request.form["author"]

    path = MEME.make_meme(img_file, body, author)

    os.remove(img_file)

    return render_template("meme.html", path=path)


if __name__ == "__main__":
    APP.run()
