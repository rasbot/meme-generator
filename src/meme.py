import os
import random
import argparse

from QuoteModel.quote_engine import Ingestor, QuoteModel
from MemeEngine.meme_engine import MemeEngine


def generate_meme(path=None, body=None, author=None):
    """ Generate a meme given a path and a quote """
    img = None
    quote = None

    if path is None:
        image_path = "./_data/photos/dog/"
        imgs = []
        for root, dirs, files in os.walk(image_path):
            imgs = [os.path.join(root, name) for name in files]

        img = random.choice(imgs)
    else:
        if not os.path.isdir(path):
            raise OSError("Directory path does not exist.")
        imgs = []
        for root, dirs, files in os.walk(path):
            imgs = [os.path.join(root, name) for name in files]

        img = random.choice(imgs)

    if body is None:
        quote_files = ['./_data/DogQuotes/DogQuotesTXT.txt',
                       './_data/DogQuotes/DogQuotesDOCX.docx',
                       './_data/DogQuotes/DogQuotesPDF.pdf',
                       './_data/DogQuotes/DogQuotesCSV.csv']
        quotes = []
        for f in quote_files:
            quotes.extend(Ingestor.parse(f))

        quote = random.choice(quotes)
    else:
        if author is None:
            raise Exception('Author Required if Body is Used')
        quote = QuoteModel(body, author)

    meme = MemeEngine('./tmp')
    path = meme.make_meme(img, quote.body, quote.author)
    return path


if __name__ == "__main__":
    # @TODO Use ArgumentParser to parse the following CLI arguments
    # path - path to an image file
    # body - quote body to add to the image
    # author - quote author to add to the image
    parser = argparse.ArgumentParser(description='Create a meme using a photo, a quote, and an author.')
    parser.add_argument('--path', type=str, default=None, help='path to image file')
    parser.add_argument('--body', type=str, default=None, help='quote to add to image')
    parser.add_argument('--author', type=str, default=None, help='author of quote to add to image')
    
    args = parser.parse_args()
    # print(generate_meme("./_data/photos/cat/", "I'm a kitty", "Mr. Snickers"))
    print(generate_meme(args.path, args.body, args.author))
