import os
import random
import argparse

from QuoteModel.quote_engine import Ingestor, QuoteModel
from MemeEngine.meme_engine import MemeEngine


def generate_meme(path='dog', body=None, author=None):
    """ Generate a meme given a path and a quote """

    if path == 'dog':
        image_path = "./_data/photos/dog/"
    else:
        image_path = "./_data/photos/cat/"
    imgs = []
    for root, dirs, files in os.walk(image_path):
        imgs = [os.path.join(root, name) for name in files]

    img = random.choice(imgs)

    if body is None:
        if author == 'gpt2':
            folder = 'gpt2'
        else:
            folder = 'normal'
        if path == 'dog':
            animal = 'Dog'
        else:
            animal = 'Cat'
        quote_files = [f'./_data/{animal}Quotes/{folder}/{animal}QuotesTXT.txt',
                    f'./_data/{animal}Quotes/{folder}/{animal}QuotesDOCX.docx',
                    f'./_data/{animal}Quotes/{folder}/{animal}QuotesPDF.pdf',
                    f'./_data/{animal}Quotes/{folder}/{animal}QuotesCSV.csv']
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
    # @TODO Fix 'path' variable info
    # text still spilling off edge of image
    parser = argparse.ArgumentParser(description='Create a meme using a photo, a quote, and an author.')
    parser.add_argument('--path', type=str, default=None, help='path to image file')
    parser.add_argument('--body', type=str, default=None, help='quote to add to image')
    parser.add_argument('--author', type=str, default=None, help='author of quote to add to image')
    
    args = parser.parse_args()
    # print(generate_meme("./_data/photos/cat/", "I'm a kitty", "Mr. Snickers"))
    print(generate_meme(args.path, args.body, args.author))
