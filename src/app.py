import random
import os
import requests
from flask import Flask, render_template, abort, request

from QuoteModel.quote_engine import Ingestor, QuoteModel
from MemeEngine.meme_engine import MemeEngine

app = Flask(__name__)

meme = MemeEngine('./static')


def get_quotes(animal, quote_type):
    paths = [f'./_data/{animal}Quotes/{quote_type}/{animal}QuotesTXT.txt',
            f'./_data/{animal}Quotes/{quote_type}/{animal}QuotesDOCX.docx',
            f'./_data/{animal}Quotes/{quote_type}/{animal}QuotesPDF.pdf',
            f'./_data/{animal}Quotes/{quote_type}/{animal}QuotesCSV.csv']
    
    quotes = []
    for f in paths:
        quotes.extend(Ingestor.parse(f))

    return quotes

def get_images(image_path):
    imgs = []
    for root, dirs, files in os.walk(image_path):
        imgs = [os.path.join(root, name) for name in files]
    return imgs

def setup(animal='dog', quote_type='normal'):
    """ Load all resources """

    quote_dict = {
        'dog': {
            'normal': get_quotes("Dog", "normal"),
            'gpt2': get_quotes("Dog", "gpt2")
            },
        'cat': {
            'normal':  get_quotes("Cat", "normal"),
            'gpt2': get_quotes("Cat", "gpt2")
            }
    }

    dog_images_path = "./_data/photos/dog/"
    cat_images_path = "./_data/photos/cat/"

    image_dict = {'dog': get_images(dog_images_path),
                  'cat': get_images(cat_images_path)
                  }

    imgs = image_dict[animal]
    quotes = quote_dict[animal][quote_type]


    return imgs, quotes

@app.route('/', methods=['GET', 'POST'])
def meme_rand():
    """ Generate a random meme """
    use_cat = False
    use_gpt2 = False
    if request.method == 'POST':
        use_cat = request.form.get('use cat')
        use_gpt2 = request.form.get('use machine learning quotes')
    animal = 'dog'
    quote_type = 'normal'
    if use_cat:
        animal = 'cat'
    if use_gpt2:
        quote_type = 'gpt2'
    imgs, quotes = setup(animal, quote_type)
    img = random.choice(imgs)
    quote = random.choice(quotes)

    path = meme.make_meme(img, quote.body, quote.author)
    return render_template('meme.html', path=path)


@app.route('/create', methods=['GET'])
def meme_form():
    """ User input for meme information """
    return render_template('meme_form.html')


@app.route('/create', methods=['POST'])
def meme_post():
    """ Create a user defined meme """
    image_url = request.form['image_url']
    img_file = f'./static/{random.randint(0,100000)}.png'

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(img_file, 'wb') as f:
            f.write(response.content)

    body = request.form['body']
    author = request.form['author']

    path = meme.make_meme(img_file, body, author)

    os.remove(img_file)

    return render_template('meme.html', path=path)


if __name__ == "__main__":
    app.run()
