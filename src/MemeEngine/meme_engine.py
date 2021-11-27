"""Create memes using an image path, a quote, and an author.

Use the `MemeEngine` class to define and create a meme. A path
for the final meme is passed into the instantiated object and
the main method `make_meme` can be called on the `MemeEngine`
object which will create the meme image.

An image is first resized, and the meme text is written
on the image in a random position. The author name will be
added below the image.
"""
from random import randint
from PIL import Image, ImageDraw, ImageFont

class MemeEngine:
    """`MemeEngine` class used to create a meme object. 
    
    Encapsulates the image object, resizing it, and adding
    text to the image. The final image is saved in the
    `out_path` location."""
    def __init__(self, out_path="./tmp", width=500):
        """

        Args:
            out_path (str, optional): [description]. Defaults to "./tmp".
            width (int, optional): [description]. Defaults to 500.
        """
        self.out_path = out_path
        self.img = None
        self.width = width
        self.filename = ""

    @staticmethod
    def get_n_lines(body_len, max_len):
        return int(body_len / max_len) + (body_len % max_len > 0)

    @staticmethod
    def split_longer_quote(body, max_len=25):
        n_lines = MemeEngine.get_n_lines(len(body), max_len)
        splt = body.split(" ")
        lines = []
        for i in range(n_lines):
            line = ""
            while len(line) <= max_len and len(splt) > 0:
                if len(line + splt[0]) <= max_len:
                    line = line + splt.pop(0) + " "
                else:
                    break
            lines.append(line)
            lines = [line.strip() for line in lines]
        return lines

    def __repr__(self):
        """Return `repr(self)`, a computer-readable string representation of this object."""
        return f"MemeEngine({self.out_path})"

    def resize_image(self):
        ratio = self.width / float(self.img.size[0])
        height = int(ratio * float(self.img.size[1]))
        self.img = self.img.resize((self.width, height), Image.NEAREST)

    def write_meme(self, body, author):

        body_font_size = 35
        author_font_size = 20
        max_quote_len = 25
            
        if len(body) > max_quote_len:
            lines = MemeEngine.split_longer_quote(body)
            body_font_size = 30
            author_font_size = 18
            x_offset = 22
        else:
            lines = [body]
            x_offset = 0

        author = "- " + author
        x_pos = randint(int(0.02 * self.img.width), int(0.12 * self.img.width))
        y_pos = randint(int(0.02 * self.img.height), int(0.8 * self.img.height))
        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype('./fonts/cambriaz.ttf', size=body_font_size)
        line_offset = 0
        for line in lines:
            draw.text((x_pos, y_pos + line_offset), line, font=font, fill='white', stroke_width=2, stroke_fill='black')
            line_offset += 35
        font = ImageFont.truetype('d:/gal/cambriaz.ttf', size=author_font_size)
        draw.text((x_pos + x_offset, y_pos + line_offset + 10), author, font=font, fill='white', stroke_width=2, stroke_fill='black')

    def save_file(self):
        self.filename = f"{self.out_path}/{randint(0,100000)}.jpeg"
        self.img.save(self.filename)

    def make_meme(self, img, body, author):
        self.img = Image.open(img)
        self.resize_image()
        self.write_meme(body, author)
        self.save_file()
        return self.filename
    