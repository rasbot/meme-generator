
from random import randint
from PIL import Image, ImageDraw, ImageFont

class MemeEngine:
    def __init__(self, out_path="./tmp", width=500):
        self.out_path = out_path
        self.width = width

    @staticmethod
    def get_n_lines(body_len, max_len):
        return int(body_len / max_len) + (body_len % max_len > 0)

    @staticmethod
    def split_longer_quote(body, max_len=32):
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
        return f"MemeEngine({self.out_path})    "

    def resize_image(self, img):
        ratio = self.width / float(img.size[0])
        height = int(ratio * float(img.size[1]))
        img = img.resize((self.width, height), Image.NEAREST)
        return img

    def write_meme(self, img, body, author):

        body_font_size = 40
        author_font_size = 20
        max_quote_len = 25
            
        if len(body) > max_quote_len:
            lines = MemeEngine.split_longer_quote(body)
            body_font_size = 30
            author_font_size = 18
            x_offset = 22
        else:
            lines = [body]

        author = "- " + author
        x_pos = randint(int(0.02 * img.width), int(0.12 * img.width))
        y_pos = randint(int(0.02 * img.height), int(0.8 * img.height))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('./fonts/cambriaz.ttf', size=body_font_size)
        line_offset = 0
        for line in lines:
            draw.text((x_pos, y_pos + line_offset), line, font=font, fill='white', stroke_width=2, stroke_fill='black')
            line_offset += 35
        font = ImageFont.truetype('d:/gal/cambriaz.ttf', size=author_font_size)
        draw.text((x_pos + x_offset, y_pos + line_offset + 10), author, font=font, fill='white', stroke_width=2, stroke_fill='black')
        return img

    def save_file(self, img):
        self.filename = f"{self.out_path}/{randint(0,100000)}.jpeg"
        img.save(self.filename)

    def make_meme(self, img, body, author):
        img = Image.open(img)
        img = self.resize_image(img)
        img = self.write_meme(img, body, author)
        self.save_file(img)
        return self.filename
    