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
from typing import List
from PIL import Image, ImageDraw, ImageFont


class MemeEngine:
    """`MemeEngine` class used to create a meme object.

    Encapsulates the image object, resizing it, and adding
    text to the image. The final image is saved in the
    `out_path` directory."""

    def __init__(self, out_path: str = "./tmp", width: int = 500) -> None:
        """Initialize a meme object.

        Takes the output path for a meme image and
        the resize width.

        Args:
            out_path (str, optional): Folder path for the meme
                created. Defaults to "./tmp".
            width (int, optional): Width of resized image.
                Defaults to 500.
        """
        self.out_path = out_path
        self.img = None
        self.width = width
        self.filename = ""

    @staticmethod
    def get_n_lines(body_len: int, max_len: int) -> int:
        """Get number of lines needed for meme.

        Use the number of characters in the body text
        and a maximum character length to determine how
        many lines the meme text will need.

        Args:
            body_len (int): Length of body string.
            max_len (int): Maximum character length per line
                in meme text.

        Returns:
            int: Number of lines needed for meme text.
        """
        return int(body_len / max_len) + (body_len % max_len > 0)

    @staticmethod
    def split_longer_quote(body: str, max_len: int = 25) -> List[str]:
        """Split longer quotes into lines less than max_len.

        For longer quotes, split the body string into shorter
        strings to prevent the text from spilling over the image edge.

        Args:
            body (str): Quote string.
            max_len (int, optional): Maximum line length of meme
                text.. Defaults to 25.

        Returns:
            List[str]: List of shorter strings.
        """
        n_lines = MemeEngine.get_n_lines(len(body), max_len)
        splt = body.split(" ")
        lines = []
        for _ in range(n_lines):
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
        """Return `repr(self)`, a computer-readable string
        representation of this object.
        """
        return f"MemeEngine({self.out_path})"

    def resize_image(self):
        """Resize an image using instance variable `width`.

        Image will be resized to the width parameter, and the height
        will be scaled proportionally to keep the original image aspect
        ratio.
        """
        ratio = self.width / float(self.img.size[0])
        height = int(ratio * float(self.img.size[1]))
        self.img = self.img.resize((self.width, height), Image.NEAREST)

    def write_meme(self, body: str, author: str) -> None:
        """Write meme text on image.

        Using the resized image, write meme text onto the image,
        scaling the text size to fit on the page. If the quote string
        is too long, break up the text to multiple lines. Text will be
        randomly placed on the image, within a range to allow it to
        be displayed correctly.

        Args:
            body (str): Quote string.
            author (str): Author string.
        """
        body_font_size = 35
        author_font_size = 20
        max_quote_len = 22

        if len(body) > max_quote_len:
            lines = MemeEngine.split_longer_quote(body)
            body_font_size = 30
            author_font_size = 18
            x_offset = 22
        else:
            lines = [body]
            x_offset = 15

        author = "- " + author
        x_pos = randint(
            int(0.02 * self.img.width), int(0.12 * self.img.width)
            )
        y_pos = randint(
            int(0.02 * self.img.height), int(0.8 * self.img.height)
            )
        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype("./fonts/cambriaz.ttf", size=body_font_size)
        line_offset = 0
        # subtract a value for every line so multi-line quotes don't go below
        # lower edge of image
        lines_offset = 0
        if len(lines) > 1 and (y_pos > 0.5 * self.img.height):
            lines_offset = 35 * (len(lines) - 1)
        for line in lines:
            draw.text(
                (x_pos, y_pos + line_offset - lines_offset),
                line,
                font=font,
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            line_offset += 35
        font = ImageFont.truetype("d:/gal/cambriaz.ttf", size=author_font_size)
        draw.text(
            (x_pos + x_offset, y_pos + line_offset - lines_offset + 10),
            author,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    def save_file(self):
        """Save final meme image to the `out_path` folder.

        Use a random file name to store the meme."""
        self.filename = f"{self.out_path}/{randint(0,100000)}.jpeg"
        self.img.save(self.filename)

    def make_meme(self, img: str, body: str, author: str) -> str:
        """Main method to create the meme.

        Reads in image path into image array, resizes the image,
        writes quote text and author text onto image, saves the file
        to a directory, and returns the file path of the saved meme.

        Args:
            img (str): Initial image location.
            body (str): Quote string.
            author (str): Author string.

        Returns:
            str: Final image file path.
        """
        self.img = Image.open(img)
        self.resize_image()
        self.write_meme(body, author)
        self.save_file()
        return self.filename
