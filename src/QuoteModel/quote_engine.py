"""Read in various file formats, and create `QuoteModel` objects for quotes.

Uses an Ingestor Interface to create parsers for txt, pdf, docx, and csv
file types.
Quotes and authors are extracted from each supported file, and
`QuoteModel` objects are created for all quote/author pairs.
"""

import os
import random
import subprocess
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import docx


class QuoteModel:
    """A `QuoteModel` object which has a quote and the associated author."""

    def __init__(self, body: str, author: str):
        """Initialize a `QuoteModel` object.

        Args:
            body (str): A quote.
            author (str): An author.
        """
        self.body = body
        self.author = author

    def __str__(self):
        """Return a readable format for this object."""
        return f"QuoteModel({self.body} - {self.author})"

    def __repr__(self):
        """Return `repr(self)`, a computer-readable string
        representation of this object.
        """
        return f"{self.body} - {self.author}"


class IngestorInterface(ABC):
    """An abstract base class for an IngestorInterface.

    Has one abstract base method which must be implemented
    in any inherited class. Also has a class method `can_ingest` which
    determines if the file can be read in by the inherited ingestor.

    Args:
        ABC (Object): Abstract Base Class super class.
    """

    allowed_extensions = []

    @classmethod
    def can_ingest(cls, path: str) -> bool:
        """Determine if the inherited class can ingest
        the file type of the file located by the path.

        Args:
            path (str): Path to the file being ingested.

        Returns:
            bool: True if file can be ingested, False if not.
        """
        ext = path.split(".")[-1]
        return ext in cls.allowed_extensions

    @classmethod
    @abstractmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """parse method must be implemented in any
        Ingestor class."""


class TXTIngestor(IngestorInterface):
    """Text ingestor class which can parse text
    files to extract quotes and authors."""

    allowed_extensions = ["txt"]

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """Parse a file and ingest if txt file.

        If file is a txt file, read in data and
        create `QuoteModel` objects for each quote
        ingested. Return a list of `QuoteModel` objects.

        Args:
            path (str): String path to txt file.

        Raises:
            Exception: If file cannot be ingested,
                raise exception.

        Returns:
            List[QuoteModel]: List of `QuoteModel objects.
        """
        if not cls.can_ingest(path):
            raise Exception("cannot ingest exception")

        quotes = []

        with open(path, "r") as infile:
            data = infile.readlines()

        for dat in data:
            dat_ = dat.strip()
            dat_split = dat_.split(" - ")
            body = dat_split[0]
            author = dat_split[1]
            new_quote = QuoteModel(body, author)
            quotes.append(new_quote)

        return quotes


class CSVIngestor(IngestorInterface):
    """Parse a file and ingest if csv file.

    If file is a csv file, read in data and
    create `QuoteModel` objects for each quote
    ingested. Return a list of `QuoteModel` objects.

    Args:
        path (str): String path to csv file.

    Raises:
        Exception: If file cannot be ingested,
            raise exception.

    Returns:
        List[QuoteModel]: List of `QuoteModel objects.
    """

    allowed_extensions = ["csv"]

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception("cannot ingest exception")

        quotes = []
        quote_df = pd.read_csv(path, header=0, encoding="ISO-8859-1")

        for _, row in quote_df.iterrows():
            body = row["body"]
            author = row["author"]
            new_quote = QuoteModel(body, author)
            quotes.append(new_quote)

        return quotes


class DocxIngestor(IngestorInterface):
    """Parse a file and ingest if docx file.

    If file is a docx file, read in data and
    create `QuoteModel` objects for each quote
    ingested. Return a list of `QuoteModel` objects.

    Args:
        path (str): String path to docx file.

    Raises:
        Exception: If file cannot be ingested,
            raise exception.

    Returns:
        List[QuoteModel]: List of `QuoteModel objects.
    """

    allowed_extensions = ["docx"]

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception("cannot ingest exception")

        quotes = []
        doc = docx.Document(path)

        for para in doc.paragraphs:
            if para.text != "":
                parse = para.text.split(",")
                body = parse[0].split('" - ')[0][1:]
                author = parse[0].split('" - ')[1]
                new_quote = QuoteModel(body, author)
                quotes.append(new_quote)

        return quotes


class PDFIngestor(IngestorInterface):
    """Parse a file and ingest if pdf file.

    If file is a pdf file, read in data and
    create `QuoteModel` objects for each quote
    ingested. Return a list of `QuoteModel` objects.

    Args:
        path (str): String path to pdf file.

    Raises:
        Exception: If file cannot be ingested,
            raise exception.

    Returns:
        List[QuoteModel]: List of `QuoteModel objects.
    """

    allowed_extensions = ["pdf"]

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception("cannot ingest exception")
        if not os.path.isdir("./tmp"):
            os.makedirs("./tmp")
        tmp = f"./tmp/{random.randint(0,100000000)}.txt"
        _ = subprocess.call(["pdftotext", path, tmp])

        with open(tmp, "r", encoding="utf-8") as file_ref:

            quotes = []

            for line in file_ref.readlines():
                line = line.strip("\n\r").strip()
                if len(line) > 0:
                    try:
                        parse = line.split(",")
                        body = parse[0].split('" - ')[0][1:]
                        author = parse[0].split('" - ')[1]
                        new_quote = QuoteModel(body, author)
                        quotes.append(new_quote)
                    except IndexError:
                        print("Index out of range")
        os.remove(tmp)
        return quotes


class Ingestor(IngestorInterface):
    """Ingestor class which ingests any allowed quote file."""

    ingestors = [TXTIngestor, DocxIngestor, CSVIngestor, PDFIngestor]

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        """Parse a file and ingest if filetype is supported.

        reate `QuoteModel` objects for each quote
        ingested if file has an extension supported by
        an ingestor. Return a list of `QuoteModel` objects.

        Args:
            path (str): String path to file.

        Returns:
            List[QuoteModel]: List of `QuoteModel objects.
        """
        for ingester in cls.ingestors:
            if ingester.can_ingest(path):
                return ingester.parse(path)
        return None
