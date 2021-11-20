import os
import random
import subprocess
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import docx


class QuoteModel:
    def __init__(self, quote, author):
        self.quote = quote
        self.author = author

    def __repr__(self):
        return f'<{self.quote} {self.author}>'


class IngestorInterface(ABC):
    allowed_extensions = []
    
    @classmethod
    def can_ingest(cls, path: str) -> bool:
        ext = path.split('.')[-1]
        return ext in cls.allowed_extensions

    @classmethod
    @abstractmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        pass


class TXTIngestor(IngestorInterface):
    allowed_extensions = ['txt']

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception('cannot ingest exception')

        quotes = []

        with open(path, "r") as infile:
            data = infile.readlines()

        for d in data:
            d_ = d.strip()
            d_split = d_.split(" - ")
            quote = d_split[0]
            author = d_split[1]
            new_quote = QuoteModel(quote, author)
            quotes.append(new_quote)

        return quotes


class CSVIngestor(IngestorInterface):
    allowed_extensions = ['csv']

    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception('cannot ingest exception')

        quotes = []
        df = pd.read_csv(path, header=0)

        for _, row in df.iterrows():
            quote = row["body"]
            author = row["author"]
            new_quote = QuoteModel(quote, author)
            quotes.append(new_quote)

        return quotes


class DocxIngestor(IngestorInterface):
    allowed_extensions = ['docx']
    
    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception('cannot ingest exception')
            
        quotes = []
        doc = docx.Document(path)
        
        for para in doc.paragraphs:
            if para.text != "":
                parse = para.text.split(',')
                quote = parse[0].split('" - ')[0][1:]
                author = parse[0].split('" - ')[1]
                new_quote = QuoteModel(quote, author)
                quotes.append(new_quote)
                
        return quotes


class PDFIngestor(IngestorInterface):
    allowed_extensions = ['pdf']
    
    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        if not cls.can_ingest(path):
            raise Exception('cannot ingest exception')
        
        tmp = f'./tmp/{random.randint(0,100000000)}.txt'
        call = subprocess.call(['pdftotext', path, tmp])
        
        with open(tmp, "r") as file_ref:

            quotes = []

            for line in file_ref.readlines():
                line = line.strip('\n\r').strip()
                if len(line) > 0:
                    parse = line.split(',')
                    quote = parse[0].split('" - ')[0][1:]
                    author = parse[0].split('" - ')[1]
                    print(quote, author)
                    new_quote = QuoteModel(quote, author)
                    quotes.append(new_quote)
        os.remove(tmp)
        return quotes


class Ingestor(IngestorInterface):
    ingestors = [TXTIngestor, DocxIngestor, CSVIngestor, PDFIngestor]
       
    @classmethod
    def parse(cls, path: str) -> List[QuoteModel]:
        for ingester in cls.ingestors:
            if ingester.can_ingest(path):
                return ingester.parse(path)