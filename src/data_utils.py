import fitz
import json
import os


def get_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_image(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        images += page.get_images()
    return images


def get_articles(directory_path):
    articles = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            articles.append(os.path.abspath(
                os.path.join(directory_path, filename)))
    return articles


if __name__ == "__main__":
    print(get_articles("../data/articles"))
