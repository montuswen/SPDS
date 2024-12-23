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
    
def load_dataset_from_dir(directory):
    """
    从指定目录加载 JSON 数据集
    :param directory: 包含 JSON 文件的目录
    :return: 合并后的数据列表
    """
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):  # 只处理 JSON 文件
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)  # 合并到一个列表中
    return all_data

if __name__ == "__main__":
    print(get_articles("../data/articles"))
