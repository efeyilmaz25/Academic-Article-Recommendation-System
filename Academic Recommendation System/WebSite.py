import os
import pymongo
from flask import Flask, render_template, request
from bson import Code
from pymongo import MongoClient

# ---------------------- Reading and listing data.
docs_path = 'Inspec/docsutf8'
keys_path = 'Inspec/keys'
articles_id = []
articles_title = []
articles_abstract = []
articles_keywords = []

for filename in os.listdir(docs_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(docs_path, filename)
        file_id = filename.split('.')[0]

        with open(file_path, 'r', encoding='utf-8') as file:
            title = file.readline().strip()
            abstract = file.read().strip()
        key_filename = f"{file_id}.key"
        key_path = os.path.join(keys_path, key_filename)

        if os.path.exists(key_path):
            with open(key_path, 'r', encoding='utf-8') as key_file:
                keywords = key_file.read().strip()
        else:
            keywords = ""

        articles_id.append(int(file_id))
        articles_title.append(str(title))
        articles_abstract.append(str(abstract))
        articles_keywords.append(str(keywords))
# ---------------------- End of this part ----------------------

# ---------------------- Data-base connection and writing.
client = MongoClient("mongodb://localhost:27017/")
db = client["AcademicArticleRecommendationSystemDatabase"]
articles_collection = db["Articles"]

articles_collection.delete_many({}) # clear collection

for i in range(len(articles_id)):
    articles_collection.insert_one({
        "article_id": articles_id[i],
        "article_title": articles_title[i],
        "article_abstract": articles_abstract[i],
        "article_keywords": articles_keywords[i]
    })
# ---------------------- End of this part ----------------------


# ---------------------- Application section.
app = Flask(__name__)

@app.route('/') # Home page
def index():
    articles = articles_collection.find()
    return render_template('index.html', articles=articles)


@app.route('/sort') # Sort Function
def sort():
    column = request.args.get('column')
    order = request.args.get('order')

    sort_direction = pymongo.ASCENDING if order == 'asc' else pymongo.DESCENDING

    if column == '0':
        articles = articles_collection.find().sort("article_id", sort_direction)
    elif column == '1':
        articles = articles_collection.find().sort("article_title", sort_direction)
    elif column == '2':
        articles = articles_collection.find().sort("article_abstract", sort_direction)
    elif column == '3':
        articles = articles_collection.find().sort("article_keywords", sort_direction)
    else:
        sort_key = list(articles_collection.find().limit(1))[0].keys()[int(column)]
        articles = articles_collection.find().sort(sort_key, sort_direction)

    return render_template('index.html', articles=articles)


@app.route('/search') # First type search function
def search():
    search_query = request.args.get('q', '')
    articles = articles_collection.find({"article_title": {"$regex": search_query, "$options": "i"}})
    return render_template('index.html', articles=articles)

@app.route('/search2') # Second type search function
def search2():
    search_query = request.args.get('q', '')
    articles = articles_collection.find({"article_keywords": {"$regex": search_query, "$options": "i"}})
    return render_template('index.html', articles=articles)

@app.route('/search3') # Third type search function
def search3():
    search_query = request.args.get('q', '')
    articles = articles_collection.find({"article_abstract": {"$regex": search_query, "$options": "i"}})
    return render_template('index.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)

# ---------------------- End of "Aplication Section" part ----------------------