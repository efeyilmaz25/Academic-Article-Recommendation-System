# -------------------- Some imports we need.
import subprocess
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from pymongo import MongoClient
import pymongo
import random
import sys

import os
import re
import nltk
import torch
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import time
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request
from bson import Code

# -------------------- End of imports we need --------------------



# -------------------- Download pre-trained model.
#fasttext.util.download_model('en', if_exists='ignore')
#ft_model = fasttext.load_model('cc.en.300.bin')
# -------------------- End of download part --------------------



# -------------------- Authorization Types.
users = {
    "user": {"password": "1"},
    "admin": {"password": "2"}
}
# -------------------- End of "users" definition --------------------


# -------------------- Get article names.
docs_path = 'Inspec/docsutf8'
article_dict = {}

for filename in os.listdir(docs_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(docs_path, filename)
        file_id = filename.split('.')[0]

        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
            article_dict[file_id] = first_line
# -------------------- End of this part --------------------



# -------------------- Article's informations.
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

        articles_id.append(file_id)
        articles_title.append(title)
        articles_abstract.append(abstract)
        articles_keywords.append(keywords)
# --------------------  End of this part --------------------


# -------------------- Main Screen.
def open_system_screen(username):

    system_screen = Toplevel(root) # some main screen settings.
    system_screen.title("System Screen")
    system_screen.geometry("1250x700+210+100")
    system_screen.configure(bg="grey")

    top_frame = Frame(system_screen, bg="grey")
    top_frame.pack(fill="x")

    left_frame = Frame(system_screen, bg='grey')
    left_frame.place(x=10, y=80, width=250, height=650)

    right_frame = Frame(system_screen, bg='grey')
    right_frame.place(x=300, y=80, width=800, height=500)

    scroll_Bar_x = Scrollbar(right_frame, orient='horizontal')
    scroll_Bar_y = Scrollbar(right_frame, orient='vertical')

    data_table = ttk.Treeview(right_frame, columns=(), xscrollcommand=scroll_Bar_x.set, yscrollcommand=scroll_Bar_y.set, show='headings')

    scroll_Bar_x.config(command=data_table.xview)
    scroll_Bar_y.config(command=data_table.yview)

    data_table.configure(xscrollcommand=scroll_Bar_x.set)

    scroll_Bar_x.pack(side='bottom', fill='x')
    scroll_Bar_y.pack(side='right', fill='y')

    data_table.pack(fill='both', expand=1)

    client = MongoClient("mongodb://localhost:27017/") # Database connection
    db = client["AcademicArticleRecommendationSystemDatabase"]
    users_collection = db["Users"]

    # -------------------- User adding function.
    def add_user():
        add_user_screen = Toplevel(system_screen)
        add_user_screen.title("Add Students Screen")
        add_user_screen.geometry("400x300+300+150")

        label_name = Label(add_user_screen, text="Name:")
        label_surname = Label(add_user_screen, text="Surname:")
        label_id = Label(add_user_screen, text="Id:")
        label_areas_of_interests = Label(add_user_screen, text="Areas of interests:")

        entry_name = Entry(add_user_screen)
        entry_surname = Entry(add_user_screen)
        entry_id = Entry(add_user_screen)
        entry_areas_of_interests = Entry(add_user_screen)

        label_name.grid(row=0, column=0)
        label_surname.grid(row=1, column=0)
        label_id.grid(row=2, column=0)
        label_areas_of_interests.grid(row=3, column=0)

        entry_name.grid(row=0, column=1)
        entry_surname.grid(row=1, column=1)
        entry_id.grid(row=2, column=1)
        entry_areas_of_interests.grid(row=3, column=1)

        def add():
            name = entry_name.get()
            surname = entry_surname.get()
            id = entry_id.get()
            areas_of_interest = entry_areas_of_interests.get()

            existing_user = users_collection.find_one({"id": id})

            if existing_user:
                messagebox.showerror("Duplicate ID", f"User with ID '{id}' already exists. Please use a unique ID.")
            else:
                users_collection.insert_one({
                    "name": name,
                    "surname": surname,
                    "id": id,
                    "areas_of_interest": areas_of_interest,
                    "reading_history": "empty"
                })
                add_user_screen.destroy()

            add_user_screen.destroy()

        add_button = Button(add_user_screen, text="Add", command=add)
        add_button.grid(row=4, column=0, columnspan=2)
    # -------------------- End of "add_user" function --------------------



    # -------------------- User recommendation function (SCIBERT).
    def user_recommendation_with_scibert():
        user_recommendation_with_scibert_screen = Toplevel(system_screen)
        user_recommendation_with_scibert_screen.title("Recommend with SCIBERT")
        user_recommendation_with_scibert_screen.geometry("400x300+300+150")

        label_id = Label(user_recommendation_with_scibert_screen, text="Id:")

        entry_id = Entry(user_recommendation_with_scibert_screen)

        label_id.grid(row=0, column=0)

        entry_id.grid(row=0, column=1)

        data_table.delete(*data_table.get_children())
        columns_name = ["Id", "Name", "Surname", "Areas of Interests", "Reading History"]
        data_table['columns'] = columns_name

        for col in columns_name:
            data_table.heading(col, text=col)

        users = users_collection.find()

        for user_data in users:
            data_table.insert("", "end", values=(
                user_data["id"], user_data["name"], user_data["surname"], user_data.get("areas_of_interest", ""),
                user_data["reading_history"]))

        def recommend():
            id = entry_id.get()
            user_recommendation_with_scibert_screen.destroy()

            user_data = users_collection.find_one({"id": id})
            recommendations = []

            if user_data:
                if user_data["reading_history"] == "empty":
                    # -------------------- Stopwords list and stemmer obj.
                    stop_words = set(stopwords.words("english"))
                    stemmer = PorterStemmer()

                    # -------------------- Tokenizer and model for SCIBERT.
                    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
                    model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

                    # -------------------- File path.
                    docsutf8_path = "Inspec/docsutf8"
                    keys_path = "Inspec/keys"

                    # -------------------- Clear document summaries and stemming process.
                    def clean_and_stem_text(text):
                        text = re.sub(r'[^\w\s]', '', text.lower())  # Some settings
                        words = word_tokenize(text)
                        cleaned_text = [stemmer.stem(word) for word in words if word not in stop_words]
                        return ' '.join(cleaned_text)

                    # -------------------- End of "clear_and_stem_text" fucntion --------------------

                    # -------------------- Read and clear all articles.
                    article_embeddings = []
                    article_indices = []

                    # -------------------- Creates SCIBERT vector representations for each article.
                    for file_name in os.listdir(docsutf8_path):
                        print("Loading ...")
                        if file_name.endswith(".txt"):
                            file_path = os.path.join(docsutf8_path, file_name)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                cleaned_text = clean_and_stem_text(text)

                                inputs = tokenizer(cleaned_text, return_tensors="pt")
                                outputs = model(**inputs)
                                article_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()

                                article_embeddings.append(article_embedding)
                                article_indices.append(int(file_name[:-4]))
                    # -------------------- End of this part --------------------

                    user_interest = [user_data["areas_of_interest"]]

                    # -------------------- Calculates the similarity between topics of interest and articles.
                    print("Recommendation of SCIBERT for users")
                    print("-------------------------------------------------------")
                    for idx, interest in enumerate(user_interest, start=1):
                        cleaned_user_interest = clean_and_stem_text(interest)
                        inputs = tokenizer(cleaned_user_interest, return_tensors="pt")
                        outputs = model(**inputs)
                        user_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()

                        similarities = []
                        for article_idx, article_embedding in zip(article_indices, article_embeddings):
                            similarity = cosine_similarity(user_embedding, article_embedding)[0][0]
                            similarities.append((article_idx, similarity))

                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_5_articles = similarities[:5]

                        print(f"Recommended articles for user {idx}:")
                        for article_idx, similarity in top_5_articles:
                            print(f"Article ID: {article_idx}, Similarity: {similarity:.4f}")
                            recommendations.append(article_idx)
                        print("\n")
                    # -------------------- End of this part --------------------

                    # -------------------- Create and average vector representations for user profiles.
                    for idx, interest in enumerate(user_interest, start=1):
                        interest_list = interest.split(", ")
                        interest_embeddings = []

                        for single_interest in interest_list:
                            cleaned_interest = clean_and_stem_text(single_interest)
                            inputs = tokenizer(cleaned_interest, return_tensors="pt")
                            outputs = model(**inputs)
                            interest_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
                            interest_embeddings.append(interest_embedding)

                        user_embedding = np.mean(interest_embeddings, axis=0)
                    # -------------------- End of this part
                else:
                    # -------------------- Stopwords list and stemmer obj.
                    stop_words = set(stopwords.words("english"))
                    stemmer = PorterStemmer()

                    # -------------------- Tokenizer and model for SCIBERT.
                    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
                    model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")

                    # -------------------- File path.
                    docsutf8_path = "Inspec500/docsutf8"
                    keys_path = "Inspec500/keys"

                    # -------------------- Clear document summaries and stemming process.
                    def clean_and_stem_text(text):
                        text = re.sub(r'[^\w\s]', '', text.lower())  # Some settings
                        words = word_tokenize(text)
                        cleaned_text = [stemmer.stem(word) for word in words if word not in stop_words]
                        return ' '.join(cleaned_text)

                    # -------------------- End of "clear_and_stem_text" fucntion --------------------

                    # -------------------- Read and clear all articles.
                    article_embeddings = []
                    article_indices = []

                    # -------------------- Creates SCIBERT vector representations for each article.
                    for file_name in os.listdir(docsutf8_path):
                        print("Loading ...")
                        if file_name.endswith(".txt"):
                            file_path = os.path.join(docsutf8_path, file_name)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                cleaned_text = clean_and_stem_text(text)

                                inputs = tokenizer(cleaned_text, return_tensors="pt")
                                outputs = model(**inputs)
                                article_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()

                                article_embeddings.append(article_embedding)
                                article_indices.append(int(file_name[:-4]))
                    # -------------------- End of this part --------------------

                    user_interest = [str(user_data["reading_history"])[1:-1]]

                    # -------------------- Calculates the similarity between topics of interest and articles.
                    print("Recommendation of SCIBERT for users")
                    print("-------------------------------------------------------")
                    for idx, interest in enumerate(user_interest, start=1):
                        cleaned_user_interest = clean_and_stem_text(interest)
                        inputs = tokenizer(cleaned_user_interest, return_tensors="pt")
                        outputs = model(**inputs)
                        user_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()

                        similarities = []
                        for article_idx, article_embedding in zip(article_indices, article_embeddings):
                            similarity = cosine_similarity(user_embedding, article_embedding)[0][0]
                            similarities.append((article_idx, similarity))

                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_5_articles = similarities[:5]

                        print(f"Recommended articles for user {idx}:")
                        for article_idx, similarity in top_5_articles:
                            print(f"Article ID: {article_idx}, Similarity: {similarity:.4f}")
                            recommendations.append(article_idx)
                        print("\n")
                    # -------------------- End of this part --------------------

                    # -------------------- Create and average vector representations for user profiles.
                    for idx, interest in enumerate(user_interest, start=1):
                        interest_list = interest.split(", ")
                        interest_embeddings = []

                        for single_interest in interest_list:
                            cleaned_interest = clean_and_stem_text(single_interest)
                            inputs = tokenizer(cleaned_interest, return_tensors="pt")
                            outputs = model(**inputs)
                            interest_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
                            interest_embeddings.append(interest_embedding)

                        user_embedding = np.mean(interest_embeddings, axis=0)
                    # -------------------- End of this part
            else:
                print("No user with this ID was found.")
                user_recommendation_with_scibert_screen.destroy()


            if user_data:
                user_recommendation_feedback_with_scibert_screen = Toplevel(system_screen)
                user_recommendation_feedback_with_scibert_screen.title("Feedback Screen")
                user_recommendation_feedback_with_scibert_screen.geometry("400x300+300+150")

                label_recommendation_one = Label(user_recommendation_feedback_with_scibert_screen, text="First Recommendation:")
                label_recommendation_two = Label(user_recommendation_feedback_with_scibert_screen, text="Second Recommendation:")
                label_recommendation_three = Label(user_recommendation_feedback_with_scibert_screen, text="Third Recommendation:")
                label_recommendation_four = Label(user_recommendation_feedback_with_scibert_screen, text="Fourth Recommendation:")
                label_recommendation_five = Label(user_recommendation_feedback_with_scibert_screen, text="Fifth Recommendation:")

                entry_recommendation_one = Entry(user_recommendation_feedback_with_scibert_screen)
                entry_recommendation_two = Entry(user_recommendation_feedback_with_scibert_screen)
                entry_recommendation_three = Entry(user_recommendation_feedback_with_scibert_screen)
                entry_recommendation_four = Entry(user_recommendation_feedback_with_scibert_screen)
                entry_recommendation_five = Entry(user_recommendation_feedback_with_scibert_screen)

                label_recommendation_one.grid(row=0, column=0)
                label_recommendation_two.grid(row=1, column=0)
                label_recommendation_three.grid(row=2, column=0)
                label_recommendation_four.grid(row=3, column=0)
                label_recommendation_five.grid(row=4, column=0)

                entry_recommendation_one.grid(row=0, column=1)
                entry_recommendation_two.grid(row=1, column=1)
                entry_recommendation_three.grid(row=2, column=1)
                entry_recommendation_four.grid(row=3, column=1)
                entry_recommendation_five.grid(row=4, column=1)

                def feedback():
                    count_of_zero = 0
                    count_of_one = 0
                    current_reading_history = []
                    first_feedback = entry_recommendation_one.get()
                    second_feedback = entry_recommendation_two.get()
                    third_feedback = entry_recommendation_three.get()
                    fourth_feedback = entry_recommendation_four.get()
                    fifth_feedback = entry_recommendation_five.get()
                    print(recommendations)
                    user_recommendation_feedback_with_scibert_screen.destroy()

                    first_feedback = int(first_feedback)
                    second_feedback = int(second_feedback)
                    third_feedback = int(third_feedback)
                    fourth_feedback = int(fourth_feedback)
                    fifth_feedback = int(fifth_feedback)

                    if first_feedback == 0:
                        count_of_zero += 1
                    elif first_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[0])

                    if second_feedback == 0:
                        count_of_zero += 1
                    elif second_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[1])

                    if third_feedback == 0:
                        count_of_zero += 1
                    elif third_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[2])

                    if fourth_feedback == 0:
                        count_of_zero += 1
                    elif fourth_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[3])

                    if fifth_feedback == 0:
                        count_of_zero += 1
                    elif fifth_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[4])

                    precision = count_of_one / (count_of_zero + count_of_one)

                    if count_of_one == 0:
                        print("Precision Score: ", precision)
                    else:
                        print("Precision Score: ", precision)
                        if user_data["reading_history"] == "empty":
                            users_collection.update_one({"id": id}, {"$set": {"reading_history": []}})
                            for article in current_reading_history:
                                users_collection.update_one({"id": id}, {"$push": {"reading_history": article_dict[str(article)]}})
                        else:
                            for article in current_reading_history:
                                users_collection.update_one({"id": id}, {"$push": {"reading_history": article_dict[str(article)]}})


                feedback_button = Button(user_recommendation_feedback_with_scibert_screen, text="Feed Back", command=feedback)
                feedback_button.grid(row=5, column=0, columnspan=2)


        recommend_button = Button(user_recommendation_with_scibert_screen, text="Recommend", command=recommend)
        recommend_button.grid(row=1, column=0, columnspan=2)
    # -------------------- End of "user_recommendation_with_scibert" function --------------------



    # -------------------- User recommendation function (FastText).
    def user_recommendation_with_fasttext():
        user_recommendation_with_fasttext_screen = Toplevel(system_screen)
        user_recommendation_with_fasttext_screen.title("Recommend with FastText")
        user_recommendation_with_fasttext_screen.geometry("400x300+300+150")

        label_id = Label(user_recommendation_with_fasttext_screen, text="Id:")

        entry_id = Entry(user_recommendation_with_fasttext_screen)

        label_id.grid(row=0, column=0)

        entry_id.grid(row=0, column=1)

        data_table.delete(*data_table.get_children())
        columns_name = ["Id", "Name", "Surname", "Areas of Interests", "Reading History"]
        data_table['columns'] = columns_name

        for col in columns_name:
            data_table.heading(col, text=col)

        users = users_collection.find()


        for user_data in users:
            data_table.insert("", "end", values=(
                user_data["id"], user_data["name"], user_data["surname"], user_data.get("areas_of_interest", ""),
                user_data["reading_history"]))

        def recommend():
            id = entry_id.get()
            user_recommendation_with_fasttext_screen.destroy()

            user_data = users_collection.find_one({"id": id})
            recommendations = []

            if user_data:
                if user_data["reading_history"] == "empty":
                    # -------------------- Stopwords list and stemmer obj.
                    stop_words = set(stopwords.words("english"))
                    stemmer = PorterStemmer()

                    # -------------------- File path.
                    docsutf8_path = "Inspec/docsutf8"
                    keys_path = "Inspec/keys"

                    # -------------------- Clear document summaries and stemming process.
                    def clean_and_stem_text(text):
                        text = re.sub(r'[^\w\s]', '', text.lower())  # Some settings
                        words = word_tokenize(text)
                        cleaned_text = [stemmer.stem(word) for word in words if word not in stop_words]
                        return ' '.join(cleaned_text)

                    # -------------------- End of "clear_and_stem_text" fucntion --------------------

                    # -------------------- Read and clear all articles.
                    article_embeddings = []
                    article_indices = []

                    # -------------------- FastText Model Loading
                    fasttext.util.download_model('en', if_exists='ignore')
                    ft_model = fasttext.load_model('cc.en.300.bin')

                    # -------------------- Get FastText embedding for a text.
                    def get_fasttext_embedding(text, model):
                        words = clean_and_stem_text(text)
                        embedding = model.get_sentence_vector(words)
                        return embedding

                    # -------------------- End of "get_fasttext_embeddings" function --------------------


                    # -------------------- Compute FastText embeddings for articles.
                    fasttext_article_embeddings = {}
                    article_indices = []

                    for file_name in os.listdir(docsutf8_path):
                        if file_name.endswith(".txt"):
                            file_path = os.path.join(docsutf8_path, file_name)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                article_embedding = get_fasttext_embedding(text, ft_model)
                                fasttext_article_embeddings[int(file_name[:-4])] = article_embedding
                                article_indices.append(int(file_name[:-4]))
                    #print("=========== Control ============")
                    #print(fasttext_article_embeddings)
                    # -------------------- End of this part --------------------

                    user_interest = [user_data["areas_of_interest"]]

                    # -------------------- Compute FastText embeddings for user profiles.
                    fasttext_user_embeddings = []
                    for idx, interest in enumerate(user_interest, start=1):
                        user_embedding = get_fasttext_embedding(interest, ft_model)
                        fasttext_user_embeddings.append(user_embedding)
                    #print("=========== Control ============")
                    #print(fasttext_user_embeddings)
                    # -------------------- End of this part --------------------


                    # -------------------- Compute averaged FastText embedding for user profiles.
                    averaged_fasttext_user_embeddings = []
                    for idx, interest in enumerate(user_interest, start=1):
                        interest_list = interest.split(", ")
                        interest_embeddings = []
                        for single_interest in interest_list:
                            user_embedding = get_fasttext_embedding(single_interest, ft_model)
                            interest_embeddings.append(user_embedding)

                        averaged_user_embedding = np.mean(interest_embeddings, axis=0)
                        averaged_fasttext_user_embeddings.append(averaged_user_embedding)
                    #print("=========== Control ============")
                    #print(averaged_fasttext_user_embeddings)
                    # -------------------- End of this part --------------------

                    # -------------------- Calculates the similarity between topics of interest and articles.
                    print("Recommendation of FastText for users")
                    print("-------------------------------------------------------")
                    for idx, interest_embedding in enumerate(fasttext_user_embeddings, start=1):
                        similarities = []
                        for article_idx, article_embedding in fasttext_article_embeddings.items():
                            similarity = cosine_similarity([interest_embedding], [article_embedding])[0][0]
                            similarities.append((article_idx, similarity))

                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_5_articles = similarities[:5]

                        print(f"Recommended articles for user {idx}:")
                        for article_idx, similarity in top_5_articles:
                            print(f"Article ID: {article_idx}, Similarity: {similarity:.4f}")
                            recommendations.append(article_idx)
                        print("\n")
                    # -------------------- End of this part --------------------
                else:
                    # -------------------- Stopwords list and stemmer obj.
                    stop_words = set(stopwords.words("english"))
                    stemmer = PorterStemmer()

                    # -------------------- File path.
                    docsutf8_path = "Inspec/docsutf8"
                    keys_path = "Inspec/keys"

                    # -------------------- Clear document summaries and stemming process.
                    def clean_and_stem_text(text):
                        text = re.sub(r'[^\w\s]', '', text.lower())  # Some settings
                        words = word_tokenize(text)
                        cleaned_text = [stemmer.stem(word) for word in words if word not in stop_words]
                        return ' '.join(cleaned_text)

                    # -------------------- End of "clear_and_stem_text" fucntion --------------------

                    # -------------------- Read and clear all articles.
                    article_embeddings = []
                    article_indices = []

                    # -------------------- FastText Model Loading
                    fasttext.util.download_model('en', if_exists='ignore')
                    ft_model = fasttext.load_model('cc.en.300.bin')

                    # -------------------- Get FastText embedding for a text.
                    def get_fasttext_embedding(text, model):
                        words = clean_and_stem_text(text)
                        embedding = model.get_sentence_vector(words)
                        return embedding

                    # -------------------- End of "get_fasttext_embeddings" function --------------------

                    # -------------------- Compute FastText embeddings for articles.
                    fasttext_article_embeddings = {}
                    article_indices = []

                    for file_name in os.listdir(docsutf8_path):
                        if file_name.endswith(".txt"):
                            file_path = os.path.join(docsutf8_path, file_name)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                article_embedding = get_fasttext_embedding(text, ft_model)
                                fasttext_article_embeddings[int(file_name[:-4])] = article_embedding
                                article_indices.append(int(file_name[:-4]))
                    #print("=========== Control ============")
                    #print(fasttext_article_embeddings)
                    # -------------------- End of this part --------------------

                    user_interest = [str(user_data["reading_history"])[1:-1]]

                    # -------------------- Compute FastText embeddings for user profiles.
                    fasttext_user_embeddings = []
                    for idx, interest in enumerate(user_interest, start=1):
                        user_embedding = get_fasttext_embedding(interest, ft_model)
                        fasttext_user_embeddings.append(user_embedding)
                    #print("=========== Control ============")
                    #print(fasttext_user_embeddings)
                    # -------------------- End of this part --------------------


                    # -------------------- Compute averaged FastText embedding for user profiles.
                    averaged_fasttext_user_embeddings = []
                    for idx, interest in enumerate(user_interest, start=1):
                        interest_list = interest.split(", ")
                        interest_embeddings = []
                        for single_interest in interest_list:
                            user_embedding = get_fasttext_embedding(single_interest, ft_model)
                            interest_embeddings.append(user_embedding)

                        averaged_user_embedding = np.mean(interest_embeddings, axis=0)
                        averaged_fasttext_user_embeddings.append(averaged_user_embedding)
                    #print("=========== Control ============")
                    #print(averaged_fasttext_user_embeddings)
                    # -------------------- End of this part --------------------

                    # -------------------- Calculates the similarity between topics of interest and articles.
                    print("Recommendation of FastText for users")
                    print("-------------------------------------------------------")
                    for idx, interest_embedding in enumerate(fasttext_user_embeddings, start=1):
                        similarities = []
                        for article_idx, article_embedding in fasttext_article_embeddings.items():
                            similarity = cosine_similarity([interest_embedding], [article_embedding])[0][0]
                            similarities.append((article_idx, similarity))

                        similarities.sort(key=lambda x: x[1], reverse=True)
                        top_5_articles = similarities[:5]

                        print(f"Recommended articles for user {idx}:")
                        for article_idx, similarity in top_5_articles:
                            print(f"Article ID: {article_idx}, Similarity: {similarity:.4f}")
                            recommendations.append(article_idx)
                        print("\n")
                    # -------------------- End of this part --------------------
            else:
                print("No user with this ID was found.")
                user_recommendation_with_fasttext_screen.destroy()

            if user_data:
                user_recommendation_feedback_with_fasttext_screen = Toplevel(system_screen)
                user_recommendation_feedback_with_fasttext_screen.title("Feedback Screen")
                user_recommendation_feedback_with_fasttext_screen.geometry("400x300+300+150")

                label_recommendation_one = Label(user_recommendation_feedback_with_fasttext_screen, text="First Recommendation:")
                label_recommendation_two = Label(user_recommendation_feedback_with_fasttext_screen, text="Second Recommendation:")
                label_recommendation_three = Label(user_recommendation_feedback_with_fasttext_screen, text="Third Recommendation:")
                label_recommendation_four = Label(user_recommendation_feedback_with_fasttext_screen, text="Fourth Recommendation:")
                label_recommendation_five = Label(user_recommendation_feedback_with_fasttext_screen, text="Fifth Recommendation:")

                entry_recommendation_one = Entry(user_recommendation_feedback_with_fasttext_screen)
                entry_recommendation_two = Entry(user_recommendation_feedback_with_fasttext_screen)
                entry_recommendation_three = Entry(user_recommendation_feedback_with_fasttext_screen)
                entry_recommendation_four = Entry(user_recommendation_feedback_with_fasttext_screen)
                entry_recommendation_five = Entry(user_recommendation_feedback_with_fasttext_screen)

                label_recommendation_one.grid(row=0, column=0)
                label_recommendation_two.grid(row=1, column=0)
                label_recommendation_three.grid(row=2, column=0)
                label_recommendation_four.grid(row=3, column=0)
                label_recommendation_five.grid(row=4, column=0)

                entry_recommendation_one.grid(row=0, column=1)
                entry_recommendation_two.grid(row=1, column=1)
                entry_recommendation_three.grid(row=2, column=1)
                entry_recommendation_four.grid(row=3, column=1)
                entry_recommendation_five.grid(row=4, column=1)

                def feedback():
                    count_of_zero = 0
                    count_of_one = 0
                    current_reading_history = []
                    first_feedback = entry_recommendation_one.get()
                    second_feedback = entry_recommendation_two.get()
                    third_feedback = entry_recommendation_three.get()
                    fourth_feedback = entry_recommendation_four.get()
                    fifth_feedback = entry_recommendation_five.get()
                    print(recommendations)
                    user_recommendation_feedback_with_fasttext_screen.destroy()

                    first_feedback = int(first_feedback)
                    second_feedback = int(second_feedback)
                    third_feedback = int(third_feedback)
                    fourth_feedback = int(fourth_feedback)
                    fifth_feedback = int(fifth_feedback)

                    if first_feedback == 0:
                        count_of_zero += 1
                    elif first_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[0])

                    if second_feedback == 0:
                        count_of_zero += 1
                    elif second_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[1])

                    if third_feedback == 0:
                        count_of_zero += 1
                    elif third_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[2])

                    if fourth_feedback == 0:
                        count_of_zero += 1
                    elif fourth_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[3])

                    if fifth_feedback == 0:
                        count_of_zero += 1
                    elif fifth_feedback == 1:
                        count_of_one += 1
                        current_reading_history.append(recommendations[4])

                    precision = count_of_one / (count_of_zero + count_of_one)

                    if count_of_one == 0:
                        print("Precision Score: ", precision)
                    else:
                        print("Precision Score: ", precision)
                        if user_data["reading_history"] == "empty":
                            users_collection.update_one({"id": id}, {"$set": {"reading_history": []}})
                            for article in current_reading_history:
                                users_collection.update_one({"id": id}, {"$push": {"reading_history": article_dict[str(article)]}})

                        else:
                            for article in current_reading_history:
                                users_collection.update_one({"id": id}, {"$push": {"reading_history": article_dict[str(article)]}})


                feedback_button = Button(user_recommendation_feedback_with_fasttext_screen, text="Feed Back", command=feedback)
                feedback_button.grid(row=5, column=0, columnspan=2)

        recommend_button = Button(user_recommendation_with_fasttext_screen, text="Recommend", command=recommend)
        recommend_button.grid(row=6, column=0, columnspan=2)
    # -------------------- End of "user_recommendation_with_fasttext" function --------------------



    # -------------------- User deletion function.
    def delete_user():
        delete_user_screen = Toplevel(system_screen)
        delete_user_screen.title("User Deletion Screen")
        delete_user_screen.geometry("400x300+300+150")

        label_user_id = Label(delete_user_screen, text="User Id:")
        entry_user_id = Entry(delete_user_screen)
        label_user_id.grid(row=0, column=0)
        entry_user_id.grid(row=0, column=1)

        def delete():
            id = entry_user_id.get()

            result = users_collection.delete_one({
                "id": id
            })

            delete_user_screen.destroy()

        user_delete_button = Button(delete_user_screen, text="Delete", command=delete)
        user_delete_button.grid(row=1, column=0, columnspan=2)
    # -------------------- End of "delete_user" function --------------------



    # -------------------- User view function.
    def view_user():
        view_user_screen = Toplevel(system_screen)
        view_user_screen.title("View User Information Screen")
        view_user_screen.geometry("400x300+300+150")

        label_user_id = Label(view_user_screen, text="User ID:")
        entry_user_id = Entry(view_user_screen)

        label_user_id.grid(row=0, column=0)
        entry_user_id.grid(row=0, column=1)

        def view():
            id = entry_user_id.get()

            if not id:
                print("User id must be provided")
                return

            data_table.delete(*data_table.get_children())

            columns_name = ["Id", "Name", "Surname", "Areas of Interests", "Reading History"]
            data_table['columns'] = columns_name

            for col in columns_name:
                data_table.heading(col, text=col)

            query = {"id": id}
            user_data = users_collection.find_one(query)

            if user_data:
                data_table.insert('', 'end', values=(user_data["id"], user_data["name"], user_data["surname"], user_data.get("areas_of_interest", ""), user_data["reading_history"]))
            else:
                print(f"User '{id}' was not found.")

            view_user_screen.destroy()

        view_user_screen_button = Button(view_user_screen, text="View", command=view)
        view_user_screen_button.grid(row=1, column=0, columnspan=2)
    # -------------------- End of "view_user" function --------------------



    # -------------------- All user view function.
    def view_all_users():
        data_table.delete(*data_table.get_children())
        columns_name = ["Id", "Name", "Surname", "Areas of Interests", "Reading History"]
        data_table['columns'] = columns_name

        for col in columns_name:
            data_table.heading(col, text=col)

        users = users_collection.find()

        for user_data in users:
            data_table.insert("", "end", values=(user_data["id"], user_data["name"], user_data["surname"], user_data.get("areas_of_interest", ""), user_data["reading_history"]))
    # -------------------- End of "view_all_users" function --------------------


    # -------------------- View web-site function.
    def web_site():
        subprocess.run(['python', 'WebSite.py'])
    # -------------------- End of "web_site" function --------------------



    # -------------------- User update function.
    def user_update():
        user_find_screen = Toplevel(system_screen)
        user_find_screen.title("User Find Screen")
        user_find_screen.geometry("400x300+300+150")

        label_id = Label(user_find_screen, text="Id:")

        entry_id = Entry(user_find_screen)

        label_id.grid(row=0, column=0)

        entry_id.grid(row=0, column=1)

        def find():
            id = entry_id.get()

            if not id:
                print("User id must be provided")
                return

            data_table.delete(*data_table.get_children())

            columns_name = ["Id", "Name", "Surname", "Areas of Interests", "Reading History"]
            data_table['columns'] = columns_name

            for col in columns_name:
                data_table.heading(col, text=col)

            query = {"id": id}
            user_data = users_collection.find_one(query)

            if user_data:
                data_table.insert('', 'end', values=(
                user_data["id"], user_data["name"], user_data["surname"], user_data.get("areas_of_interest", ""),
                user_data["reading_history"]))
            else:
                print(f"User '{id}' was not found.")

            user_find_screen.destroy()

            user_update_screen = Toplevel(system_screen)
            user_update_screen.title("User Find Screen")
            user_update_screen.geometry("400x300+300+150")

            label_user_name = Label(user_update_screen, text="Name:")
            label_user_surname = Label(user_update_screen, text="Surname:")
            label_user_areas_of_interest = Label(user_update_screen, text="Areas of interest:")
            label_user_reading_history = Label(user_update_screen, text="Reading history:")

            entry_user_name = Entry(user_update_screen)
            entry_user_surname = Entry(user_update_screen)
            entry_user_areas_of_interest = Entry(user_update_screen)
            entry_user_reading_history = Entry(user_update_screen)

            label_user_name.grid(row=0, column=0)
            label_user_surname.grid(row=1, column=0)
            label_user_areas_of_interest.grid(row=2, column=0)
            label_user_reading_history.grid(row=3, column=0)

            entry_user_name.grid(row=0, column=1)
            entry_user_surname.grid(row=1, column=1)
            entry_user_areas_of_interest.grid(row=2, column=1)
            entry_user_reading_history.grid(row=3, column=1)

            def update():
                name = entry_user_name.get()
                surname = entry_user_surname.get()
                areas_of_interest = entry_user_areas_of_interest.get()
                reading_history = entry_user_reading_history.get()

                existing_user = users_collection.find_one({"id": id})

                users_collection.update_one(
                    {"id": id},
                    {
                        "$set": {
                            "name": name,
                            "surname": surname,
                            "areas_of_interest": areas_of_interest,
                            "reading_history": reading_history
                        }
                    },
                    upsert=True
                )

                user_update_screen.destroy()

            user_update_screen_button = Button(user_update_screen, text="Update", command=update)
            user_update_screen_button.grid(row=4, column=0, columnspan=2)

        user_find_screen_button = Button(user_find_screen, text="Find", command=find)
        user_find_screen_button.grid(row=1, column=0, columnspan=2)
    # -------------------- End of "user_update" function --------------------


    # -------------------- End of "user_update" function --------------------


    # -------------------- Table clearing function.
    def clear_table():
        children = data_table.get_children()
        data_table.delete(*children)
    # -------------------- End of "clear_table" function --------------------



    # -------------------- System exit fucntion.
    def exit():
        system_screen.destroy()
    # -------------------- End of "exit" funciton --------------------



    # ==================== -------------------- Button definitions -------------------- ====================
    add_user_button = ttk.Button(left_frame, text='Add User', cursor='hand2', width=25, state='normal', command=add_user)
    add_user_button.grid(row=1, column=0, padx=20, pady=10)

    user_recommendation_with_scibert_button = ttk.Button(left_frame, text='Recommend with SCIBERT', cursor='hand2', width=25, state='normal', command=user_recommendation_with_scibert)
    user_recommendation_with_scibert_button.grid(row=2, column=0, padx=20, pady=10)

    user_recommendation_with_fasttext_button = ttk.Button(left_frame, text='Recommend with FastText', cursor='hand2', width=25, state='normal', command=user_recommendation_with_fasttext)
    user_recommendation_with_fasttext_button.grid(row=3, column=0, padx=20, pady=10)

    view_user_button = ttk.Button(left_frame, text="View User", cursor='hand2', width=25, state='normal', command=view_user)
    view_user_button.grid(row=4, column=0, padx=20, pady=10)

    view_all_users_button = ttk.Button(left_frame, text="View All Users", cursor='hand2', width=25, state='normal', command=view_all_users)
    view_all_users_button.grid(row=5, column=0, padx=20, pady=10)

    delete_user_button = ttk.Button(left_frame, text="Delete User", cursor='hand2', width=25, state='normal', command=delete_user)
    delete_user_button.grid(row=6, column=0, padx=10, pady=10)

    web_site_button = ttk.Button(left_frame, text="Web Site", cursor='hand2', width=25, state='normal', command=web_site)
    web_site_button.grid(row=7, column=0, padx=10, pady=10)

    user_update_button = ttk.Button(left_frame, text="User Update", cursor='hand2', width=25, state='normal', command=user_update)
    user_update_button.grid(row=8, column=0, padx=10, pady=10)

    clear_table_button = ttk.Button(left_frame, text='Clear Table', cursor='hand2', width=25, state='normal', command=clear_table)
    clear_table_button.grid(row=9, column=0, padx=20, pady=10)

    exit_button = ttk.Button(left_frame, text='EXIT', cursor='hand2', width=25, command=exit)
    exit_button.grid(row=10, column=0, padx=20, pady=10)
    # ==================== -------------------- End of button definitions part -------------------- ====================


    # -------------------- Activities and passives according to authorization type.
    if username == "user":
        add_user_button.config(state='normal')
        clear_table_button.config(state='normal')
        exit_button.config(state='normal')

        delete_user_button.config(state='disabled')
        delete_user_button.grid_remove()

        view_all_users_button.config(state='disabled')
        view_all_users_button.grid_remove()

    elif username == "admin":
        add_user_button.config(state='normal')
        delete_user_button.config(state='normal')
        view_user_button.config(state='normal')
        clear_table_button.config(state='normal')
        exit_button.config(state='normal')

        user_recommendation_with_scibert_button.config(state='disabled')
        user_recommendation_with_scibert_button.grid_remove()

        user_recommendation_with_fasttext_button.config(state='disabled')
        user_recommendation_with_fasttext_button.grid_remove()
    # -------------------- End of authorization type part --------------------



# -------------------- Splash screen function.
def login():
    user = user_entry.get()
    password = password_entry.get()

    if user in users and password == users[user]["password"]:
        messagebox.showinfo("Successful", "Login Successful!")

        open_system_screen(user)
        user_entry.delete(0, END)
        password_entry.delete(0, END)
    else:
        messagebox.showerror("Error", "Invalid Username or Password")
# -------------------- End of "login" function --------------------
#
#
#
# -------------------- Used to sort on columns.
def sort_column(tv, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    l.sort(reverse=reverse)

    for index, (val, k) in enumerate(l):
        tv.move(k, '', index)

    tv.heading(col, command=lambda: sort_column(tv, col, not reverse))
# -------------------- End of "sort_column" function --------------------
#
#
#
# -------------------- "Background", "Text" and "font" color settings.
root = Tk()
root.option_add('*TButton*Background', '#06283D')
root.option_add('*TButton*Foreground', 'white')
root.option_add('*TButton*Font', ('Arial', 12))

root.geometry('720x300+275+100')
root.resizable(0, 0)
root.title('Academic Article Recommendation System')
# -------------------- End of "Background", "Text" and "font" color settings part --------------------
#
#
#
# -------------------- Splash screen settings.
login_frame = ttk.Frame(root)
login_frame.pack(pady=50)

user_label = Label(login_frame, text='Authorization Type:', font=('Comic Sans MS', 15, 'bold'))
user_label.grid(row=1, column=0, padx=20, pady=10)
user_entry = Entry(login_frame, font=('Comic Sans MS', 15, 'bold'), bd=3, fg='#1c1c1c')
user_entry.grid(row=1, column=1)

password_label = Label(login_frame, text='Password:', font=('Comic Sans MS', 15, 'bold'))
password_label.grid(row=2, column=0, padx=20, pady=10)
password_entry = Entry(login_frame, font=('Comic Sans MS', 15, 'bold'), bd=3, fg='#1c1c1c', show="*")
password_entry.grid(row=2, column=1)

login_button = Button(login_frame, text='Entry', font=('Comic Sans MS', 13, 'bold'), width=13, cursor='hand2', command=login)
login_button.grid(row=3, column=1, pady=10)

root.mainloop()
# -------------------- End of splash screen settings part --------------------

