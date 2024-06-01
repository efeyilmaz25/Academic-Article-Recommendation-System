# Academic Article Recommendation System
 It is a prompt that provides personalized article recommendations based on users' interests and reading history, using the Inspec dataset. The main goal of the project is to direct users to the right articles of interest.

Text pre-processing was carried out on the article abstracts using Python and the NLTK library. As pre-processing, cleaning of ineffective words, removal of punctuation marks and finding word roots were carried out. Vector representations for articles and user profiles were created using FastText and SCIBERT models. Vectors of users' interests were evaluated in terms of similarity, and 5 suggestions were presented separately for FastText and SCIBERT vector representations.

A user-friendly interface has been designed. Necessary operations are presented to the user via buttons.
