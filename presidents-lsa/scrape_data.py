# -*- coding: utf-8 -*-
import re
import wikipedia
import csv
import sys
import os
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

"""
This script grabs the Wikipedia pages of all Democratic and Republican U.S. presidents
and performs some basic text preprocessing on them: removal of stop words, punctuation,
and lemmatizing of each word.

To use this script:

$ python scrape_data.py <absolute path to data directory>

"""

# list of Republican and Democrat US presidents. Values are tuples containing party
# affiliation and the year of the first elected term as president.
RD_PRESIDENTS = {
    'Andrew Jackson': ('D', 1828), 'Martin Van Buren': ('D', 1836), 'James K. Polk': ('D', 1844),
    'Franklin Pierce': ('D', 1852), 'James Buchanan': ('D', 1856), 'Abraham Lincoln': ('R', 1860),
    'Ulysses S. Grant': ('R', 1868), 'Rutherford B. Hayes': ('R', 1876), 'James A. Garfield': ('R', 1880),
    'Chester A. Arthur': ('R', 1881), 'Grover Cleveland': ('D', 1884), 'Benjamin Harrison': ('R', 1888),
    'William McKinley': ('R', 1896), 'Theodore Roosevelt': ('R', 1900), 'William Howard Taft': ('R', 1908),
    'Woodrow Wilson': ('D', 1912), 'Warren G. Harding': ('R', 1920), 'Calvin Coolidge': ('R', 1924),
    'Herbert Hoover': ('R', 1928), 'Franklin D. Roosevelt': ('D', 1932), 'Harry S. Truman': ('D', 1948),
    'Dwight D. Eisenhower': ('R', 1952), 'John F. Kennedy': ('D', 1960), 'Lyndon B. Johnson': ('D', 1964),
    'Richard Nixon': ('R', 1968), 'Gerald Ford': ('R', 1972), 'Jimmy Carter': ('D', 1976),
    'Ronald Reagan': ('R', 1980), 'George H. W. Bush': ('R', 1988), 'Bill Clinton': ('D', 1992),
    'George W. Bush': ('R', 2000), 'Barack Obama': ('D', 2008), 'Donald Trump': ('R', 2016)
}

# A dict containing the titles of 2 wiki articles on conservative and liberal politics/ideology in the US.
# Keys contain the title of the documents, while the values contain the party referenced, along with the
# section to begin reading the document from. This helps in keeping the ideology and politics of both parties
# (relatively) consistent across the document corpus.
IDEOLOGY_DOCS = {
    'Conservatism_in_the_United_States': ('R', '\n\n\n=== Early 20th century'),
    'Modern_liberalism_in_the_United_States': ('D', '\n\n\n=== Progressive Era')
}

# some helpful tools for pre-processing the wiki document texts
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
word_punct_tokenizer = WordPunctTokenizer()


def cleanse_wiki(doc):
    # chop off the notes and references at the end of the article
    # and replace characters denoting section titles with whitespace
    processed_doc = doc.split('\n\n\n== Notes and references')[0].replace('\n', ' ').replace('=', ' ')
    tokens = word_punct_tokenizer.tokenize(processed_doc)
    filt_tokens = filter(lambda token: token not in stop_words and re.match('[a-zA-Z0-9]', token), tokens)
    normalized = [lemma.lemmatize(ft) for ft in filt_tokens]
    return ' '.join(normalized).lower()


def main():
    pres_file_name = os.path.join(sys.argv[1], 'presidents.csv')
    with open(pres_file_name, 'w') as csvfile:
        fieldnames = ['President', 'Party', 'Year', 'Text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (k, v) in RD_PRESIDENTS.iteritems():
            pres_page = wikipedia.WikipediaPage(k.replace(' ', '_')).content
            writer.writerow({
                'President': k,
                'Party': v[0],
                'Year': v[1],
                'Text': cleanse_wiki(pres_page).encode('utf-8')
            })

    ideo_file_name = os.path.join(sys.argv[1], 'ideology_queries.csv')
    with open(ideo_file_name, 'w') as ideocsv:
        fieldnames = ['Title', 'Party', 'Text']
        writer = csv.DictWriter(ideocsv, fieldnames=fieldnames)
        writer.writeheader()
        for (k, v) in IDEOLOGY_DOCS.iteritems():
            page = wikipedia.WikipediaPage(k).content.split(v[1])[1]
            writer.writerow({
                'Title': k,
                'Party': v[0],
                'Text': cleanse_wiki(page).encode('utf-8')
            })


if __name__ == '__main__':
    main()
