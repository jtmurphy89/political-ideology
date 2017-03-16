# -*- coding: utf-8 -*-
import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

"""
Our strategy here is to build an LSA query engine with the Republican/Democrat US president
Wikipedia documents. For the purpose of this exercise, we only include presidents from 1890
onward so that each party's political ideology still reflects what it is today: indeed, the
previous era of American politics saw Democrats being what would today be called Republicans
(and vice versa). To determine the political ideologies that distinguish Republican from
Democratic presidents, we construct 'ideology' vectors (e.g. using Wikipedia articles on
Republican vs. Democratic ideologies), transform them into the lower dimensional space
(specified by the user) and compute the average cosine similarity between the ideology vector
and Republican/Democrat US president documents. The goal is to then find 'ideology' vectors for
which the difference between the avg. cosine similarity from Republican and Democratic presidents
is greatest.

To use this script:

$ python PresidentsLSA.py <num singular values> <presidents csv file> <ideology csv file> <abs path to .png directory>

"""
REDUCED_QUERIES = [
    'liberal social welfare social security public works environment progressive income tax labor union',
    'conservative big business limited small government entrepreneur trickle down economics private spending'
]

COLORS = {
    'D': 'blue',
    'R': 'red'
}


class PresLSAQueryMachine():
    def __init__(self, pres, party, years, docs, K, tf_idf, doc_tf_idf, U, s, Vt):
        self.pres = pres
        self.party = party
        self.years = years
        self.docs = docs
        self.K = K
        self.tf_idf = tf_idf
        self.vocab = tf_idf.vocabulary_
        self.r_vocab = {v: k for (k, v) in tf_idf.vocabulary_.iteritems()}
        self.U = U[:, :K]
        self.s = s
        self.Vt = Vt[:K, :]
        self.transformed_docs = np.dot(np.dot(np.linalg.inv(np.diag(s)[:K, :K]), U[:, :K].transpose()), doc_tf_idf)
        self.normed_docs = normalize(self.transformed_docs, axis=0, norm='l2')

    def top_terms_per_val(self, num_terms):
        top_terms = []
        for n in range(0, self.K):
            term_idcs = [
                self.r_vocab[i[1]] for i in
                sorted(zip(self.U[:, n], range(0, self.U.shape[0])), reverse=True)[:num_terms]
                ]
            top_terms.append(', '.join(term_idcs))
        return top_terms

    def top_docs_per_val(self, num_docs):
        top_docs = []
        for n in range(0, self.K):
            doc_idcs = [
                self.pres[i[1]] for i in sorted(zip(self.Vt[n], range(0, self.Vt.shape[1])), reverse=True)[:num_docs]
                ]
            top_docs.append(', '.join(doc_idcs))
        return top_docs

    def avg_doc_query_dist_per_party(self, query):
        q = np.array(self.tf_idf.transform([query]).toarray()).transpose()
        # transform the query into K-dim column vector and normalize the result
        normed_q_hat = normalize(np.dot(self.U.transpose(), q), axis=0, norm='l2')
        # calculate the cosine similarity between the query and and the documents
        cos_sim_vec = np.dot(normed_q_hat.transpose(), self.normed_docs)
        # take all entries corresponding to wiki docs of either democratic or republican presidents
        d_similarity = [cos_sim_vec[0][i] for i in range(0, len(self.party)) if self.party[i] == 'D']
        r_similarity = [cos_sim_vec[0][i] for i in range(0, len(self.party)) if self.party[i] == 'R']
        # compute the average similarity for each political party and return the result
        return [sum(d_similarity) / len(d_similarity), sum(r_similarity) / len(r_similarity)]


def main():
    # load the documents along with their metadata
    presidents = []
    party_indices = []
    years = []
    documents = []
    ideo_docs = {}
    K = int(sys.argv[1])
    with open(sys.argv[2], 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['Year']) > 1888:
                presidents.append(row['President'])
                party_indices.append(row['Party'])
                years.append(row['Year'])
                documents.append(row['Text'])
    with open(sys.argv[3], 'rb') as ideocsv:
        reader = csv.DictReader(ideocsv)
        for row in reader:
            ideo_docs.update({row['Title']: (row['Party'], row['Text'])})

    # take the singular value decomposition of the documents' tf idf matrix
    tf_idf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    doc_tf_idf = np.array(tf_idf.fit_transform(documents).toarray()).transpose()
    U, s, Vt = np.linalg.svd(doc_tf_idf, full_matrices=False)

    # plot the singular values of the tf idf matrix
    plt.bar(range(0, len(s)), s)
    plt.title('Singular Values of the TF IDF Matrix')
    plt.savefig(os.path.join(sys.argv[4], 'sigmas.png'))
    plt.clf()

    # plot the transformed document vectors along the second and third dimensions
    transformed_docs = np.dot(np.dot(np.linalg.inv(np.diag(s)[:K, :K]), U[:, :K].transpose()), doc_tf_idf)
    dim_2_3 = transformed_docs[1:3, :]
    for (p, c) in COLORS.iteritems():
        x = [dim_2_3[0][i] for i in range(0, len(presidents)) if party_indices[i] == p]
        y = [dim_2_3[1][i] for i in range(0, len(presidents)) if party_indices[i] == p]
        plt.scatter(x, y, c=c, label=p)
    pres_initials = map(lambda p: ''.join([s[0].upper() for s in p.split()]), presidents)
    pres_year = [pres_initials[i] + ', ' + years[i] for i in range(0, len(pres_initials))]
    for pres, p_x, p_y in zip(pres_year, dim_2_3[0], dim_2_3[1]):
        if pres.startswith('JFK') or pres.startswith('GHWB'):
            p_y += 0.015
        if pres.startswith('BC'):
            p_y -= 0.02
        plt.annotate(pres, xy=(p_x + 0.01, p_y), size=7)
    plt.savefig(os.path.join(sys.argv[4], 'documents_along_dim_2_and_3.png'))

    # initialize query engine
    query_engine = PresLSAQueryMachine(presidents, party_indices, years, documents, K, tf_idf, doc_tf_idf, U, s, Vt)

    # print the top 15 terms and the top 10 wiki docs for each of the K singular values
    top_terms = query_engine.top_terms_per_val(15)
    top_docs = query_engine.top_docs_per_val(10)
    for i in range(0, K):
        print 'Top terms for singular value ' + str(i + 1) + ': ' + top_terms[i].encode('utf-8')
        print 'Top docs for singular value ' + str(i + 1) + ': ' + top_docs[i].encode('utf-8')
        print '\n\n'

    # find the cosine similarity between the (transformed) documents and the (transformed) queries
    for (k, v) in ideo_docs.iteritems():
        avg_distances = query_engine.avg_doc_query_dist_per_party(v[1])
        print 'Query: ' + k.encode('utf-8')
        print 'avg. distance to Democratic presidents = ' + str(avg_distances[0])
        print 'avg. distance to Republican presidents = ' + str(avg_distances[1])
        print '\n\n'

    # we formed 'reduced queries' containing a narrower set of terms we believed were especially
    # salient to each party's respective ideology. here we find the average cosine
    # distance between the reduced queries and the president docs.
    for q in REDUCED_QUERIES:
        avg_distances = query_engine.avg_doc_query_dist_per_party(q)
        print 'Query: ' + q.encode('utf-8')
        print 'avg. distance to Democratic presidents = ' + str(avg_distances[0])
        print 'avg. distance to Republican presidents = ' + str(avg_distances[1])
        print '\n\n'


if __name__ == '__main__':
    main()
