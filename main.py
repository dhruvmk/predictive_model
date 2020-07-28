import pandas as pd
import numpy as np

df = pd.read_csv("/home/dhruv/Documents/python/movie reviews/IMDB Dataset.csv")
x = np.array(df["review"])
y = np.array(df["sentiment"])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_x)
test_vectors = vectorizer.transform(test_x)

from sklearn import svm
svc = svm.SVC(kernel = "linear")
svc.fit(train_vectors, train_y)

from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(train_vectors.toarray(), train_y)

from sklearn.neighbors import KNeighborsClassifier
clf_knc = KNeighborsClassifier(n_neighbors=5)
clf_knc.fit(train_vectors, train_y)

from sklearn.neighbors import NearestCentroid
clf_ncc = NearestCentroid()
clf_ncc.fit(train_vectors, train_y)


from sklearn.linear_model import SGDClassifier
clf_sgd = SGDClassifier(alpha=0.0001, average=False,
                                     class_weight=None, early_stopping=False,
                                     epsilon=0.1, eta0=0.0, fit_intercept=True,
                                     l1_ratio=0.15, learning_rate='optimal',
                                     loss='hinge', max_iter=1000,
                                     n_iter_no_change=5, n_jobs=None,
                                     penalty='l2', power_t=0.5,
                                     random_state=None, shuffle=True, tol=0.001,
                                     validation_fraction=0.1, verbose=0,
                                     warm_start=False)
clf_sgd.fit(train_vectors, train_y)



from sklearn.linear_model import LogisticRegression
clf_log=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                          fit_intercept=True,
                                          intercept_scaling=1, l1_ratio=None,
                                          max_iter=100, multi_class='warn',
                                          n_jobs=None, penalty='l2',
                                          random_state=None, solver='warn',
                                          tol=0.0001, verbose=0,
                                          warm_start=False)
clf_log.fit(train_vectors, train_y)





