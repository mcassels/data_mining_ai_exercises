# data_mining_ai_exercises
**factorize.py** creates a utility matrix of users and ratings from a dataset of movie ratings in which each user does not have a rating for every movie. It then factorizes the utility matrix into 2 matrices, U and V, whose product could then be used for a recommendation system.

**pagerank.py** takes a dataset of links between nodes (representing webpages) and computes a PageRank score for each node. Deadends and spider traps in the graph are handled.

**stochastic_gradient_descent** trains a linear regression model using stochastic gradient descent. It takes a tsv file where each row is the features of a datapoint (test set had 100,000 points and 300 features) and outputs a tsv containing the co-efficients of each feature for the linear regression model.

**naive_bayes.py** takes a dataset of positive and negative movie reviews and uses a predetermined set of keywords with high polarity values (e.g. "awful" and "hilarious") as features for naive bayes classification. It produces 2 confusion matrices, one for classification using the entire dataset for training and testing, and one for classification using K-folds cross-validation. It also generates movie review vectors given a sentiment value (positive or negative).
