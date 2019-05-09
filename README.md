# data_mining_ai_exercises
**factorize.py** creates a utility matrix of users and ratings from a dataset of movie ratings in which each user does not have a rating for every movie. It then factorizes the utility matrix into 2 matrices, U and V, whose product could then be used for a recommendation system.

**pagerank.py** takes a dataset of links between nodes (representing webpages) and computes a PageRank score for each node. The graph has many deadends. pagerank.py removes each deadend until there are no more dead ends, computes the PageRank scores for each non-deadend node, and then re-adds the deadends in their reverse removal order. The score of each re-added deadend is computed using the weighted sum of the scores of the nodes that link to it.

**stochastic_gradient_descent** trains a linear regression model using stochastic gradient descent. It takes a tsv file where each row is the features of a datapoint (test set had 100,000 points and 300 features) and outputs a tsv containing the co-efficients of each feature for the linear regression model.
