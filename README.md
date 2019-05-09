# data_mining_ai_exercises
factorize.py creates a utility matrix of users and ratings from a dataset of movie ratings in which each user does not have a rating for every movie. It then factorizes the utility matrix into 2 matrices, U and V, whose product could then be used for a recommendation system.

pagerank.py takes a dataset of links between nodes (representing webpages) and computes a PageRank score for each node. The graph has many deadends. pagerank.py removes each deadend until there are no more dead ends, computes the PageRank scores for each non-deadend node, and then re-adds the deadends in their reverse removal order. The score of each deadend is computed using the weighted sum of the scores of the nodes that link to it.
