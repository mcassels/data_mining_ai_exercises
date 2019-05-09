from __future__ import division
import os
import random

keywords = ["awful", "bad", "boring", "dull", "effective", "enjoyable", "great", "hilarious"]

def convert_file_to_vector(file):
    #indices of vector correspond to indices of keywords array
    vector = [0]*len(keywords)
    for line in open(file):
        for word in line.split():
            if word in keywords:
                vector[keywords.index(word)] = 1

    return vector

#return all file vectors within a directory
def get_file_vectors(directory):
    files = os.listdir(directory);
    return [convert_file_to_vector(directory+"/"+file) for file in files]

pos_file_vectors = get_file_vectors("review_polarity/txt_sentoken/pos")
neg_file_vectors = get_file_vectors("review_polarity/txt_sentoken/neg")

#returns array holding probability of each keyword
def calc_probabilities(file_vectors):
    #get the total number of containing files for each keyword
    totals = [0]*len(keywords)
    for vector in file_vectors:
        totals = [x + y for x, y in zip(totals, vector)]

    probabilities = [total/len(file_vectors) for total in totals]
    return probabilities

def calc_word_probability_given_review_sentiment(word,sentiment):
    word_index = keywords.index(word)
    file_vectors = []
    if(sentiment=="pos"):
        file_vectors = train_pos_file_vectors
    else:
        file_vectors = train_neg_file_vectors
    probabilities = calc_probabilities(file_vectors)
    return probabilities[word_index]

def calc_word_probability(word):
    word_index = keywords.index(word)
    probabilities = calc_probabilities(train_all_vectors)
    return probabilities[word_index]

#P(pos|word) = (P(pos)P(word|pos))/P(word)
def calc_probability_pos_given_word(word):
    prob_word_given_pos = calc_word_probability_given_review_sentiment(word,"pos")
    prob_word = overall_probabilities[keywords.index(word)]
    prob_pos_given_word = prob_pos*prob_word_given_pos/prob_word
    return prob_pos_given_word

#P(neg|word) = (P(neg)P(word|neg))/P(word)
def calc_probability_neg_given_word(word):
    prob_word_given_neg = calc_word_probability_given_review_sentiment(word,"neg")
    prob_word = overall_probabilities[keywords.index(word)]
    prob_neg_given_word = prob_neg*prob_word_given_neg/prob_word
    return prob_neg_given_word


def classify_file(file_vector):
    probability_pos = 1
    probability_neg = 1
    #go through the presence or absence of each keyword and multiply
    #the probability of a file with that sentiment
    #having the same presence value (0 or 1) for that word
    for i,val in enumerate(file_vector):
        if(val == 1):
            probability_pos *= probability_pos_given_each_keyword[i]
            probability_neg *= probability_neg_given_each_keyword[i]
        else:
            probability_pos *= 1-probability_pos_given_each_keyword[i]
            probability_neg *= 1-probability_neg_given_each_keyword[i]

    if(probability_pos>probability_neg):
        return "pos"
    else:
        return "neg"



#USING WHOLE DATASET FOR BOTH TRAINING AND TESTING
train_pos_file_vectors = pos_file_vectors
train_neg_file_vectors = neg_file_vectors

test_pos_file_vectors = pos_file_vectors
test_neg_file_vectors = neg_file_vectors

train_all_vectors = []
train_all_vectors.extend(train_pos_file_vectors)
train_all_vectors.extend(train_neg_file_vectors)

test_all_vectors = []
test_all_vectors.extend(test_pos_file_vectors)
test_all_vectors.extend(test_neg_file_vectors)

prob_pos = len(train_pos_file_vectors)/len(train_all_vectors)
prob_neg = len(train_neg_file_vectors)/len(train_all_vectors)
overall_probabilities = [calc_word_probability(word) for word in keywords]
probability_pos_given_each_keyword = [calc_probability_pos_given_word(word) for word in keywords]
probability_neg_given_each_keyword = [calc_probability_neg_given_word(word) for word in keywords]


p = []
for i in range(8):
    p.append(calc_word_probability_given_review_sentiment(keywords[i],"pos"))
print("probability each word given review is positive: "+str(p))
p = []
for i in range(8):
    p.append(calc_word_probability_given_review_sentiment(keywords[i],"neg"))
print("probability each word given review is negative: "+str(p)+"\n")

num_correct_pos = 0
num_correct_neg = 0
num_pos_classified_neg = 0
num_neg_classified_pos = 0

for i,file_vector in enumerate(test_all_vectors):
    prediction = classify_file(file_vector)
    actual = "pos"
    if(i>len(test_pos_file_vectors)):#since all the pos ones are at the beginning
        actual = "neg"
    if(prediction=="pos" and actual=="pos"):
        num_correct_pos+=1
    elif(prediction=="neg" and actual == "neg"):
        num_correct_neg+=1
    elif(prediction=="neg" and actual == "pos"):
        num_pos_classified_neg+=1
    elif(prediction=="pos" and actual == "neg"):
        num_neg_classified_pos+=1

accuracy = (num_correct_pos+num_correct_neg)/len(test_all_vectors)
print("USING WHOLE SET FOR TRAINING AND TESTING")
print("accuracy: "+str(accuracy))
print("num pos classified pos: "+str(num_correct_pos))
print("num neg classified neg: "+str(num_correct_neg))
print("num pos classified neg: "+str(num_pos_classified_neg))
print("num neg classified pos: "+str(num_neg_classified_pos)+"\n\n")


#K-FOLDS CROSS VALIDATION WITH K=10

def get_folds(k):
    folds = []
    for i in range(k):
        #each fold holds 100 pos and 100 neg files, in numerical order
        fold_i = pos_file_vectors[100*i:100*(i+1)] + neg_file_vectors[100*i:100*(i+1)]
        folds.append(fold_i)
    return folds

k = 10 #using 10 folds
folds = get_folds(k)

num_correct_pos = 0
num_correct_neg = 0
num_pos_classified_neg = 0
num_neg_classified_pos = 0

for i in range(len(folds)):
    test_all_vectors = folds[i] #use one fold for validation
    test_pos_file_vectors = test_all_vectors[:100] #first 100 are pos
    test_neg_file_vectors = test_all_vectors[100:] #second 100 are neg

    #vectors from all other folds go into training set
    train_all_vectors = []
    train_pos_file_vectors = []
    train_neg_file_vectors = []

    for j in range(len(folds)-1):
        train_all_vectors.extend(folds[j+1])
        train_pos_file_vectors.extend(folds[j+1][:100]) #first 100 are pos
        train_neg_file_vectors.extend(folds[j+1][100:]) #second 100 are neg

    # get probabilities for Bayes theorem
    prob_pos = len(train_pos_file_vectors)/len(train_all_vectors)
    overall_probabilities = [calc_word_probability(word) for word in keywords]
    probability_pos_given_each_keyword = [calc_probability_pos_given_word(word) for word in keywords]
    probability_neg_given_each_keyword = [calc_probability_neg_given_word(word) for word in keywords]

    for j,file_vector in enumerate(test_all_vectors):
        prediction = classify_file(file_vector)
        actual = "pos"
        if(j>len(test_pos_file_vectors)):#since all the pos ones are at the beginning
            actual = "neg"
        if(prediction=="pos" and actual=="pos"):
            num_correct_pos+=1
        elif(prediction=="neg" and actual == "neg"):
            num_correct_neg+=1
        elif(prediction=="neg" and actual == "pos"):
            num_pos_classified_neg+=1
        elif(prediction=="pos" and actual == "neg"):
            num_neg_classified_pos+=1

#total number of files classified = k * length of one fold i.e. size of whole dataset
accuracy = (num_correct_pos+num_correct_neg)/(len(test_all_vectors)*k)
print("USING 10-FOLD CROSS VALIDATION")
print("accuracy: "+str(accuracy))
print("num pos classified pos: "+str(num_correct_pos))
print("num neg classified neg: "+str(num_correct_neg))
print("num pos classified neg: "+str(num_pos_classified_neg))
print("num neg classified pos: "+str(num_neg_classified_pos)+"\n")



def generate_movie_review(sentiment):
    review_vector = [] #will be a vector of length 8
    for i in range(8):
        word_present = 0 #flag indicating whether word will appear in review
        rand_val = random.random()
        # if the random value is below the probability, the word will appear
        # otherwise it will not. This ensures the distribution of the words follows the
        # probabilities of them appearing using the training data
        if(rand_val <= calc_word_probability_given_review_sentiment(keywords[i],sentiment)):
            word_present = 1
        review_vector.append(word_present)

    review_words = [word for word in keywords if review_vector[keywords.index(word)]==1]
    return review_words

print("POSITIVE REVIEWS\n")
for i in range(5):
    print(str(generate_movie_review("pos"))+"\n")

print("NEGATIVE REVIEWS\n")
for i in range(5):
    print(str(generate_movie_review("neg"))+"\n")
