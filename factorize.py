import numpy as np
import time
import math

def factorize(M,d,T):
    n,m = M.shape
    #randomly initialize matrices U and V
    u = np.random.rand(n,d)
    v = np.random.rand(d,m)

    Ni = []
    Nj = []
    for i in range(n):
        Ni.append(np.flatnonzero(M[i])) #set of indices of items that user i rates
    for j in range(m):
        Nj.append(np.flatnonzero(M.T[j])) #set of indices of users that rate item i

    return factorize_recursive(M,u,v,n,m,d,Ni,Nj,T)

def factorize_recursive(M,u,v,n,m,d,Ni,Nj,T):
    if(T<0): #recursion done
        return u,v

    uv = np.zeros((n,m))

    #use equations from page 334 of http://infolab.stanford.edu/~ullman/mmds/ch9.pdf to get optimum value of each element of U and V
    for k in range(d-1):
        u_i_minus_k = []
        v_j_minus_k = []
        for i in range(n):
            u_i_minus_k.append(np.delete(u[i],k))
        for j in range(m):
            v_j_minus_k.append(np.delete(v.T[j],k))

        for i in range(n):
            numerator_sum = 0
            denominator_sum = 0
            for j in Ni[i]:
                numerator_sum += (np.dot(u_i_minus_k[i],v_j_minus_k[j]) - M[i][j])*v[k][j]
                denominator_sum += math.pow(v[k][j],2)

            x_i = -1 * (numerator_sum/denominator_sum)
            u[i][k] = x_i

        for j in range(m):
            numerator_sum = 0
            denominator_sum = 0
            for i in Nj[j]:
                numerator_sum += (np.dot(u_i_minus_k[i],v_j_minus_k[j]) - M[i][j])*u[i][k]
                denominator_sum += math.pow(u[i,k],2)

            y_j = -1 * (numerator_sum/denominator_sum)
            v[k][j] = y_j

    return factorize_recursive(M,u,v,n,m,d,Ni,Nj,T-1)

def calc_rmse(M,u,v): #root mean squared error
    uv = np.matmul(u,v)
    n,m = M.shape
    P_M = np.transpose(np.nonzero(M))
    numerator_sum = 0
    for p in P_M:
        i = p[0]
        j = p[1]
        numerator_sum += math.pow(np.dot(u[i],v.T[j]) - M[i][j], 2)

    return math.sqrt(numerator_sum/np.size(P_M))

def print_to_output(u,v,user_ids,movie_ids):
    u_rows = {}
    for i in range(len(user_ids)): #build formatted rows
        index = int(user_ids[i])
        row = user_ids[i]+"\t"
        for val in u[i][:-1]:
            row += str(val)+"\t"
        row += str(u[i][-1])+"\n"
        u_rows[index] = row

    with open('UT.tsv', 'w') as f:
        for key in sorted(u_rows): #sort in order of increasing user id
            f.write(u_rows[key])
        f.close()

    v_rows = {}
    for i in range(len(movie_ids)): #build formated rows
        index = int(movie_ids[i])
        row = movie_ids[i]+"\t"
        for val in v.T[i][:-1]: #COLUMNS
            row += str(val)+"\t"
        row += str(v.T[i][-1])+"\n"
        v_rows[index] = row

    with open('VT.tsv', 'w') as f:
        for key in sorted(v_rows): #sort in order of increasing movie id
            f.write(v_rows[key])
        f.close()



start = time.time()

#each line of u.data has 4 values: UserId MovieId Rating Timestamp, representing one user's rating of one movie
with open('u.data', 'r') as f:
    lines = [line.rstrip('\n').split()[:-1] for line in f]
    f.close()

user_ids = []
movie_ids = []
movie_id_index_map = {}
user_id_index_map= {}

#collect all unique movie and user ids
movie_index = 0
user_index = 0
for line in lines:
    user_id = line[0]
    movie_id = line[1]
    if(user_id not in user_ids):
        user_ids.append(user_id)
        user_id_index_map[user_id] = user_index
        user_index+=1
    if(movie_id not in movie_ids):
        movie_ids.append(movie_id)
        movie_id_index_map[movie_id] = movie_index
        movie_index+=1

#build utility matrix M
M = np.zeros((user_index,movie_index))
for line in lines:
    user_id = line[0]
    movie_id = line[1]
    rating = line[2]
    M[user_id_index_map[user_id]][movie_id_index_map[movie_id]] = rating

d = 2 #number of columns of U and number of rows of V
T = 20 #number of repetitions for factorization
u,v = factorize(M,d,T)
rmse = calc_rmse(M,u,v)
print("RMSE: "+str(rmse))
print_to_output(u,v,user_ids,movie_ids)

end = time.time()
print("elapsed time: "+str((end-start)/60.0)+" minutes")
