import time

def get_dead_ends(num_pages,edge_list,id_index_map,page_ids):
    L = [[] for i in range(num_pages)] #adjacency list
    D = [0] * num_pages  #degree array
    original_D = [0] * num_pages  #degree array
    dead_end_indices = []

    i = 0
    for edge in edge_list:
        i1 = id_index_map[edge[0]]
        i2 = id_index_map[edge[1]]
        L[i2].append(i1) #L[i] is the set of nodes that link to i
        D[i1]+=1 #D[i] is the out degree of i
        original_D[i1]+=1 #D[i] is the out degree of i
        i+=1

    dead_ends = []
    dead_end_found = True
    while(dead_end_found):
        dead_end_found = False
        new_dead_ends = []
        i = 0
        for deg in D: #for each out degree
            if(deg == 0):
                new_dead_ends.append(i)
                dead_end_found = True #keep looking
                dead_ends.append(page_ids[i])
                D[i] = -1 #so it won't be checked again
                dead_end_indices.append(i)
            i+=1

        for end_i in new_dead_ends:
            nodes_linked_to_i = L[end_i]
            for node_i in nodes_linked_to_i:
                D[node_i]-=1

    return D,L,dead_end_indices,original_D

def get_pageid_index_mappings(edge_list):
    id_index_map = {}
    pageids_with_dups = [p for edge in edge_list for p in edge]
    page_ids = list(set(pageids_with_dups))
    i = 0
    for id in page_ids:
        id_index_map[id] = i
        i+=1
    return id_index_map,page_ids

def print_to_output(page_ranks,page_ids):
    with open('PR_800k.tsv', 'w') as f:
        f.write("PageRank\tIds\n")
        pr_ids = []
        for i in range(len(page_ranks)):
            id = page_ids[i]
            pr = page_ranks[i]
            pr_ids.append([pr,id])

        for pr_id in sorted(pr_ids,reverse=True): #sort by descending order of PRs
            f.write(str(pr_id[0])+"\t"+str(pr_id[1])+"\n")
        f.close()

def get_PRs_no_dead_ends(num_pages,D,L,dead_end_indices):
    Np = num_pages - len(dead_end_indices)
    initial_score = 1.0/Np
    PRs = [initial_score]*num_pages #this unecessarily sets dead ends but they will be overwritten

    B = 0.85
    T = 10
    adding_term = (1.0-B)*(1.0/Np)
    return compute_page_ranks(B,T,D,L,PRs,Np,adding_term)

def compute_page_ranks(B,T,D,L,PRs,Np,adding_term):
    if(T==0): #end recursion
        return PRs

    new_PRs = [0.0]*num_pages
    for i in range(num_pages):
        if (D[i] != -1): #skip dead ends
            nodes_linked_to_i = L[i]
            sum = 0
            for node_j_index in nodes_linked_to_i:
                deg_j = D[node_j_index]
                sum += float(PRs[node_j_index])/deg_j
            sum *= B
            sum += adding_term
            new_PRs[i] = sum

    return compute_page_ranks(B,T-1,D,L,new_PRs,Np,adding_term)

def add_dead_end_scores(page_rank_scores,D,L,dead_end_indices,original_D):
    i = len(dead_end_indices) - 1 #start at the last removed dead end
    while(i > -1):
        node_index = dead_end_indices[i]
        nodes_linked_to_i = L[node_index]
        sum = 0
        for node_j_index in nodes_linked_to_i:
            deg_j = original_D[node_j_index]
            sum += page_rank_scores[node_j_index]/deg_j
        page_rank_scores[node_index] = sum
        i-=1
    return page_rank_scores


start = time.time()
with open('web-Google.txt', 'r') as f:
    lines = [line.rstrip('\n').split() for line in f]
    num_pages = int(lines[2][2])
    edge_list = lines[4:]
    f.close()

id_index_map,page_ids = get_pageid_index_mappings(edge_list)
#dead_end_indices are in removal order
D,L,dead_end_indices,original_D = get_dead_ends(num_pages,edge_list,id_index_map,page_ids)
PRs_no_dead_ends = get_PRs_no_dead_ends(num_pages,D,L,dead_end_indices)
PRs = add_dead_end_scores(PRs_no_dead_ends,D,L,dead_end_indices,original_D)
print_to_output(PRs,page_ids)

end = time.time()
print("elapsed time: "+str(end-start)+" seconds")
