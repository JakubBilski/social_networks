import pathlib
import numpy as np
import tqdm
import statistics
import math
from scipy.sparse import csr_matrix
import scipy.sparse
import heapq
from tqdm.notebook import tqdm as tqdm1
import random

DATA_PATH = pathlib.Path('.').absolute() / 'data'
print(DATA_PATH)


VISUALIZATION_MODE = False
K = 3

print("Loading data...")
data = []
movie_id = -1

line_counter_poor_ram = 40_000_000
# line_counter_poor_ram = 5_000_000

print(f'{min(24058263, line_counter_poor_ram)}it - goal')

for i in range(1, 2):  # should be to range(1,5) but I dont have that much RAM XD
    with open(DATA_PATH / 'combined_data_1.txt') as f:
        for line in tqdm.tqdm(f):
            line = line[:-1]

            if line.endswith(':'):
                movie_id = int(line[:-1])
            else:
                splitted = line.split(',')
                data.append([splitted[0], movie_id, splitted[1]])

            line_counter_poor_ram -= 1
            if line_counter_poor_ram == 0:
                break


np.random.seed(2)

print("1st stage of converting data...")
data = np.array(data)
np.random.shuffle(data)
data = data.astype(int)


print("2nd stage of converting data...")
N = len(data)
test = data[:1000]
training = data[1000:]
tr1 = csr_matrix((training[:, 2], (training[:, 0], training[:, 1])), dtype=int)


print("Creating set of users...")
users = set(
    case[0] for case in training
)


def similarity_score(a_videos, u_videos):
    m1 = 0
    m2 = 0
    m3 = 0
    a_mi = np.mean(list(a_videos.values()))
    u_mi = np.mean(list(u_videos.values()))
    for movie in (a_videos.keys() & u_videos.keys()):
        m1 += (a_videos[movie] - a_mi)*(u_videos[movie] - u_mi)
        m2 += (a_videos[movie] - a_mi)*(a_videos[movie] - a_mi)
        m3 += (u_videos[movie] - u_mi)*(u_videos[movie] - u_mi)
    if m1 == 0 or m2 == 0 or m3 == 0:
        return 0
    return (m1/math.sqrt(m2))/math.sqrt(m3)


def get_videos(u, tr):
    return {
        movie: score
        for _, movie, score in zip(*scipy.sparse.find(tr.getrow(u)))
    }


def get_prediction(a_videos, best_matches, movie, tr1):
    a_mi = np.mean(list(a_videos.values()))
    a0 = 0
    a1 = 0
    for u, similarity in best_matches:
        u_videos = get_videos(u, tr1)
        if movie not in u_videos:
            print("This should never happen")
            continue
        u_mi = np.mean(list(u_videos.values()))
        a0 += (u_videos[movie] - u_mi)*similarity
        a1 += similarity
    if a1 == 0:
        return a_mi
    return a_mi + a0/a1

if not VISUALIZATION_MODE:

    print("Starting testing...")
    test_id = random.randint(0, 10000000)
    with open(f"results_{test_id}.txt", 'w') as f:
        print('User_id\tMovie_id\tReference\tHypothesis', file=f)

    for testcase in test:
        a = testcase[0]
        movie = testcase[1]
        a_videos = get_videos(a, tr1)
        scores = []
        for u in tqdm.tqdm(users):
            if u == a:
                continue
            u_videos = get_videos(u, tr1)
            if movie not in u_videos.keys():
                continue
            scores.append((u,
                similarity_score(
                    a_videos,
                    u_videos
                )))

        best_matches = heapq.nlargest(K, scores, key=lambda x: x[1])
        pred = int(np.round(get_prediction(a_videos, best_matches, movie, tr1)))
        with open(f"results_{test_id}.txt", 'a') as f:
            print(f'{testcase[0]}\t{testcase[1]}\t{testcase[2]}\t{pred}', file=f)
        print(f'{testcase[0]}\t{testcase[1]}\t{testcase[2]}\t{pred}')

else: # if VISUALIZATION_MODE

    print("Starting visualization run")
    testcase = test[0]
    a = testcase[0]
    movie = testcase[1]
    a_videos = get_videos(a, tr1)
    print(f"Our user test case: {a}")
    print("Movies watched by our test case:")
    print(a_videos)
    print(f"{len(list(a_videos.keys()))} watched movies")
    print(f"Mean rating given by our user: {np.mean(list(a_videos.values()))}")
    print("Movie that we want the prediction for:")
    print(movie)
    scores = []
    for u in tqdm.tqdm(users):
        if u == a:
            continue
        u_videos = get_videos(u, tr1)
        if movie not in u_videos.keys():
            continue
        scores.append((u,
            similarity_score(
                a_videos,
                u_videos
            )))

    best_matches = heapq.nlargest(K, scores, key=lambda x: x[1])
    print("Most similar users that watched the film:")
    for u, score in best_matches:
        print(f"User {u} (score {score})")
        u_videos = get_videos(u, tr1)
        print(f"Common movies: {u_videos.keys() & a_videos.keys()}")

    pred = int(np.round(get_prediction(a_videos, best_matches, movie, tr1)))
    print(f"Prediction: {pred}")
    print(f"Real rating: {testcase[2]}")
    # print(f'{testcase[0]}\t{testcase[1]}\t{testcase[2]}\t{pred}')
