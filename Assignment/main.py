# COMP9417 Assignment Recommender System using collaborative filtering
# Date: 07/23/2019
# Main function: Estimated scores of two system: Item-base and User-base Recommender System
import os
import json
import math

cwd = os.getcwd()
global training_data
global item_training_data
global test_data
global top_k
global json_file_users
global json_file_items
global user_result_file
global item_result_file


#top_k is the nearest neighbor, in our assignment, it is equal to 10,20,30,40,50...100.
top_k = 100
json_file_users = "similarity_users.json"
json_file_items = "similarity_movies.json"
user_result_file = "users_result.txt"
item_result_file = "movies_result.txt"


# ########################################## save the file in the json #################################################
def save_similarity_scores(item, filename, file_path=cwd):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(os.path.join(file_path, filename), 'w') as file:
            json.dump(item, file, indent=4)
        file.close()


def load_file(filename, file_path=cwd):
        with open(os.path.join(file_path, filename), 'r') as file:
            data = json.load(file)
        return data


# ######################################### loading the training data #################################################
# From the file getting the items and then build a dictionary to calculate
# The dict format is {UserId:{MovieId: Rating}}
#u1.base,u2.base,u3.base...
train_path = "u5.base"
training_data = {}
item_training_data = {}
average_movieId = {}
with open(train_path) as f:
    for line in f:
        (UserId, MovieId, Rating, _) = line.strip().split("\t")
        if MovieId not in average_movieId:
            average_movieId[MovieId] = [0, 0]
        else:
            average_movieId[MovieId][0] = average_movieId[MovieId][0] + float(Rating)
            average_movieId[MovieId][1] += 1
        training_data.setdefault(UserId, {})
        training_data[UserId][MovieId] = float(Rating)
        item_training_data.setdefault(MovieId, {})
        item_training_data[MovieId][UserId] = float(Rating)
# print(f'average score is {average_movieId}')
sum_ave = 0
count_ave = 0
for movies in average_movieId.keys():
    if average_movieId[movies][1] != 0:
        ave = float(average_movieId[movies][0])/float(average_movieId[movies][1])
        sum_ave = float(sum_ave + ave)
        count_ave += 1
print(f'average score is:{sum_ave/count_ave}')
# print(training_data)

# ######################################### loading the test data ######################################################
#u1.test,u2.test,u3.test...It must be same with the filename in the evaluation,py
test_path = "u5.test"
test_data = []
with open(test_path) as predict:
    for line in predict:
        (userId, movieId, _, _) = line.strip().split('\t')
        movieId = movieId.replace('\r\r\n', '')
        test_data.append((userId, movieId))

# print(f'test data is {test_data}')


# ######################################### User-based Recommender System ##############################################
# ###function
#
def pearson_sim(target, others):
    user1_data = training_data[target]
    user2_data = training_data[others]
    common = {}

    # finding the movies that both users marked
    for key in user1_data.keys():
        if key in user2_data.keys():
            common[key] = 1
    if len(common) == 0:
        return 0  # if there is not common movies
    n = len(common)  # the number of common movies
    # print(f'the number of common movies is {n} and the dict is {common}')
    length_user1 = len(user1_data)
    length_user2 = len(user2_data)
    length = max(length_user1, length_user2)

    # calculating the sum of scores
    sum1 = sum([float(user1_data[movie]) for movie in common])
    sum2 = sum([float(user2_data[movie]) for movie in common])

    # calculating the sum of squares of scores
    sum1Sq = sum([pow(float(user1_data[movie]), 2) for movie in common])
    sum2Sq = sum([pow(float(user2_data[movie]), 2) for movie in common])

    # calculating the sum of products
    PSum = sum([float(user1_data[it]) * float(user2_data[it]) for it in common])

    # calculating the correlation coefficient
    num = float(PSum - (sum1 * sum2 / n))
    den = float(math.sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n)))
    if den == 0:
        return 0
    r = float(num / den) * (float(n) / float(length))
    return round(r, 7)


def top_k_similarity(users):
    top_k_result = []
    for userid in training_data.keys():
        if userid != users:
            similar = pearson_sim(users, userid)
            value = (similar, userid)
            top_k_result.append(value)
    top_k_result.sort()
    top_k_result.reverse()
    return top_k_result[0:top_k]


# ####################################### User-base system class #######################################################
class UserRecommenderSystem:
    def __init__(self):
        self.data_similarity = {}

    def find_similarity_users(self):
        similarity_users = {}
        for key in training_data.keys():
            similarity_users.setdefault(key, [])

        for users in training_data.keys():   # traverse the data set
            top_k_result = top_k_similarity(users)
            similarity_users[users] = top_k_result
        save_similarity_scores(similarity_users, json_file_users)
        self.data_similarity = similarity_users

    def predict_score(self, users, movies):
        total_ratings = 0.0
        similarity_sum = 0.0
        if self.data_similarity == {}:
            self.find_similarity_users()

        # if users never score or he doesn't have any similarity users
        # his scoring will be the movies average score
        if users not in self.data_similarity:
            # if the movies never be scored, we randomly make the movies' scores equal to 4.0
            if movies not in average_movieId:
                predict_score = 3.0
                return predict_score

            if average_movieId[movies][1] != 0:
                predict_score = float(average_movieId[movies][0]/average_movieId[movies][1])
                return predict_score
            else:
                predict_score = 3.0
                return predict_score

        # loop over all user in the list
        for other_user in self.data_similarity[users]:
            # don't compare to myself
            if other_user[1] == users:
                continue
            similarity = other_user[0]

            # loop continue if there is no similarity
            if similarity <= 0:
                continue

            # do predict for all those item which the user have not rated yet
            if movies not in training_data[users] or training_data[users][movies] == 0:
                # we can find the movie in other users
                if movies in training_data[other_user[1]]:
                    # get weighted ratings from all users in similar
                    total_ratings += training_data[other_user[1]][movies] * similarity
                    similarity_sum += similarity

        # if there still have the similarity users, using the average score of the movie
        if similarity_sum == 0:
            # if the movies never be scored
            if movies not in average_movieId:
                predict_score = 3.0
                return predict_score

            if average_movieId[movies][1] != 0:
                predict_score = float(average_movieId[movies][0]/average_movieId[movies][1])
                return predict_score
            else:
                predict_score = 3.0
                return predict_score

        else:
            predict_score = total_ratings / similarity_sum
        return predict_score

    def predict_result(self):
        with open(user_result_file, 'w') as file:
            for val in test_data:
                estimated_score = self.predict_score(val[0], val[1])
                file.write(val[0] + '\t' + val[1] + '\t' + str(estimated_score) + '\n')
        file.close()


def user_base():
    result = UserRecommenderSystem()
    result.find_similarity_users()
    result.predict_result()


# ######################################### Item-based Recommender System ##############################################
# ###function
#
def cosine_sim(target, others):
    movie1_data = item_training_data[target]
    movie2_data = item_training_data[others]
    common = {}

    # finding the common movies
    for key in movie1_data.keys():
        if key in movie2_data.keys():
            common[key] = 1
    if len(common) == 0:
        return 0  # if there is not common users
    n = len(common)  # the number of common users
    total_count = len(movie1_data) + len(movie2_data) - n
    x = math.sqrt(sum([movie1_data[it] ** 2 for it in common]))
    y = math.sqrt(sum([movie2_data[it] ** 2 for it in common]))
    xy = sum([movie1_data[it] * movie2_data[it] for it in common])
    cos = xy / (x * y)
    result = cos * (float(n) / float(total_count))
    return round(result, 7)


def top_k_similarity_items(movies):
    top_k_result = []
    for movieid in item_training_data.keys():
        if movieid != movies:
            similar = cosine_sim(movies, movieid)
            value = (similar, movieid)
            top_k_result.append(value)
    top_k_result.sort()
    top_k_result.reverse()
    return top_k_result[0:top_k]


# ####################################### Item-base system class #######################################################
class ItemRecommenderSystem:
    def __init__(self):
        self.movies_similarity = {}

    def find_similarity_items(self):
        similarity_items = {}
        for key in item_training_data.keys():
            similarity_items.setdefault(key, [])

        for movies in item_training_data.keys():   # traverse the data set
            top_k_result = top_k_similarity_items(movies)
            similarity_items[movies] = top_k_result
        save_similarity_scores(similarity_items, json_file_items)
        self.movies_similarity = similarity_items

    def predict_score(self, users, movies):
        total_ratings = 0.0
        similarity_sum = 0.0
        if self.movies_similarity == {}:
            self.find_similarity_items()

        # if movies never be scored
        if movies not in self.movies_similarity:
            # if the movies never be scored, we randomly make the movies' scores equal to 4.0
                predict_score = 3.0
                return predict_score

        for other_movie in self.movies_similarity[movies]:
            # don't compare to myself
            if other_movie[1] == movies:
                continue
            similarity = other_movie[0]

            # loop continue if there is no similarity
            if similarity <= 0:
                continue

            # do predict for all those item which the user have not rated yet
            if movies not in training_data[users] or training_data[users][movies] == 0:
                # we can find the movie in other users
                if users in item_training_data[other_movie[1]]:
                    # get weighted ratings from all users in similar
                    total_ratings += item_training_data[other_movie[1]][users] * similarity
                    similarity_sum += similarity

        # if there still have the similarity users, using the average score of the movie
        if similarity_sum == 0:
            predict_score = 3.0
        else:
            predict_score = total_ratings / similarity_sum
        return predict_score

    def predict_result(self):
        with open(item_result_file, 'w') as file:
            for val in test_data:
                estimated_score = self.predict_score(val[0], val[1])
                file.write(val[0] + '\t' + val[1] + '\t' + str(estimated_score) + '\n')
        file.close()


def item_base():
    result = ItemRecommenderSystem()
    result.find_similarity_items()
    result.predict_result()


if __name__ == '__main__':
    #user-base
    user_base()
    #item-base
    #item_base()
