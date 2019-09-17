# COMP9417 Assignment Recommender system using collaborative filtering
# Date: 07/25/2019
# Evaluation function: evaluating our result of these two comparing system
# For these two system, there are the same method to evaluate their rating
import os
import math

global test_path
global result_path
global test_dict
global result_dict

#test_path may be u1.test, u2.test, u3.test...
test_path = 'u5.test'
#It's different in user-base and item-base
result_path = 'users_result.txt'
#result_path = 'movies_result.txt'
test_dict = {}
result_dict = {}


class ResultEvaluation:
    def comparing_train_and_test(self):
        count = 0
        sum_RMSE = 0
        sum_MAE = 0
        with open(test_path, 'r') as file:
            for lines in file:
                (userId, movieId, rating, _) = lines.strip().split("\t")
                movieId = movieId.replace('\r\r\n', '')
                test_dict.setdefault(userId, {})
                test_dict[userId][movieId] = float(rating)

        with open(result_path, 'r') as f:
            for lines in f:
                (userId, movieId, rating) = lines.strip().split("\t")
                result_dict.setdefault(userId, {})
                result_dict[userId][movieId] = float(rating)

        for userid in test_dict.keys():
            for movieid in test_dict[userid]:
                diff = test_dict[userid][movieid] - result_dict[userid][movieid]
                count += 1
                sum_RMSE += diff ** 2
                sum_MAE += abs(diff)

        MAE = sum_MAE/count
        RMSE = math.sqrt(sum_RMSE/count)

        print(f'So the result is MAE: {MAE}, RMSE: {RMSE} and the count number is {count}')
        print(f"({MAE}, {RMSE}, {count})")


def user_base():
    result = ResultEvaluation()
    result.comparing_train_and_test()


if __name__ == '__main__':
    user_base()


