import pandas as pd
import os
import numpy as np
import pickle
# from multiprocessing import Pool
# import timeit
# from itertools import repeat
# from scipy.sparse import csr_matrix
# from pandas.api.types import CategoricalDtype

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class Dataset:
    def __init__(self, params):
        cwd = os.getcwd()
        self.dataset_dir = os.path.join(cwd, '..\\Book-Crossing Dataset')
        self.params = params
        self.users_df = self.load_users()
        self.books_df = self.load_books()
        self.ratings_df = self.load_ratings()
        self.pre_processing_rating()

        self.test_ratings_df = self.load_ratings_test_data()

    def load_users(self):
        users_df = []
        try:
            pickle_path = os.path.join(self.dataset_dir, 'Users.pkl')
            # check if pickle file is exists, then skip processing data
            if os.path.exists(pickle_path):
                users_df = pd.read_pickle(pickle_path)
            else:
                csv_path = os.path.join(self.dataset_dir, 'Users.csv')
                users_df = pd.read_csv(
                    csv_path, sep=';', encoding=self.params.get('encoding', None))
                users_df["Location"] = users_df["Location"].str.replace(
                    r'\r\n', '', regex=True)
        except Exception as ex:
            print(ex)

        return users_df

    def load_books(self):
        books_df = []
        try:
            pickle_path = os.path.join(self.dataset_dir, 'Books.pkl')
            # check if pickle file is exists, then skip processing data
            if os.path.exists(pickle_path):
                books_df = pd.read_pickle(pickle_path)
            else:
                csv_path = os.path.join(self.dataset_dir, 'Books.csv')
                books_df = pd.read_csv(
                    csv_path, sep=r'(?<=\");(?=\")', encoding=self.params.get('encoding', None), engine='python')
                books_df.columns = books_df.columns.str.strip('"')
                books_df = books_df.apply(lambda x: x.str.strip('"'))
                books_df = books_df.apply(
                    lambda x: x.str.replace('&amp;', '&'))
        except Exception as ex:
            print(ex)

        return books_df

    def transform_ISBN_format(self, ISBN):
        ISBN = str(ISBN)
        if ISBN.startswith('ISBN'):
            ISBN = ISBN.split('ISBN')[1]
        if len(ISBN) != 10 and len(ISBN) != 13:
            ISBN = None
        return ISBN

    def load_ratings(self):
        ratings_df = []
        try:
            pickle_path = os.path.join(self.dataset_dir, 'Ratings.pkl')
            # check if pickle file is exists, then skip processing data
            if os.path.exists(pickle_path):
                ratings_df = pd.read_pickle(pickle_path)

                # p_userid = 188593
                # user_rating_books = ratings_df.loc[ratings_df['User-ID']== p_userid, 'ISBN'].tolist()
                # print(f"user_rating_books:\n{user_rating_books}")
            else:
                csv_path = os.path.join(self.dataset_dir, 'Ratings.csv')
                ratings_df = pd.read_csv(
                    csv_path, sep=";", encoding=self.params.get('encoding', None))

                ratings_df["ISBN"] = ratings_df["ISBN"].apply(
                    lambda x: x.strip('"').strip("\\"))

                ratings_df["ISBN"] = ratings_df["ISBN"].apply(
                    self.transform_ISBN_format)

                ratings_df = ratings_df[ratings_df['ISBN'].isin(
                    self.books_df['ISBN'])]

                ratings_df = ratings_df[ratings_df['User-ID'].isin(
                    self.users_df['User-ID'])]
                
                ratings_df['User-ID'] = ratings_df['User-ID'].apply(str)
                ratings_df['ISBN'] = ratings_df['ISBN'].apply(str)

                ratings_df.to_pickle(os.path.join(
                    self.dataset_dir, 'Ratings.pkl'))
        except Exception as ex:
            print(ex)

        return ratings_df

    def pre_processing_rating(self):
        # remove rating score of 0
        rating = self.ratings_df[self.ratings_df['Book-Rating']
                                 != 0].reset_index(drop=True)

        # print(f'rating head():\n{rating.head()}')
        # print(f'rating.shape:\n{rating.shape}')
        rating_groupby_ISBN = rating.groupby(['ISBN'])
        # avg_book_ratings = rating_groupby_ISBN['Book-Rating'].mean()
        # avg_book_ratings.name = 'book ratings average'
        # print(f'avg_book_ratings:{avg_book_ratings.head()}')

        num_book_ratings = rating_groupby_ISBN['Book-Rating'].count()
        num_book_ratings.name = 'number of book ratings'
        # mean_num_of_book_ratings = num_book_ratings.mean()
        # print(f'mean_num_of_book_ratings:{mean_num_of_book_ratings}')
        # print(f'num_book_ratings.quantile(.5):{num_book_ratings.quantile(.5)}')

        mean_book_ratings = rating_groupby_ISBN['Book-Rating'].mean()
        mean_book_ratings.name = 'mean of book ratings score'
        rating = rating.join(num_book_ratings, on="ISBN")
        rating = rating.join(mean_book_ratings, on="ISBN")
        # print(f'rating.head:\n{rating.head()}')

        rating_groupby_user = rating.groupby(['User-ID'])
        num_user_ratings = rating_groupby_user['User-ID'].count()
        num_user_ratings.name = 'number of user gave ratings'
        # mean_num_of_user_ratings = num_user_ratings.mean()
        # print(f'mean_num_of_user_ratings:{mean_num_of_user_ratings}')

        rating = rating.join(num_user_ratings, on="User-ID")
        # print(f'rating.head:\n{rating.head()}')

        # filter the rating record that:
        # 1. number of book ratings needs to bigger than number of book ratings mean.
        # 2. number of book ratings needs to bigger than number of book ratings mean.
        # rating = rating.loc[rating['number of book ratings']
        #                     > mean_num_of_book_ratings]
        # rating = rating.loc[rating['number of user gave ratings']
        #                     > mean_num_of_user_ratings]
        # rating = rating.sort_values(
        #     by=['number of user gave ratings', 'number of book ratings'], ascending=False)

        self.ratings_df = rating

    def load_ratings_test_data(self):
        test_ratings_df = []
        try:
            pickle_path = os.path.join(self.dataset_dir, 'Ratings_testX.pkl')
            test_result_csv_path = os.path.join(os.getcwd(), '509557023-org.csv')
            # check if pickle file is exists, then skip processing data
            if os.path.exists(test_result_csv_path):
                test_ratings_df = pd.read_csv(test_result_csv_path, sep=",")
            elif os.path.exists(pickle_path):
                test_ratings_df = pd.read_pickle(pickle_path)
            else:
                csv_path = os.path.join(self.dataset_dir, 'Ratings_testX.csv')
                test_ratings_df = pd.read_csv(
                    csv_path, sep=",", encoding=self.params.get('encoding', None), dtype=str)

                # test_ratings_df.to_pickle(os.path.join(
                #     self.dataset_dir, 'Ratings_testX.pkl'))
        except Exception as ex:
            print(ex)

        return test_ratings_df

    def get_mean_book_rating_score(self, ISBN):
        ratings = self.ratings_df
        return ratings.loc[ratings['ISBN'] == ISBN, 'mean of book ratings score'].iloc[0]

    def get_rating(self, user_id, ISBN):
        ratings = self.ratings_df
        return ratings.loc[(ratings['User-ID'] == user_id) & (
            ratings['ISBN'] == ISBN), 'Book-Rating'].iloc[0]

    def get_recommend_items_to_user(self, user_id, most_similar_users):
        # get rating record
        ratings = self.ratings_df

        # get user tried(already give rating) books list
        user_tried_books = ratings.loc[ratings['User-ID']
                                       == user_id, 'ISBN'].tolist()

        # find the similar user rating books that user not tried, and get the mean rating score of book
        cadidate_books = []
        recommend_books_score = []
        for s_user_id in most_similar_users:
            if most_similar_users[s_user_id] <=0: # similarity <=0
                continue
            for ISBN in ratings.loc[ratings['User-ID'] == s_user_id, 'ISBN'].tolist():
                if ISBN not in user_tried_books + cadidate_books:
                    cadidate_books.append(ISBN)
                    recommend_books_score.append(tuple((self.get_mean_book_rating_score(
                        ISBN), ratings.loc[(ratings['ISBN'] == ISBN), 'number of book ratings'].iloc[0], ISBN)))

        # order by desc (the front books have the higher score)
        # if two of scores are the same, then sort the number of rating
        recommend_books_score.sort(
            key=lambda tup: (tup[0], tup[1]), reverse=True)
        # print(f"cadidate_books:{cadidate_books}\n")
        # print(ratings.head())
        # print(f"recommend_books_score: {recommend_books_score}")
        return recommend_books_score

    def get_user_mean_book_rating(self, user_id):
        ratings = self.ratings_df
        have_rating_books = ratings.loc[ratings['User-ID']
                                        == user_id, 'ISBN'].tolist()

        if len(have_rating_books) == 0:
            return 0

        mean_rating = ratings_df.loc[ratings_df['User-ID']
                                     == user_id, 'Book-Rating'].mean()
        return mean_rating

    def pearson_correlation_score(self, user1, user2):
        '''
        user1 & user2 : user ids of two users between which similarity score is to be calculated.
        '''
        ratings = self.ratings_df
        # A list of books rating by both the users.
        both_rating_books = []

        # Finding books rating by both the users.
        for ISBN in ratings.loc[ratings['User-ID'] == user1, 'ISBN'].tolist():
            if ISBN in ratings.loc[ratings['User-ID'] == user2, 'ISBN'].tolist():
                both_rating_books.append(ISBN)

        # print(f'len(both_rating_books):\n{len(both_rating_books)}')
        # Returning '0' correlation for bo common books.
        if len(both_rating_books) == 0:
            return 0

        # # Calculating Co-Variances.
        mean_rating_1 = ratings.loc[ratings['User-ID']
                                    == user1, 'Book-Rating'].mean()
        mean_rating_2 = ratings.loc[ratings['User-ID']
                                    == user2, 'Book-Rating'].mean()

        # print(f'rating_sum_2:{rating_sum_2}')

        rating_squared_sum_1 = np.sum(
            [np.power((self.get_rating(user1, ISBN)-mean_rating_1), 2) for ISBN in both_rating_books])
        rating_squared_sum_2 = np.sum(
            [np.power((self.get_rating(user2, ISBN)-mean_rating_2), 2) for ISBN in both_rating_books])
        numerator = np.sum([(self.get_rating(user1, ISBN)-mean_rating_1) *
                            (self.get_rating(user2, ISBN)-mean_rating_2) for ISBN in both_rating_books])

        # Returning pearson correlation between both the users.
        denominator = np.sqrt(rating_squared_sum_1) * \
            np.sqrt(rating_squared_sum_2)

        # Handling 'Divide by Zero' error.
        if denominator == 0:
            return 0

        p_score = round(numerator/denominator, 3)
        print(f"Pearson Corelation between user {user1} & {user2}: {p_score}")
        return p_score

    def save_predict_books_rating(self, user_id, ISBN, predict_score):
        # correct prediction score
        if predict_score > 10:
            predict_score = 10
        if predict_score < 0:
            predict_score = 0

        test_ratings = self.test_ratings_df
        test_ratings.loc[(test_ratings['User-ID'] == user_id) & (
            test_ratings['ISBN'] == ISBN), 'Predict-Book-Rating'] = predict_score
        test_ratings.to_csv('509557023-org.csv', index=False)
        result = test_ratings['Predict-Book-Rating']
        result = result.fillna(0)
        result =  result.astype('int')
        result.to_csv('509557023.csv', index=False, header=False)
        print(
            f"saved user:{user_id}, ISBN:{ISBN}, predict_score:{predict_score}")
        

if __name__ == '__main__':
    params = {
        'encoding': 'ISO-8859-1',  # utf-8 might get error
    }
    ds = Dataset(params)

    ratings_df = ds.ratings_df
    # ratings_df = ds.ratings_df.head(20)
    # print(ratings_df)
    # user_id = sorted(ratings_df['User-ID'].unique())
    # isbns = sorted(ratings_df['ISBN'].unique())
    # user_ids_c = CategoricalDtype(
    #     sorted(ratings_df['User-ID'].unique()), ordered=True)
    # isbns_c = CategoricalDtype(
    #     sorted(ratings_df['ISBN'].unique()), ordered=True)
    # row = ratings_df['User-ID'].astype(user_ids_c).cat.codes
    # col = ratings_df['ISBN'].astype(isbns_c).cat.codes
    # sparse_matrix = csr_matrix((ratings_df['Book-Rating'], (row, col)), shape=(
    #     user_ids_c.categories.size, isbns_c.categories.size))
    # dfs = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
    # print(dfs)

    k = 10
    similarity = {}
    if os.path.exists('user-similarity.pkl'):
        with open('user-similarity.pkl', 'rb') as fp:
            similarity = pickle.load(fp)

    if 'Predict-Book-Rating' in ds.test_ratings_df:
        test_items = ds.test_ratings_df.loc[ds.test_ratings_df['Predict-Book-Rating'].isna()].to_dict(
            'records')
    else:
        test_items = ds.test_ratings_df.to_dict('records')

    total_len = len(test_items)
    processed_len = 0

    # 測試少量用
    # test_items = test_items[:5]
    for row in test_items:
        processed_len += 1
        print(f"processed percentage:{(processed_len/total_len): .0%}")

        predict_score = 0
        userA = row['User-ID']
        ISBN = row['ISBN']
        # 1. 先把除了自己以外，有 rating 過這本書的 unique user list 找出來
        unique_book_rating_users = sorted(ratings_df.loc[(ratings_df['ISBN']
                                                         == ISBN) & (ratings_df['User-ID'] != userA), 'User-ID'].unique().tolist())
        
        if len(unique_book_rating_users) == 0:
            predict_score = round(ds.get_user_mean_book_rating(userA))
            ds.save_predict_books_rating(userA, ISBN, predict_score)
            continue

        print(
            f"unique_book_rating_users length:{len(unique_book_rating_users)}")

        # cadidate_users
        most_similar_users = {}
        # 2. 查詢 這個 user 與 unique user list 的 simularity，沒有在 similarity dict 裡面的話，才補算
        for userB in unique_book_rating_users:
            if userA not in similarity:
                similarity[userA] = {}

            if userB not in similarity:
                similarity[userB] = {}

            if userB not in similarity[userA]:
                s_score = ds.pearson_correlation_score(userA, userB)
                print(f"user:{userA} and user:{userB} similarity is:{s_score}")
                # 寫入 userA > userB 以及 userB > userA 的 dict (兩邊對應的 dict 都要可以找的到)
                similarity[userA][userB] = s_score
                similarity[userB][userA] = s_score

            # (similarity score of userA and userB)
            most_similar_users[userB] = similarity[userA][userB]

        # 儲存最新的 similarity dict 資料
        with open('user-similarity.pkl', 'wb') as fp:
            pickle.dump(similarity, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # 3. 算完之後，取 unique user list 內前 k 個最相似的 user
        most_similar_users = dict(
            sorted(most_similar_users.items(), key=lambda item: item[1], reverse=True)[:k])
        print(
            f"user '{userA}' top {k} similar users: {[u_id for u_id in most_similar_users.keys()]}")

        # 4. 開始預測
        numerator = []
        denominator = []
        for u in most_similar_users:
            u_bias = round(ds.get_user_mean_book_rating(u), 2)
            s_score = most_similar_users[u]
            # (rating - user bias) * similarity
            numerator.append((ds.get_rating(u, ISBN)-u_bias)*s_score)
            denominator.append(s_score)

        numerator = np.sum(numerator)
        denominator = np.sum(denominator)
        userA_bias = round(ds.get_user_mean_book_rating(userA), 2)
        if denominator == 0:
            predict_score = 0
        else:
            predict_score = round(userA_bias+(numerator/denominator))
        ds.save_predict_books_rating(userA, ISBN, predict_score)

        # recommend_items = ds.get_recommend_items_to_user(userA, most_similar_users)
        # print(f"recommend_items: {recommend_items}")