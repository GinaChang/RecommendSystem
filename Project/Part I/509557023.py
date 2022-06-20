# from cProfile import label
# from json.tool import main
# import numpy as np
# from matplotlib import pyplot as plt
# import seaborn as sns
from copyreg import pickle
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
                # print(
                #     f'users: \n{users_df.loc[users_df["User-ID"]== 275081]}')
                # print(f'users: \n{users_df.loc[users_df["User-ID"]== 275082]}')
                # print(f'users: \n{users_df.loc[users_df["User-ID"]== 275083]}')
                # print(users_df.head(5))
                # users_df.to_pickle(os.path.join(
                #     self.dataset_dir, 'Users.pkl'))

                # with open(csv_path, 'r', encoding=self.params.get('encoding', None)) as f:
                #     reader = csv.reader(f, delimiter='\n')
                #     columns = []
                #     for i, line in enumerate(reader):
                #         # Clear line symbol of ""
                #         line = line[0].replace('"', '')

                #         #  Header Columns
                #         if i == 0:
                #             columns = line.split(';')
                #             continue

                #         # get each column value
                #         values = line.split(';')
                #         # check values length and skip this row after spilting format
                #         if len(values) < len(columns):
                #             continue

                #         user = {}
                #         # combine user data for each row
                #         for col_index, col in enumerate(columns):
                #             user[col] = values[col_index]
                #         users.append(user)

                #     # save preprocessing data to pandas dataframe and pickle file
                #     users_df = pd.DataFrame(users)
                #     users_df.to_pickle(os.path.join(
                #         self.dataset_dir, 'Users.pkl'))

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

                # print(books_df.loc[books_df['ISBN']=="0393045218"])
                # print(books_df.head(5).to_string())
                # books_df.to_pickle(os.path.join(
                #     self.dataset_dir, 'Books.pkl'))

                # books = []
                # with open(csv_path, 'r', encoding=self.params.get('encoding', None)) as f:
                #     reader = csv.reader(f, delimiter='\n')
                #     columns = []
                #     for i, line in enumerate(reader):
                #         # Clear line symbol of ""
                #         line = line[0].replace('"', '')

                #         #  Header Columns
                #         if i == 0:
                #             columns = line.split(';')
                #             continue

                #         # get each column value
                #         values = line.split(';')
                #         # check values length and skip this row after spilting format
                #         if len(values) < len(columns):
                #             continue

                #         book = {}
                #         # combine book data for each row
                #         for col_index, col in enumerate(columns):
                #             book[col] = values[col_index]
                #         books.append(book)

                #     # save preprocessing data to pandas dataframe and pickle file
                #     books_df = pd.DataFrame(books)
                #     books_df.to_pickle(os.path.join(
                #         self.dataset_dir, 'Books.pkl'))

        except Exception as ex:
            print(ex)

        return books_df

    def transform_ISBN_format(self, ISBN):
        # print(ISBN)
        ISBN = str(ISBN)
        if ISBN.startswith('ISBN'):
            # print(f"prefix is ISBN: {ISBN}")
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
            else:
                csv_path = os.path.join(self.dataset_dir, 'Ratings.csv')
                ratings_df = pd.read_csv(
                    csv_path, sep=";", encoding=self.params.get('encoding', None))

                ratings_df["ISBN"] = ratings_df["ISBN"].apply(
                    lambda x: x.strip('"').strip("\\"))

                ratings_df["ISBN"] = ratings_df["ISBN"].apply(
                    self.transform_ISBN_format)
                # print(ratings_df.loc[ratings_df['User-ID']==135809])

                ratings_df = ratings_df[ratings_df['ISBN'].isin(
                    self.books_df['ISBN'])]
                # print(ratings_df.head(5))
                # ratings_df.to_pickle(os.path.join(
                #     self.dataset_dir, 'Ratings.pkl'))

                # ratings = []
                # with open(csv_path, 'r', encoding=self.params.get('encoding', None)) as f:
                #     reader = csv.reader(f, delimiter='\n')
                #     columns = []
                #     for i, line in enumerate(reader):
                #         # Clear line symbol of ""
                #         line = line[0].replace('"', '')

                #         #  Header Columns
                #         if i == 0:
                #             columns = line.split(';')
                #             continue

                #         # get each column value
                #         values = line.split(';')
                #         # check values length and skip this row after spilting format
                #         if len(values) < len(columns):
                #             continue

                #         rating = {}
                #         # combine rating data for each row
                #         for col_index, col in enumerate(columns):
                #             rating[col] = values[col_index]
                #         ratings.append(rating)

                #     # save preprocessing data to pandas dataframe and pickle file
                #     ratings_df = pd.DataFrame(ratings)
                #     # remove the row that ISBN is not numeric
                #     ratings_df = ratings_df[ratings_df['ISBN'].apply(
                #         lambda x: x.isnumeric())]
                #     ratings_df.to_pickle(os.path.join(
                #         self.dataset_dir, 'Ratings.pkl'))

        except Exception as ex:
            print(ex)

        return ratings_df


params = {
    'encoding': 'ISO-8859-1',  # utf-8 might get error
    # 'on_bad_lines': 'skip'
}
dataset = Dataset(params)

# Q3
user_ratings_df = dataset.ratings_df.groupby(['User-ID']).size().reset_index(
    name='Number of interactions')
percentiles_info = user_ratings_df.describe(
    percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
print(f"Q3: \n{percentiles_info}")
ticks = percentiles_info.index[4:-1].tolist()
number_of_interactions = percentiles_info.reset_index(
    drop=True)['Number of interactions'][4:-1].to_list()
number_of_interactions.reverse()

plt.figure(figsize=(15, 10))
x = np.arange(len(ticks))
plt.bar(x, number_of_interactions)
plt.xticks(x, ticks, fontsize=14)
plt.xlabel('percentage of users', fontsize=16)
plt.ylabel('number of interactions', fontsize=16)
plt.show()

# Q4
item_ratings_df = dataset.ratings_df.groupby(['ISBN']).size().reset_index(
    name='Number of interactions')
percentiles_info = item_ratings_df.describe(
    percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
print(f"Q4: \n{percentiles_info}")


ticks = percentiles_info.index[4:-1].tolist()
times_of_items = percentiles_info.reset_index(
    drop=True)['Number of interactions'][4:-1].to_list()
times_of_items.reverse()

plt.figure(figsize=(15, 10))
x = np.arange(len(ticks))
plt.bar(x, times_of_items)
plt.xticks(x, ticks, fontsize=14)
plt.xlabel('percentage of items', fontsize=16)
plt.ylabel('times of items', fontsize=16)
plt.show()


# Q6
book_groupby_ISBN = dataset.ratings_df.groupby(['ISBN'])
avg_book_ratings = book_groupby_ISBN['Book-Rating'].mean()
avg_book_ratings.name = 'Avg-Book-Rating'
num_book_ratings = book_groupby_ISBN['Book-Rating'].count()
num_book_ratings.name = 'Number-of-Book-Rating'
books = dataset.books_df.join(num_book_ratings, on="ISBN")
books = books.join(avg_book_ratings, on="ISBN")
num_book_ratings_mean = num_book_ratings.mean()

results_1 = books.sort_values(
    by=['Avg-Book-Rating'], ascending=False).head(5)
print(f"Just sort Avg-Book-Rating asc:\n {results_1[['ISBN', 'Book-Title', 'Avg-Book-Rating', 'Number-of-Book-Rating']].to_string(index=False)}")

print("\n\n")

results_2 = books.loc[books['Number-of-Book-Rating'] > num_book_ratings_mean].sort_values(
    by=['Avg-Book-Rating', 'Number-of-Book-Rating'], ascending=False).head(5)
# print(books.head(5))
print(f"Sort Avg-Book-Rating and Number-of-Book-Rating asc:\n {results_2[['ISBN', 'Book-Title', 'Avg-Book-Rating', 'Number-of-Book-Rating']].to_string(index=False)}")
