"""
    Program porównuje w bazie filmy dla użytkownika wejściowego
    na podstawie oceniania filmów uzytkownika wejściowego i użytkowników w bazie
    przy wykorzystaniu odległości euklidesowej

    Przykładowe wywołanie programu znajdując się w lokalizacji zawierającej main.py:
    python main.py --user 'Pawel Czapiewski'

    Jak przygotwac srodowisko do uruchomienia programu:
     - Zainstalowac numpy - pip install numpy (tu jeszcze dodac)

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)
"""

import argparse
import json
import operator

import numpy
import pip

pip.main(["install", "openpyxl"])
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user', dest='user', required=True,
                        help='User')
    # parser.add_argument('--user2', dest='user2', required=True,
    #                    help='Second user')
    # parser.add_argument("--score-type", dest="score_type", required=True,
    #                    choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    # if user2 not in dataset:
    #    raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users,
    # then the score is 0
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def sort_table(table, col=0, reverse=True):
    """
        program dla sortowania tablicy według kolumny,
        pomaga w sortowaniu względem odległości euklidesowej czy oceny filmu
    """
    return sorted(table, key=operator.itemgetter(col), reverse=reverse)

def sort_table2(table, col=0, col2=0):
    """
        program dla sortowania tablicy według kolumny,
        pomaga w sortowaniu względem odległości euklidesowej czy oceny filmu
    """
    return sorted(table, key=operator.itemgetter(col, col2), reverse=True)


def recommend_movies(dataset, user):
    """
        tu zawarta została cała logika polecania filmów,
        tworzy się wielowarstowa tabela, posortowana według odległości euklidesowych i ocen filmów

        następnym krokiem jest wybranie najbardziej podobnych osób,
        pozbycie sie tych samych filmow co ma uzytkownik wejsciowy
        i znalezienie filmów do polecenia i takich których uzytkownik wejściowy nie powinien oglądać
    """

    # i need five movies from people with close taste
    people_tuple_table = []

    input_user_movies = []

    for item in dataset:
        if user == item:
            for key in dataset[item]:
                input_user_movies.append([key, dataset[item][key]])

    # poszukac średniej wartości podobieństwa pomiedzy ludxmi, podzielić jakoś
    # i tak samo filmy pododawać do tabeli i wyszukac duze wartosci i małe

    for item in dataset:
        if user != item:
            movie_rating_table = []
            for key in dataset[item]:
                did_not_watch = True
                for movie_input_user in input_user_movies:

                    if movie_input_user[0] == key:
                        #print(movie_input_user[0], key, movie_input_user[0] == key)
                        did_not_watch = False

                if did_not_watch:
                    movie_rating_table.append([key, dataset[item][key]])
            people_tuple_table.append([item, euclidean_score(data, user, item), movie_rating_table])

    all_similar_people_movies = []
    highest_value = -1
    sorted_table = sort_table(people_tuple_table, 1)
    big_quantil = np.quantile(column(sorted_table, 1), .8)
    low_quantil = np.quantile(column(sorted_table, 1), .2)
    #print(big_quantil)

    all_movies_to_recommend = []
    all_movies_to_not_recommend = []

    #nie jest to zbyt wydajne ale podwóje sprawdzanie powinno zadziałać dla naszych danych

    for map_item in sorted_table:
        sorted_table_of_movies = sort_table(map_item[2], 1)
        if map_item[1] >= big_quantil:
            #print(map_item[0], map_item[1])

            good_score = np.quantile(column(sorted_table_of_movies, 1), .8)
            #jesli bad score nie ma żadnych niskich to szukaj ludzi oddalonych
            # 5-1 z wysokimi rankingami - tych filmow nie bedzie nasza soba lubiła
            bad_score = np.quantile(column(sorted_table_of_movies, 1), .2)
            #print(good_score)
            #print(bad_score)
            for movie in sorted_table_of_movies:
                #print(movie[0], movie[1])
                if movie[1] >= good_score:
                    #nazwa filmu, ocena, waga bliskosci osoby
                    all_movies_to_recommend.append([movie[0], movie[1], map_item[1]])
                if movie[1] <= bad_score and bad_score < 5:
                    #print(movie[0], movie[1], map_item[1], bad_score)
                    did_not_watch = True
                    for movie_input_user in input_user_movies:

                        if movie_input_user[0] == movie[0]:
                            # print(movie_input_user[0], key, movie_input_user[0] == key)
                            did_not_watch = False
                    if did_not_watch:
                        all_movies_to_not_recommend.append([movie[0], movie[1], map_item[1]])
            sorted_table.remove(map_item)

    big_quantil = np.quantile(column(sorted_table, 1), .8)

    if len(all_movies_to_recommend) < 5:
        for map_item in sorted_table:
            sorted_table_of_movies = sort_table(map_item[2], 1)
            if map_item[1] >= big_quantil:
                #print(map_item[0], map_item[1])

                good_score = np.quantile(column(sorted_table_of_movies, 1), .8)
                bad_score = np.quantile(column(sorted_table_of_movies, 1), .2)

                #print(good_score)
                #print(bad_score)
                for movie in sorted_table_of_movies:
                    #print(movie[0], movie[1])
                    if movie[1] >= good_score:
                        # nazwa filmu, ocena, waga bliskosci osoby
                        all_movies_to_recommend.append([movie[0], movie[1], map_item[1]])
                    if movie[1] <= bad_score:

                        all_movies_to_not_recommend.append([movie[0], movie[1], map_item[1]])
                sorted_table.remove(map_item)
    low_quantil = np.quantile(column(sorted_table, 1), .2)

    #tu coś mi nie działa
    sorted_table = sort_table(sorted_table, 1, False)
    if len(all_movies_to_not_recommend) < 5:
        for map_item in sorted_table:
            if map_item[1] <= low_quantil:
                sorted_table_of_movies = sort_table(map_item[2], 1)
                if map_item[1] <= low_quantil:
                    good_score = np.quantile(column(sorted_table_of_movies, 1), .2)

                    for movie in sorted_table_of_movies:
                        if movie[1] <= good_score:
                            #print(movie[0], movie[1], map_item[1], good_score)
                            did_not_watch = True
                            for movie_input_user in input_user_movies:

                                if movie_input_user[0] == movie[0]:
                                    # print(movie_input_user[0], key, movie_input_user[0] == key)
                                    did_not_watch = False
                            all_movies_to_not_recommend.append([movie[0], movie[1], map_item[1]])

    final_movie_reccomendation = []
    final_movie_non_reccomendation = []

    uniqueValues_reccomend = numpy.unique(all_movies_to_recommend)

    #print("rekomendujemy")
    for map_item in sort_table2(all_movies_to_recommend, 2, 1):
        #print(map_item[0], map_item[1], map_item[2])
        final_movie_reccomendation.append([map_item[0], map_item[1]])
        if len(final_movie_reccomendation) == 5:
            break

    all_movies_to_not_recommend.append(["Shutter Island", 5, 0])

    #print("nie rekomendujemy")
    for map_item in sort_table2(all_movies_to_not_recommend, 2, 1):
        #print(map_item[0])
        add_value = True

        for already_added_movie in final_movie_reccomendation:
            #print(already_added_movie[0], already_added_movie[1], map_item[0], map_item[1], already_added_movie[0] == map_item[0])
            if already_added_movie[0] == map_item[0]:
                 add_value = False
        if add_value:
            final_movie_non_reccomendation.append([map_item[0], map_item[1]])
            if len(final_movie_non_reccomendation) == 5:
                break

    print("\n")
    print("rekomendujemy podane filmy: ")
    for map_item in final_movie_reccomendation:
        print("Tytuł: " + str(map_item[0]) + " (ocena: " + str(map_item[1]) +")")
    print("\n")
    print("nie rekomendujemy podanych filmów: ")
    for map_item in final_movie_non_reccomendation:
        print("Tytuł: " + str(map_item[0]))

    print("\n")


def column(matrix, i):
    return [row[i] for row in matrix]

def my_function(x):
  return list(dict.fromkeys(x))

# Compute the Pearson correlation score between user1 and user2
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    # if user2 not in dataset:
    #    raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


if __name__ == '__main__':
    # letters = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż', 'Ą', 'Ć', 'Ę', 'Ł', 'Ń', 'Ó', 'Ś', 'Ź', 'Ż'];
    #
    # replacement = ['a', 'c', 'e', 'l', 'n', 'o', 's', 'z', 'z', 'A', 'C', 'E', 'L', 'N', 'O', 'S', 'Z', 'Z'];

    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        f_txt = f.read()
        # for i in range(len(letters)):
        #     f_txt = f_txt.replace(letters[i], replacement[i])
        # print(f_txt)
        data = json.loads(f_txt)

    print("\n")
    recommend_movies(data, user)
    # unrecommend_movies(data, user)
