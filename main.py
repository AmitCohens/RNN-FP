import json
def getGenres():
    genres = []
    with open("IMDB_data_(89938 movies).json","r") as file:
        data = json.load(file)
        for item in data:
            if data[item]["genre"]:
                genres+= data[item]["genre"]
            # print(data[item]["genre"])
        genres = list(set(genres))
    return genres

def countMoviesByGenre(genres):
    genres_count = {genre : 0 for genre in genres}

    with open("IMDB_data_(89938 movies).json","r") as file:
        data = json.load(file)
        for item in data:
            if data[item]['genre']:
                for g in data[item]['genre']:
                    genres_count[g] += 1
    print(genres_count)
    return genres_count

def filterData():
    new_data = {}

    with open("IMDB_data_(89938 movies).json","r") as file:
        data = json.load(file)
        for item in data:
            if data[item]['genre']:
                if 'Drama' in data[item]['genre'] and 'Comedy' not in data[item]['genre']:
                    movie = {'name': data[item]['name'],
                             'description': data[item]['description'],
                             'genre': 'Drama'}
                    new_data[item] = movie

                elif 'Comedy' in data[item]['genre'] and 'Drama' not in data[item]['genre']:
                    movie = {'name': data[item]['name'],
                             'description': data[item]['description'],
                             'genre': 'Comedy'}
                    new_data[item] = movie

    with open('data_without_embedded.json', 'w+') as f:
        json.dump(new_data, f, indent=4)


# 'Comedy': 25007
# 'Drama': 46812

if __name__ == '__main__':
    # genres = getGenres()
    # countMoviesByGenre(genres)
    filterData()
