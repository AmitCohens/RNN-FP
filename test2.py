import json


def filterData():
    d = 0
    c = 0
    threshold = 20000
    new_data = {}

    with open("data.json", "r") as file:
        data = json.load(file)
        for item in data:
            if 'Drama' in data[item]['genre'] and d < threshold:
                new_data[item] = data[item]
                d += 1
            if 'Comedy' in data[item]['genre'] and c < threshold:
                new_data[item] = data[item]
                c += 1

    with open('small_data.json', 'w+') as f:
        json.dump(new_data,f,indent=4)
filterData()