import fasttext
import csv

threshold = 0.75
model_file = "/workspace/datasets/fasttext/title_model.bin"
top_words_file = "/workspace/datasets/fasttext/top_words.txt"
synonym_file = "/workspace/datasets/fasttext/synonyms.csv"

model = fasttext.load_model(model_file)
file = open(top_words_file, "r")

with open(synonym_file, "w", newline='') as synonymCsv:
    lineWriter = csv.writer(synonymCsv)

    for term in file.readlines():
        term = term.strip()
        nearest_neighbours = model.get_nearest_neighbors(term)

        row = [term]
        for i in nearest_neighbours:
            if i[0] >= threshold:
                row.append(i[1])
        lineWriter.writerow(row)
