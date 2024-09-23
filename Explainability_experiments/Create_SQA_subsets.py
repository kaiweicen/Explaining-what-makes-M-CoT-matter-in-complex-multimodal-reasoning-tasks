import json
test_infile = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_Testset_7.json"
data = {"earth-science": [], "economics": [], "chemistry": [], "writing-strategies": [], "world-history": [], "civics": [], "reading-comprehension": [], "word-study": [], "literacy-in-science": [], "vocabulary": [], "science-and-engineering-practices": [], "us-history": [], "biology": [], "physics": [], "geography": []}
sqa = json.load(open(test_infile))
strip_solution = ["", " ", ' ', '']
for each_sqa in sqa:
    if each_sqa["solution"] not in strip_solution:
        topic = each_sqa["topic"]
        solution = each_sqa["solution"]
        data[topic].append(solution)

save_path = "/ScienceQA/SQA_Testset_relevance_replacement.json"
with open(save_path, 'w') as fp:
    json.dump(data, fp)


