import random
import json
from random import sample

topic2label={}

testset_file = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_testset_v2.json"
testset = json.load(open(testset_file))
replacement_dataset_file = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_Testset_relevance_replacement.json"
replacement_dataset = json.load(open(replacement_dataset_file))

#for
#print(replacement_dataset["civics"])

#strip_solution = ["", " ", ' ', '']

for each_sqa in testset:
    #if each_sqa["topic"] == "civics":
        #print(each_sqa)
    if each_sqa["solution"] != "Null":
        topic = each_sqa["topic"]
        #print(topic)
        all_solutions = replacement_dataset[topic]
        #print(replacement_dataset[topic])
        #print(all_solutions)
        if len(all_solutions) == 0:
            print(topic)
        selected_solution = random.choice(all_solutions)
        #print(selected_solution)
        if selected_solution != each_sqa["solution"]:
            each_sqa["solution"] = selected_solution
#print(testset)

save_path = "./SQA_Testset_relevance_experiment_v2.json"
with open(save_path, 'w') as fp:
    json.dump(testset, fp)