import json
import jsonlines
from tqdm import tqdm
from pathlib import Path
from os.path import exists
import re

#select testset and questions with images

def get_sqa_test_data(infile,save_path):
    sqa_list = []
    strip_solution = ["", " ", ' ', '']
    sqa = json.load(open(infile))
    #print(sqa)
    for id, each_qa in sqa.items():
        #print(each_qa)
        question_id = 0
        if each_qa["split"] == "test" and each_qa["image"] != None:
            image_testset_root = "./ScienceQA/test/"
            path = image_testset_root + id
            #print(path)
            if exists(path):
                    image_num = {"image_id": id}
                    each_qa.update(image_num)
                    question = each_qa["question"].replace("\n"," ")
                    hint = each_qa["hint"]
                    processed_hint = hint.replace("\n"," ")
                    choices = each_qa["choices"]
                    for choice in choices:
                        if "\n" in choice:
                            print("\n!")
                    if each_qa["solution"] not in strip_solution:
                        solution = each_qa["solution"]
                        processed_solution = solution.replace("\n", " ")
                    else:
                        processed_solution = "Null"
                    each_qa["hint"] = processed_hint
                    each_qa["question"] = question
                    each_qa["solution"] = processed_solution

                    sqa_list.append(each_qa)
                    #prompt_length = len(question) + len(processed_hint) + len(processed_solution) + len()

    print("the length of sqa_test:")
    print(len(sqa_list))

    with open(save_path, 'w') as fp:
        json.dump(sqa_list, fp)


if __name__ == '__main__':
    infile = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/problems.json"
    save_path = "./ScienceQA/SQA_testset_v2.json"
    get_sqa_test_data(infile, save_path)

