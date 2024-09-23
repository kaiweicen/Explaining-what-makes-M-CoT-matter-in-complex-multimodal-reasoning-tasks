import json
import jsonlines
from tqdm import tqdm
from pathlib import Path
from os.path import exists
import re

#select testset and questions with images

def get_sqa_test_data(infile,save_path):
    sqa_list = []
    sqa = json.load(open(infile))
    #print(sqa)
    for id, each_qa in sqa.items():
        #print(each_qa)
        question_id = 0
        if each_qa["split"] == "val" and each_qa["image"] != None:
            image_testset_root = "./ScienceQA/val/"
            path = image_testset_root + id
            #print(path)
            if exists(path):
                    image_num = {"image_id": id}
                    each_qa.update(image_num)
                    question = each_qa["question"].replace("\n"," ")
                    hint = each_qa["hint"]
                    processed_hint = hint.replace("\n"," ")
                    if each_qa["solution"] != "":
                        solution = each_qa["solution"]
                        #processed_solution = solution.replace("\n"," ")
                        #processed_solution = ''.join(solution.splitlines())
                        processed_solution = solution.replace("\n"," ")
                        #processed_solution = re.sub(r'\n', ' ', solution)
                        subject = each_qa["subject"]
                        topic = each_qa["topic"]
                        category = each_qa["category"]
                        skill = each_qa["skill"]
                        each_qa["hint"] = processed_hint
                        each_qa["question"] = question
                        each_qa["solution"] = processed_solution
                        each_qa["subject"] = subject
                        each_qa["topic"] = topic
                        each_qa["category"] = category
                        each_qa["skill"] = skill
                        #question_id += 1
                        #each_qa["question_id"] = question_id
                        sqa_list.append(each_qa)

    print("the length of sqa_val:")
    print(len(sqa_list))

    with open(save_path, 'w') as fp:
        json.dump(sqa_list, fp)
    #print(sqa_list)

if __name__ == '__main__':
    infile = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/problems.json"
    save_path = "./ScienceQA/SQA_Valset.json"
    get_sqa_test_data(infile, save_path)

solution = "Look at each object.\nFor each object, decide if it has that property.\nAn opaque object does not let light through. All three objects are opaque.\nA slippery object is hard to hold onto or stand on. The tortoise shell and the basketball are not slippery.\nA shiny object reflects a lot of light. You can usually see your reflection in a shiny object. The crown is shiny, but the basketball is not.\nThe property that all three objects have in common is opaque."
processed_solution = solution.replace("\n"," ")
print(processed_solution)