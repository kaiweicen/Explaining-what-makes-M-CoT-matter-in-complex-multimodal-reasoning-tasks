import json
import random

zero_shot_file = "/Users/cenkaiwei/Documents/VLM_CoT/Results/202406201547_high_prob.json"
cot_file = "/Users/cenkaiwei/Documents/VLM_CoT/Results/202406201612_high_prob_cot.json"
ground_truth_file = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_testset_v2.json"
zero_shot_results = json.load(open(zero_shot_file))

cot_results = json.load(open(cot_file))

ground_truth = json.load(open(ground_truth_file))
print(len(ground_truth))
ground_truth_list = []
ground_truth_list_converted = []
useful_cot_index_list = []
zero_shot_results_converted = []
cot_results_converted = []

for each_qa_pair in ground_truth:
    label = each_qa_pair["answer"]
    ground_truth_list.append(label)

label2class = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
def get_key(val):

    for key, value in label2class.items():
        if val == value:
            return key

    return "key doesn't exist"

for index, each_class in enumerate(zero_shot_results):
    label = get_key(each_class)
    zero_shot_results_converted.append(label)
    #print(zero_shot_results_converted)

for index, each_class in enumerate(cot_results):
    label = get_key(each_class)
    cot_results_converted.append(label)
    #print(cot_results_converted)

for index, label in enumerate(ground_truth_list):
    zero_shot_result = zero_shot_results_converted[index]
    cot_result = cot_results_converted[index]
    if zero_shot_result != label and cot_result == label:
        useful_cot_index_list.append(index)

#print(len(useful_cot_index_list))
useful_cot_list = []
for index in useful_cot_index_list:
    qa_pair = ground_truth[index]
    useful_cot_list.append(qa_pair)

#print(useful_cot_list)
print(len(useful_cot_list))

#save_path = "./useful_cot_13b.json"
#with open(save_path, 'w') as fp:
    #json.dump(useful_cot_list, fp)

sample_36_examples = []
sampled_examples = random.choices(useful_cot_list, k=40)
print(sampled_examples)

save_path = "./sampled_examples_v4.json"
with open(save_path, 'w') as fp:
    json.dump(sampled_examples, fp)


