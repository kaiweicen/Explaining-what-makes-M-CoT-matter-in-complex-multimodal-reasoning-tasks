import json

zero_shot_file = "/Users/cenkaiwei/Documents/VLM_CoT/Results/202406201547_high_prob.json"
relevance_file = "/Users/cenkaiwei/Documents/VLM_CoT/Results/202406231511_high_prob_relevance.json"
zero_shot_results = json.load(open(zero_shot_file))
relev_results = json.load(open(relevance_file))
same_prediction = []

for index, label in enumerate(zero_shot_results):
    zero_shot_result = zero_shot_results[index]
    cot_result = relev_results[index]
    if zero_shot_result == cot_result:
        same_prediction.append(index)

print(len(same_prediction))

