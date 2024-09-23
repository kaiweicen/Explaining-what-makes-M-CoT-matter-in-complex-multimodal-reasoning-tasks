import re
import json

predicted_label = []
ground_truth_answers = []

label2class = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
data = json.load(open("/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_testset_v2.json"))
predicted_results = json.load(open("/Users/cenkaiwei/Documents/VLM_CoT/Results/202407031233_high_prob_relevance.json"))

predicted_result = []
correct_prediction = 0
pred_label_num = 0
for each_qa_pair in data:
    label = each_qa_pair["answer"]
    ground_truth_answers.append(label)


num_A = 0
num_B = 0
num_C = 0
num_D = 0
num_E = 0


for index, label in enumerate(ground_truth_answers):
    if label == 0:
        num_A += 1
    elif label == 1:
        num_B += 1
    elif label == 2:
        num_C += 1
    elif label == 3:
        num_D += 1
    elif label == 4:
        num_E += 1



for index, answer in enumerate(predicted_results):
    label = re.search(r'A|B|C|D|E', answer)
    if label != None:
        label = label.group()
        pred_label_num += 1
    else:
        label = 'B'
        print(answer)
    predicted_result.append(label)

print(pred_label_num)

def get_key(val):

    for key, value in label2class.items():
        if val == value:
            return key

    return "key doesn't exist"

for index, predicted_class in enumerate(predicted_result):
    label = get_key(predicted_class)
    predicted_label.append(label)

for index, predicted_label in enumerate(predicted_label):
  if predicted_label == ground_truth_answers[index]:
     correct_prediction +=1

acc = correct_prediction/len(ground_truth_answers) * 100
print("acc:")
print(acc)

