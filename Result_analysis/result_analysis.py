import json

testset = json.load(open("/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_testset_v2.json"))
zero_shot_answerset = json.load(open("/Users/cenkaiwei/Documents/VLM_CoT/Results/202406061552_high_prob.json"))
nl_example_index = []
zero_shot_answer_list = []
ground_truth_answer_list = []
zero_shot_answer_label = []
low_grade = ["grade1","grade2","grade3","grade4","grade5","grade6"]
high_grade = ["grade7","grade8","grade9","grade10","grade11","grade12"]
for index, each_qa in enumerate(testset):
    #if each_qa["grade"] in low_grade:
     if each_qa["topic"] == "us-history":
        nl_example_index.append(index)
        ground_truth_answer = each_qa["answer"]
        ground_truth_answer_list.append(ground_truth_answer)

for index in nl_example_index:
    answer = zero_shot_answerset[index]
    zero_shot_answer_list.append(answer[0])

print("zero_shot_answer_list")
print(zero_shot_answer_list)
print("len nl examples")
print(len(nl_example_index))
print("len ground_truth_answer")
print(len(ground_truth_answer_list))
print("len zero_shot_answer_list")
print(len(zero_shot_answer_list))

#compute acc
correct_prediction = 0
#classtolabel = {"A":0, "B":1, "C":2, "D":3}
#label2class = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e'}
label2class = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}

# function to return key for any value
def get_key(val):

    for key, value in label2class.items():
        if val == value:
            return key

    return "key doesn't exist"

for index, predicted_class in enumerate(zero_shot_answer_list):
    #predicted_class = i for i in label2class if label2class[i] == predicted_class
    label = get_key(predicted_class)
    zero_shot_answer_label.append(label)
print(zero_shot_answer_label)

for index, predicted_label in enumerate(zero_shot_answer_label):
  if predicted_label == ground_truth_answer_list[index]:
     correct_prediction +=1
     #print("correct_prediction")
     #print(correct_prediction)

acc = correct_prediction/len(nl_example_index) * 100
print("acc:")
print(acc)








