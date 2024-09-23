#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
import numpy as np
import torch
from PIL import Image
import requests
import json
from datetime import datetime
from torchvision import transforms

def load_dataset(dataroot):

    """

    Construct the instructions
    Load the instructions and image_ids as entries

    """

    qa_dataset = json.load(open(dataroot))
    entries = []
    strip_solution = ["", " ", ' ', '']

    for each_qa_pair in qa_dataset:
        qa_pair = {}
        question = each_qa_pair["question"]
        #question_in_instruction = "Question:" + " " + "{" + question + "}"
        question_in_instruction = "Question:" + " " + question
        image_id = each_qa_pair["image_id"]
        choice = each_qa_pair["choices"]
        label = each_qa_pair["answer"]

        if len(choice) == 2:
            option_a = choice[0]
            option_b = choice[1]
            #choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + "}."  #到底要不要split""
            choice_in_instruction = "Options:" + " " + "(A)" + " " + option_a + " " + "(B)" + " " + option_b
        if len(choice) == 3:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            #choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + " " + "(c)" + " " + option_c + "}."
            choice_in_instruction = "Options:" + " " + "(A)" + " " + option_a + " " + "(B)" + " " + option_b + " " + "(C)" + " " + option_c
        if len(choice) == 4:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            option_d = choice[3]
            #choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + " " + "(c)" + " " + option_c + " " + "(d)" + " " + option_d + "}."
            choice_in_instruction = "Options:" + " " + "(A)" + " " + option_a + " " + "(B)" + " " + option_b + " " + "(C)" + " " + option_c + " " + "(D)" + " " + option_d
        if len(choice) == 5:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            option_d = choice[3]
            option_e = choice[4]
            #choice_in_instruction = "Options:" + "{" + " " + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + " " + "(c)" + " " + option_c + " " + "(d)" + " " + option_d + " " + "(e)" + " " + option_e + "}."
            choice_in_instruction = "Options:" + " " + "(A)" + " " + option_a + " " + "(B)" + " " + option_b + " " + "(C)" + " " + option_c + " " + "(D)" + " " + option_d + " " + "(E)" + " " + option_e
        if each_qa_pair["hint"] != "":
            hint = each_qa_pair["hint"]
            #hint_in_instruction = "Context:" + " " + "{" + hint + "}"
            hint_in_instruction = "Context:" + " " + hint + " "
        else:
            hint_in_instruction = ""
        if each_qa_pair["solution"] != "Null":
            solution = "Solution: Let's think step by step." + " " + each_qa_pair["solution"] + " "
        else:
            solution = ""
        answer_instruct = "Answer: The answer is: (" #need to have "The answer is"?
        #answer_instruct = "Answer: ("
        #text_instruction = question_in_instruction + " " + hint_in_instruction + " " + choice_in_instruction + " " + solution + " " + answer_instruct
        text_instruction = question_in_instruction + " " + hint_in_instruction + choice_in_instruction + " " + solution + answer_instruct #cot
        #text_instruction = question_in_instruction + " " + hint_in_instruction + choice_in_instruction + " " + answer_instruct
        qa_pair["text_instruction"] = text_instruction
        qa_pair["image_id"] = image_id
        qa_pair["answer"] = label
        qa_pair["choices_num"] = len(choice)
        entries.append(qa_pair)
        #print("entries")
        #print(entries)

    return entries

entries = load_dataset("./SQA_testset_v2.json")
image_path = "./data/test/" #lisbon
#image_path = "./test/"

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

access_token = "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"

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", token=access_token)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")

model.to(device)
model.eval()

start_timestamp = datetime.now()
start_formatted_timestamp = start_timestamp.strftime('%Y-%m-%d %H:%M:%S')

predict_result = []

high_prob_log_info = []
all_log_info = []
for index, entry in enumerate(entries):
    entry = entries[index]
    num_of_choice = entry["choices_num"]
    image = Image.open(image_path+entry["image_id"]+"/"+"image.png").convert("RGB")
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.ConvertImageDtype(torch.float32)
                                ])
    image_to_tensor = trans(image)
    textual_instruction = entry["text_instruction"]
    encoding = processor(image_to_tensor, textual_instruction, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)

    #text_encoding = processor.tokenizer(text=textual_instruction, padding=True, max_length=256, return_tensors="pt")


    #index_list = []
    #vocab = processor.tokenizer.get_vocab()
    #index_A = list(vocab).index('A')
    #index_list.append(index_A)
    #print("index_A")
    #print(index_A)
    #index_B = list(vocab).index('B')
    #index_list.append(index_B)
    #print("index_B")
    #print(index_B)
    #index_C = list(vocab).index('C')
    #index_list.append(index_C)
    #print("index_C")
    #print(index_C)
    #index_D = list(vocab).index('D')
    #index_list.append(index_D)
    #print("index_D")
    #print(index_D)
    #qformer_text_encoding = processor.qformer_tokenizer(text=textual_instruction, padding=True, max_length=256, return_tensors="pt")
    #image_encoding = processor.image_processor(images=image_to_tensor, return_tensors="pt")

    #pixel_values = image_encoding["pixel_values"].to(device)
    #qformer_input_ids = qformer_text_encoding.pop("input_ids").to(device)
    #qformer_attention_mask = qformer_text_encoding.pop("attention_mask").to(device)
    #input_ids = text_encoding["input_ids"].to(device)
    #attention_mask = text_encoding["attention_mask"].to(device)
    #input_ids = encoding["input_ids"]
    #length_of_prompt = encoding["input_ids"].size()[1]

    #outputs = model.generate(pixel_values, qformer_input_ids, qformer_attention_mask, input_ids, attention_mask, max_new_tokens=30, do_sample=False).scores
    #output_sequences = model.generate(**encoding, max_new_tokens=256, return_dict_in_generate=True, output_scores=True).sequences
    output_sequences = model.generate(**encoding, max_length=912, return_dict_in_generate=True, output_scores=True).sequences
    #print("index")
    #print(index)
    #print("outputs")
    #print(output_sequences)
    #log_info["output_sequence_ids"] = output_sequences

    generated_text = processor.batch_decode(output_sequences, skip_special_tokens=True)[0].strip()
    #print("generated_text")
    #print(generated_text)
    #if "(" in generated_text:
        #index_of_bracket = generated_text.rfind("(")
        #index_of_label = index_of_bracket
    #else:

    #log_info["generated_text"] = generated_text
    #print("generated_text")
    #print(generated_text)
    #if generated_text[1]:
        #log_info["second generated_text"] = generated_text[1]
    #else:
        #print("generated_text[1] not exists, generated_text:")
        #print(generated_text)

    #outputs_scores = model.generate(**encoding, max_new_tokens=256, do_sample=False, return_dict_in_generate=True, output_scores=True).scores
    outputs_scores = model.generate(**encoding, max_length=912, do_sample=False, return_dict_in_generate=True, output_scores=True).scores
    #print("outputs_scores")
    #print(outputs_scores)
    #log_info["outputs_scores"] = outputs_scores
    #print("size of scores")
    #print(outputs_scores.size())
    #outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)

    #generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    #print(generated_text)

    vocab_size = 32001
    class2position = {"A":0,"B":1,"C":2,"D":3,"E":4}
    #class2position = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    #scores = outputs.scores
    #print("logits")
    #print(scores)
    scores = torch.stack(outputs_scores,0)
    #print("scores")
    #print(scores)

    #print("size of scores")
    #print(scores.size())

    #scores_reshaped = scores.reshape(-1,32001)
    #print("size of scores #sq * 32001")
    #print(scores_reshaped.size())

    score_generated_first_token = scores[0] #length of input_ids + 1
    #print("score_generated_first_token")
    #print(score_generated_first_token)
    #log_info["score_generated_first_token"] = score_generated_first_token

    #scores_reshaped = score_generated_first_token.reshape(-1,32001)
    #print("size of score_generated_first_token")
    #print(score_generated_first_token.size())

    A_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("A")].item()
    #A_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("a")].item()
    #print("A_score")
    #print(A_score)
    B_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("B")].item()
    #B_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("b")].item()
    #print("B_score")
    #print(B_score)
    #print("token b")
    #print(processor.tokenizer.convert_tokens_to_ids("b"))
    C_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("C")].item()
    #C_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("c")].item()
    #print("C_score")
    #print(C_score)
    #print("token c")
    #print(processor.tokenizer.convert_tokens_to_ids("c"))
    #print("token (")
    #print(processor.tokenizer.convert_tokens_to_ids("("))
    D_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("D")].item()
    #D_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("d")].item()
    #print("D_score")
    #print(D_score)
    E_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("E")].item()
    #E_score = score_generated_first_token[0, processor.tokenizer.convert_tokens_to_ids("e")].item()
    class_list = []

    #case two choices:
    if num_of_choice == 2:
        class_list.append(A_score)
        class_list.append(B_score)
    #case three choices:
    elif num_of_choice == 3:
        class_list.append(A_score)
        class_list.append(B_score)
        class_list.append(C_score)
    #case four choices:
    elif num_of_choice == 4:
        class_list.append(A_score)
        class_list.append(B_score)
        class_list.append(C_score)
        class_list.append(D_score)
    elif num_of_choice == 5:
        class_list.append(A_score)
        class_list.append(B_score)
        class_list.append(C_score)
        class_list.append(D_score)
        class_list.append(E_score)
    else:
        print("there is case that has choices more than 5 or less than two")
        print(index)
        print(num_of_choice)

    classes_scores = torch.tensor([class_list])
    #print("classes_scores")
    #print(classes_scores)
    #log_info["classes_scores"] = classes_scores.cpu().numpy()

    softmax_score = torch.nn.functional.softmax(classes_scores,dim=1)
    #print("softmax_score")
    #print(softmax_score)
    #log_info["softmax_score"] = softmax_score.cpu().numpy()

    max_score_position = int(torch.argmax(softmax_score))
    #print("max_score_positio")
    class_predicted = [k for k, v in class2position.items() if v == max_score_position][0]
    high_prob_log_info.append(class_predicted)
    #print("class_predicted")
    #print(class_predicted)
    #log_info["class_predicted"] = class_predicted

    # sanity check
    softmax_score_over_all_vocab = torch.nn.functional.softmax(score_generated_first_token, dim=-1)
    max_score_position_over_all_vocab = torch.argmax(softmax_score_over_all_vocab)
    #print("max_score_position_over_all_vocab")
    #print(max_score_position_over_all_vocab)
    max_score_word = processor.tokenizer.convert_ids_to_tokens(int(max_score_position_over_all_vocab))
    #print("max_score_word")
    #print(max_score_word)
    #if max_score_word == class_predicted == generated_text[1]:
    if max_score_word == class_predicted:
        pass
    else:
        print("sanity check is not passed!")
        print(index)
        print("max_score_word")
        print(max_score_word)
        print("class_predicted")
        print(class_predicted)
        #print("generated_text[1]")
        #print(generated_text[1])
        print("generated_text")
        print(generated_text)

    #print(list(vocab)[max_score_position_over_all_vocab])

    #outputs_generate_func = model.generate(pixel_values, qformer_input_ids, qformer_attention_mask, input_ids, attention_mask, num_beams=1, max_length=100, do_sample=False)
    #generated_text_generate_func = processor.batch_decode(outputs_generate_func, skip_special_tokens=True)[0].strip()
    #print("generated_text_generate_func")
    #print(generated_text_generate_func)

    predict_result.append(class_predicted)
    all_log_info.append(generated_text)
    #print("predict_result")
    #print(predict_result)

#calculate acc
correct_prediction = 0
label2class = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}
#label2class = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e'}
ground_truth_answer = []
predicted_label = []
#class2label in the prediction list

# function to return key for any value
def get_key(val):

    for key, value in label2class.items():
        if val == value:
            return key

    return "key doesn't exist"

for index, predicted_class in enumerate(predict_result):
    #predicted_class = i for i in label2class if label2class[i] == predicted_class
    label = get_key(predicted_class)
    predicted_label.append(label)

#print("predicted_label")
#print(predicted_label)

for index, qa_pair in enumerate(entries):
    ground_truth_answer.append(qa_pair["answer"])

#print("ground_truth_answer")
#print(ground_truth_answer)

for index, predicted_label in enumerate(predicted_label):
  if predicted_label == ground_truth_answer[index]:
     correct_prediction +=1
     #print("correct_prediction")
     #print(correct_prediction)

acc = correct_prediction/len(entries) * 100
print("acc:")
print(acc)
#all_log_info["acc"] = acc

end_timestamp = datetime.now()
end_formatted_timestamp = start_timestamp.strftime('%Y-%m-%d %H:%M:%S')
eval_time = end_timestamp - start_timestamp
print("eval_time:")
print(eval_time)

#all_log_info["eval_time"] = eval_time

save_path = "./202406201612_gen_text_cot.json"
with open(save_path, 'w') as fp:
    json.dump(all_log_info, fp)

save_path = "./202406201612_high_prob_cot.json"
with open(save_path, 'w') as fp:
    json.dump(high_prob_log_info, fp)

print("eval ends!")