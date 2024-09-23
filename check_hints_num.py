import json

def load_dataset(dataroot):

    """

    Construct the instructions
    Load the instructions and image_ids as entries

    """

    qa_dataset = json.load(open(dataroot))
    entries = []
    num_no_solu = 0

    for each_qa_pair in qa_dataset:
        qa_pair = {}
        question = each_qa_pair["question"]
        question_in_instruction = "Question:" + " " + "{" + question + "}"
        image_id = each_qa_pair["image_id"]
        choice = each_qa_pair["choices"]
        label = each_qa_pair["answer"]
        if each_qa_pair["solution"]:
            rationale = each_qa_pair["solution"]
        else:
            num_no_solu += 1

        if len(choice) == 2:
            option_a = choice[0]
            option_b = choice[1]
            choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + "}."  #到底要不要split""
        if len(choice) == 3:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + " " + "(c)" + " " + option_c + "}."
        if len(choice) == 4:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            option_d = choice[3]
            choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + " " + "(c)" + " " + option_c + " " + "(d)" + " " + option_d + "}."
        if len(choice) == 5:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            option_d = choice[3]
            option_e = choice[4]
            choice_in_instruction = "Options:" + " " + "{" + "(a)" + " " + option_a + " " + "(b)" + " " + option_b + " " + "(c)" + " " + option_c + " " + "(d)" + " " + option_d + " " + "(e)" + " " + option_e + "}."
        if each_qa_pair["hint"] != "":
            hint = each_qa_pair["hint"]
            hint_in_instruction = "Context:" + " " + "{" + hint + "}"
        answer_instruct = "Answer: (" #need to have "The answer is"?
        text_instruction = hint_in_instruction + " " + question_in_instruction + " " + choice_in_instruction + " " + answer_instruct
        qa_pair["text_instruction"] = text_instruction
        qa_pair["image_id"] = image_id
        qa_pair["answer"] = label
        qa_pair["choices_num"] = len(choice)
        entries.append(qa_pair)
        #print("entries")
        #print(entries)

    return entries, num_no_solu

entries, num_no_solu = load_dataset("./ScienceQA/SQA_Testset_5.json")
print(num_no_solu) #181