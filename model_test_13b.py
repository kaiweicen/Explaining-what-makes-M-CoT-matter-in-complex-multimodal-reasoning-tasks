from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from utils import Load_Dataset_Eval
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
#from genre.trie import MarisaTrie
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import marisa_trie as MarisaTrie
from datetime import datetime

args = {
"from_pretrained" : "Salesforce/instructblip-vicuna-13b",
"num_subiters" :2,
"zero_shot" : True,
"batch_size" : 30,
"drop_last" : False,
"seed" : 42,
"local_rank" : -1,
"num_workers" : 2,
"num_val_workers" : 2,
"in_memory" : False,
"use_chunk" : 0,
}

def load_dataset(dataroot):

    """

    Construct the instructions
    Load the instructions and image_ids as entries

    """

    qa_dataset = json.load(open(dataroot))
    entries = []

    for each_qa_pair in qa_dataset:
        qa_pair = {}
        question = each_qa_pair["question"]
        question_in_instruction = "Question:" + " " + question
        image_id = each_qa_pair["image_id"]
        choice = each_qa_pair["choices"]
        if len(choice) == 2:
            option_a = choice[0]
            option_b = choice[1]
            choice_in_instruction = "Options:" + "(A)" + option_a + "(B)" + option_b   #到底要不要split""
        if len(choice) == 3:
            option_a = choice[0]
            option_b = choice[1]
            option_c = choice[2]
            choice_in_instruction = "Options:" + "(A)" + option_a + "(B)" + option_b + "(C)" + option_c #the case of choices > 3 ?
        if each_qa_pair["hint"] != "":
            hint = each_qa_pair["hint"]
            hint_in_instruction = "Context:" + hint
        answer_instruct = "Answer: The answer is" #need to have "The answer is"?
        text_instruction = question_in_instruction + " " + choice_in_instruction + " " + hint_in_instruction + " " + answer_instruct
        qa_pair["text_instruction"] = text_instruction
        qa_pair["image_id"] = image_id
        entries.append(qa_pair)

    return entries

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
entries = load_dataset("./SQA_Testset_3.json")
image_path = "./ScienceQA/test/"
model = InstructBlipForConditionalGeneration.from_pretrained(args["from_pretrained"])
model.to(device)
model.eval()
start_timestamp = datetime.now()
start_formatted_timestamp = start_timestamp.strftime('%Y-%m-%d %H:%M:%S')
for index, entry in enumerate(entries):
    entry = entries[index]
    image = Image.open(image_path+entry["image_id"]+"/"+"image.png").convert("RGB")
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.ConvertImageDtype(torch.float32)
                                ])
    image_to_tensor = trans(image)
    textual_instruction = entry["text_instruction"]
    encoding = processor(image_to_tensor, textual_instruction, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)
    pixel_size = encoding["pixel_values"].size()
    #print("pixel_size")
    #print(pixel_size)
    #for k, v in encoding.items():
        #encoding[k] = v.squeeze()
    #prompt_length = len(entry["text_instruction"])
    #outputs = model.generate(**encoding, pad_token_id = tokenizer.eos_token_id, max_length = 20, prefix_allowed_tokens_fn = lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))
    outputs = model.generate(**encoding,  max_new_tokens=1024)
    generated_text = processor.batch_decode(outputs[0], skip_special_tokens=True)
    #print(generated_text)
    with open("./ScienceQA/zero_shot_generation_text_13b.json", 'w') as fp:
        json.dump(generated_text, fp)

end_timestamp = datetime.now()
end_formatted_timestamp = start_timestamp.strftime('%Y-%m-%d %H:%M:%S')
eval_time = end_timestamp - start_timestamp
print("eval_time:")
print(eval_time)