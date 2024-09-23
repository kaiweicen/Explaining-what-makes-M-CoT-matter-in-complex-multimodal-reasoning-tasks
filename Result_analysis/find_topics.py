import json

testset_file = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_Testset_7.json"
testset = json.load(open(testset_file))
topic_social_science = []
for each_sqa in testset:
    if each_sqa["subject"] == "social science":
        topic = each_sqa["topic"]
        if topic not in topic_social_science:
            topic_social_science.append(topic)

print(topic_social_science) #['us-history', 'geography', 'economics', 'world-history', 'civics']