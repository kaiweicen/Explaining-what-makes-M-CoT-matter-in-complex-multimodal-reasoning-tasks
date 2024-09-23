import json

infile = "/Users/cenkaiwei/Documents/VLM_CoT/ScienceQA/SQA_Testset_3.json"
natural_science = 0
social_science = 0
language_science = 0
biology = 0
physics = 0
geography = 0
chemistry = 0
engineering = 0
history = 0
earth_science = 0
economics = 0
writing_strategies = 0
reading_comprehension = 0
world_history = 0
vocabulary = 0
sqa = json.load(open(infile))
for each_sqa in sqa:
    if each_sqa["subject"] == "natural science":
        natural_science += 1
    if each_sqa["subject"] == "social science":
        social_science += 1
    if each_sqa["subject"] == "language science":
        language_science += 1
    if each_sqa["topic"] == "biology":
        biology += 1
    if each_sqa["topic"] == "physics":
        physics += 1
    if each_sqa["topic"] == "geography":
        geography += 1
    if each_sqa["topic"] == "chemistry":
        chemistry += 1
    if each_sqa["topic"] == "science-and-engineering-practices":
        engineering += 1
    if each_sqa["topic"] == "us-history":
        history += 1
    if each_sqa["topic"] == "earth-science":
        earth_science += 1
    if each_sqa["topic"] == "economics":
        economics += 1
    if each_sqa["topic"] == "writing-strategies":
        writing_strategies += 1
    if each_sqa["topic"] == "reading-comprehension":
        reading_comprehension += 1
    if each_sqa["topic"] == "world-history":
        world_history += 1

print(natural_science) #1209
print(social_science)  #764
print(language_science) #44
print(biology) #403
print(physics) #425
print(geography) #600
print(chemistry) #94
print(engineering) #128
print(history) #84
print(earth_science) #156
print(economics) #71
print(writing_strategies) #23
print(reading_comprehension) #12
print(world_history) #7


