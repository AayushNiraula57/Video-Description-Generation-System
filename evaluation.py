import json
from nltk.translate.bleu_score import sentence_bleu
from statistics import mean

with open('test_greedy.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
li = [x.strip() for x in content]
candidate_list = []
for idx,cap in enumerate(li):
    temp = cap.split(",")[1]
    candidate_list.append(temp.split())

print(candidate_list)

f = open('testing_public_label.json')
data = json.load(f)


reference_list = []
ref_list = []
for x in data:
    captions = x['caption'].split()
    ref_list.append(captions)
print("This is ref_list:")
print(ref_list)


reference = []
candidate = []
BLEU1=[]
BLEU2=[]
BLEU3=[]
BLEU4=[]
y=0
for data in ref_list:
    reference.append(data)
    candidate = candidate_list[y]
    BLEU1.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
    BLEU2.append(sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))

    
    y=y+1
    reference.clear()

BLEU_1 = mean(BLEU1)
BLEU_2 = mean(BLEU2)


print('Individual 1-gram: %f' % BLEU_1)
print('Individual 2-gram: %f' % BLEU_2)


