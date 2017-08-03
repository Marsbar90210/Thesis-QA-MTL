import os
from data import read_challenge

qas_to_challenges = {
    'qa1': 'qa1_single-supporting-fact_{}.txt',
    'qa2': 'qa2_two-supporting-facts_{}.txt',
    'qa3': 'qa3_three-supporting-facts_{}.txt',
    'qa4': 'qa4_two-arg-relations_{}.txt',
    'qa5': 'qa5_three-arg-relations_{}.txt',
    'qa6': 'qa6_yes-no-questions_{}.txt',
    'qa7': 'qa7_counting_{}.txt',
    'qa8': 'qa8_lists-sets_{}.txt',
    'qa9': 'qa9_simple-negation_{}.txt',
    'qa10': 'qa10_indefinite-knowledge_{}.txt',
    'qa11': 'qa11_basic-coreference_{}.txt',
    'qa12': 'qa12_conjunction_{}.txt',
    'qa13': 'qa13_compound-coreference_{}.txt',
    'qa14': 'qa14_time-reasoning_{}.txt',
    'qa15': 'qa15_basic-deduction_{}.txt',
    'qa16': 'qa16_basic-induction_{}.txt',
    'qa17': 'qa17_positional-reasoning_{}.txt',
    'qa18': 'qa18_size-reasoning_{}.txt',
    'qa19': 'qa19_path-finding_{}.txt',
    'qa20': 'qa20_agents-motivations_{}.txt'
}

FILE_DIR = './../data/'
widxfilename = FILE_DIR + "babiword_idx_{}.txt"

def getdata(ver):
    ver = ver.replace('/', '_')
    fname = widxfilename.format(ver)
    if os.path.exists(fname):
        with open(fname, "r") as file:
            fdata = file.read()
            word_idx = eval(fdata)
            return word_idx
    else:
        return None

data = getdata('en');
print('en : {}'.format(len(data)));
data = getdata('hn');
print('hi : {}'.format(len(data)));

for i in range(1, 21):
    challenge = 'tasks_1-20_v1-2/{}/{}'.format('en', qas_to_challenges['qa{}'.format(i)])
    train, test = read_challenge(challenge, flatten=0, E2E=1)
    count = 0;
    sum = 0;
    for some in train + test:
        sum += len(some[0])
        count += 1
    print(sum/count)