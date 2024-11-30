import sys
import os
import json


in_file = sys.argv[1]
save_path, file_name = os.path.split(in_file)
save_name = os.path.splitext(file_name)[0]

with open(in_file, 'r') as f:
    data = json.load(f)

src_list = [i['src'] for i in data]
hyp_list = [i['hyp'] for i in data]

with open(os.path.join(save_path, save_name + '.src'), 'w') as src_file, \
    open(os.path.join(save_path, save_name + '.hyp'), 'w') as hyp_file:
    src_file.write('\n'.join(src_list) + '\n')
    hyp_file.write('\n'.join(hyp_list) + '\n')
