import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--entity_file', help='entity file path')
    parser.add_argument('-o', '--out_file', help='output file path')
    args = parser.parse_args()

    record = dict()
    with open(args.entity_file, 'r') as f:
        lines = [line.strip() for line in f]

    for line in lines:
        info_list = eval(line)
        for info in info_list:
            src_ent, hyp_ent, his_ent, loc, prob = info
            if src_ent not in record:
                hist_prob = float(loc.split(':')[-1])
                record[src_ent] = {'total_num': 1, 'inconsistent_num': 0, his_ent if his_ent is not None else src_ent: [1, hist_prob, hist_prob]}
            
            record[src_ent]['total_num'] += 1
            if hyp_ent != his_ent and hyp_ent != src_ent:
                record[src_ent]['inconsistent_num'] += 1

            if hyp_ent not in record[src_ent]:
                record[src_ent][hyp_ent] = [0, float('-inf'), float('inf')]
            record[src_ent][hyp_ent][0] += 1
            record[src_ent][hyp_ent][1] = max(record[src_ent][hyp_ent][1], prob)
            record[src_ent][hyp_ent][2] = min(record[src_ent][hyp_ent][2], prob)

    key_list = list(record.keys())
    key_list.sort(key=lambda x: (record[x]['inconsistent_num']), reverse=True)

    sorted_record = {k: record[k] for k in key_list}
    for ent in sorted_record:
        hyp_list = list(sorted_record[ent].keys())
        hyp_list = [i for i in hyp_list if i not in ['total_num', 'inconsistent_num']]
        hyp_list.sort(key=lambda x: sorted_record[ent][x][0], reverse=True)
        
        sorted_item = {'total_num': sorted_record[ent]['total_num'], 'inconsistent_num': sorted_record[ent]['inconsistent_num']}
        for hyp_item in hyp_list:
            sorted_item[hyp_item] = sorted_record[ent][hyp_item]
        sorted_record[ent] = sorted_item

    json.dump(sorted_record, open(args.out_file, 'w'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
