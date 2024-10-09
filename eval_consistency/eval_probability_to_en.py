import argparse
import re
import json
import copy


entity_list = ['&amp;', '& amp;', '&amp ;', '& amp ;', '&amp', 'amp;',
               '&AMP;', '& AMP;', '&AMP ;', '& AMP ;', '&AMP', 'AMP;',
               '&lt;', '& lt;', '&lt ;', '& lt ;', '&lt', 'lt;',
               '&LT;', '& LT;', '&LT ;', '& LT ;', '&LT', 'LT;',
               '&gt;', '& gt;', '&gt ;', '& gt ;', '&gt', 'gt;', 
               '&GT;', '& GT;', '&GT ;', '& GT ;', '&GT', 'GT;',
               '&apos;', '& apos;', '&apos ;', '& apos ;', '&apos', 'apos;',
               '&APOS;', '& APOS;', '&APOS ;', '& APOS ;', '&APOS', 'APOS;',
               '&quot;', '& quot;', '&quot ;', '& quot ;', '&quot', 'quot;',
               '&QUOT;', '& QUOT;', '&QUOT ;', '& QUOT ;', '&QUOT', 'QUOT;'
               ]
value_list = ['&', '<', '>', '\'', '\"']

punctuations = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'

sep_dict = {'zh': '', 'ja': '', 'en': ' ', 'de': ' ', 'fr': ' ', 'ko': ' '}


def list_insert(id_list, new_id, prob_list, new_prob):
    if new_id < id_list[0] - 1 or new_id > id_list[-1] + 1:  # nonadjacent aligned tokens shouldn't be merged, set margin threshold to 1
        return

    # maintain an acending id list during insertion
    i = 0
    for num in id_list:
        if num < new_id:
            i += 1
        else:
            break
    id_list.insert(i, new_id)
    prob_list.insert(i, new_prob)


def E2V(sequence_list):
    "Substitute the Value for the Character Entities in the file"

    return_list = []
    for sequence in sequence_list:
        for i in range(len(entity_list)):
            (sequence, times) = re.subn(entity_list[i], value_list[int(i / 12)], sequence)
        (sequence, times) = re.subn(' U . S . ', ' U.S. ', sequence)
        return_list.append(sequence)

    return return_list


def find_max_idx(lst: list):
    max_value = max(lst)
    max_value_idx = lst.index(max_value)
    return max_value_idx


def max_prob_hyp_idx(prob_list: list, align_list: list):
    max_idx = prob_list.index(max(prob_list))
    return align_list[max_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--align_file', type=str, help='align file path')
    parser.add_argument('-t', '--token_file', type=str, help='token file path')
    parser.add_argument('--tag_file', type=str, help='tag file path')
    parser.add_argument('-p', '--pos_tool', choices=['nltk', 'spacy'], help='pos analysing tool')
    parser.add_argument('-e', '--entity_file', type=str, help='entity file path')
    parser.add_argument('--stopwords', type=str, default=None, help='stop words file')
    parser.add_argument('--prob_file', type=str, help='probability file')
    parser.add_argument('-o', '--out_file', type=str, help='output file path')
    parser.add_argument('-r', '--record_file', type=str, help='entity translation record file path')
    parser.add_argument('-l', '--language', type=str, default='en-zh', help='languge pair in src-tgt format, e.g. en-zh')
    args = parser.parse_args()

    src_lang = args.language[:2]
    tgt_lang = args.language[-2:]

    src_sep = sep_dict[src_lang]
    tgt_sep = sep_dict[tgt_lang]

    with open(args.align_file, 'r') as af, open(args.token_file, 'r') as tf, open(args.tag_file) as tagf, open(args.prob_file, 'r') as pf:
        aligh_list = [line.strip() for line in af]
        aligh_list = [line.split() for line in aligh_list]
        tok_list = [line.strip() for line in tf]
        tag_list = [line.strip() for line in tagf]

        prob_list = [line.strip() for line in pf]
        prob_list = [[float(prob) for prob in line.split()] for line in prob_list]
    
    assert len(aligh_list) == len(tok_list) == len(tag_list) == len(prob_list), (len(aligh_list), len(tok_list), len(tag_list), len(prob_list))

    stopwords = None
    if args.stopwords is not None:
        with open(args.stopwords, 'r') as f:
            stopwords = [line.strip() for line in f]
        stopwords = set(stopwords)

    history_dict = dict()
    trans_record = dict()

    ent_total, exact_ent_match, fuzzy_ent_match = 0, 0, 0
    pn_name = 'NNP' if args.pos_tool == 'nltk' else 'PROPN'

    for sent_idx, (aligns, probs, toks, tags) in enumerate(zip(aligh_list, prob_list, tok_list, tag_list)):
        
        entity_list = []

        align_dict = dict()
        prob_dict = dict()
        for align, prob in zip(aligns, probs):
            src_align_id, hyp_align_id = align.split('-')
            src_align_id, hyp_align_id = int(src_align_id), int(hyp_align_id)
            if src_align_id in align_dict:
                list_insert(align_dict[src_align_id], hyp_align_id, prob_dict[src_align_id], prob)  # insert the target word into alignment list
            else:
                align_dict[src_align_id] = [hyp_align_id]
                prob_dict[src_align_id] = [prob]

        if len(toks.split(' ||| ')) != 2:
            continue

        src_tok, hyp_tok = toks.split(' ||| ')

        src_tok_list = src_tok.split()
        hyp_tok_list = hyp_tok.split()
        
        # denormalize the punctuations
        src_tok_list = E2V(src_tok_list)
        hyp_tok_list = E2V(hyp_tok_list)
        
        src_tag_list = eval(tags)
        src_tag_list = [i for i in src_tag_list if i != 'SPACE']

        skip_cnt = 0
        assert len(src_tag_list) == len(src_tok_list), (src_tag_list, src_tok_list)
        for tok_idx, tag in enumerate(src_tag_list):
            if skip_cnt > 0:  # current entity is merged with previous ones, skip
                skip_cnt -= 1
                continue
            if tag == pn_name and tok_idx in align_dict:  # the current word is a proper noun and has some aligned target words

                src_ent = src_tok_list[tok_idx]
                
                # merge continues entities such as "Harry" and "Potter"
                jdx = tok_idx + 1
                tmp_src_ent = copy.deepcopy(src_ent)
                tmp_hyp_ent_ids = [max_prob_hyp_idx(prob_dict[tok_idx], align_dict[tok_idx])]
                tmp_probs = [max(prob_dict[tok_idx])]
                while jdx < len(src_tok_list):
                    post_src_ent = src_tok_list[jdx]
                    # if two entities share the same aligned word, merge them
                    if post_src_ent[0].isupper() and jdx in align_dict and align_dict[jdx][0] == align_dict[tok_idx][0]:
                        tmp_src_ent = tmp_src_ent + src_sep + post_src_ent
                        post_hyp_idx = max_prob_hyp_idx(prob_dict[jdx], align_dict[jdx])  # pick up the target word with the highest alignment probability as the aligned word of the post entity
                        # merge the alignment list
                        if post_hyp_idx not in tmp_hyp_ent_ids:
                            tmp_hyp_ent_ids.append(post_hyp_idx)
                            tmp_probs.append(max(prob_dict[jdx]))
                        skip_cnt += 1
                    else:
                        break
                    jdx += 1
                if skip_cnt > 0:
                    src_ent = copy.deepcopy(tmp_src_ent)
                    hyp_ent = tgt_sep.join(hyp_tok_list[i] for i in sorted(tmp_hyp_ent_ids))
                    align_prob = sum(tmp_probs) / len(tmp_probs)  # calculate the average alignment probability as the new probability
                else:
                    hyp_ent = tgt_sep.join([hyp_tok_list[i] for i in align_dict[tok_idx]])
                    align_prob = sum(prob_dict[tok_idx]) / len(prob_dict[tok_idx]) 
                
                if len(hyp_ent) < 1:
                    continue

                tmp_str = re.sub(punctuations, "", src_ent)  # remove all punctuations

                if len(tmp_str) <= 1:
                    continue
                
                if stopwords is not None:  # filter out stop words
                    if src_ent.lower() in stopwords:
                        continue
                
                # matching window is from the previous word to the post word to address alignment errors
                prev_hyp_tok = hyp_tok_list[align_dict[tok_idx][0] - 1] if align_dict[tok_idx][0] - 1 >= 0 else ''
                next_hyp_tok = hyp_tok_list[align_dict[tok_idx][-1] + 1] if align_dict[tok_idx][-1] + 1 < len(hyp_tok_list) else ''

                if src_ent not in trans_record:
                    trans_record[src_ent] = dict()
                if hyp_ent in trans_record[src_ent]:
                    trans_record[src_ent][hyp_ent] += 1
                elif prev_hyp_tok + hyp_ent in trans_record[src_ent]:
                    trans_record[src_ent][prev_hyp_tok + hyp_ent] += 1
                elif hyp_ent + next_hyp_tok in trans_record[src_ent]:
                    trans_record[src_ent][hyp_ent + next_hyp_tok] += 1
                elif prev_hyp_tok + hyp_ent + next_hyp_tok in trans_record[src_ent]:
                    trans_record[src_ent][prev_hyp_tok + hyp_ent + next_hyp_tok] += 1
                else:
                    trans_record[src_ent][hyp_ent] = 1

                if src_ent in history_dict:
                    # history_dict[src_ent]['name'][0] is the entity in source language, history_dict[src_ent]['name'][0] is that in target language
                    # the entity has not been translated, encountered a new translation
                    if history_dict[src_ent]['name'][0] is None and hyp_ent != src_ent:
                        history_dict[src_ent]['name'][0] = hyp_ent
                        continue
                    
                    ent_total += 1

                    history_loc = f"{history_dict[src_ent]['loc']['line']}:{','.join(history_dict[src_ent]['loc']['ids'])}:{history_dict[src_ent]['loc']['prob']}"
                    if hyp_ent in history_dict[src_ent]['name']:
                        entity_list.append((src_ent, hyp_ent, history_dict[src_ent]['name'][0], history_loc, align_prob))
                        exact_ent_match += 1
                        fuzzy_ent_match += 1
                    elif prev_hyp_tok + hyp_ent in history_dict[src_ent]['name']:
                        entity_list.append((src_ent, prev_hyp_tok + hyp_ent, history_dict[src_ent]['name'][0], history_loc, align_prob))
                        exact_ent_match += 1
                        fuzzy_ent_match += 1
                    elif hyp_ent + next_hyp_tok in history_dict[src_ent]['name']:
                        entity_list.append((src_ent, hyp_ent + next_hyp_tok, history_dict[src_ent]['name'][0], history_loc, align_prob))
                        exact_ent_match += 1
                        fuzzy_ent_match += 1
                    elif prev_hyp_tok + hyp_ent + next_hyp_tok in history_dict[src_ent]['name']:
                        entity_list.append((src_ent, prev_hyp_tok + hyp_ent + next_hyp_tok, history_dict[src_ent]['name'][0], history_loc, align_prob))
                        exact_ent_match += 1
                        fuzzy_ent_match += 1
                    else:  # unmatched, translation inconsistency detected
                        entity_list.append((src_ent, hyp_ent, history_dict[src_ent]['name'][0], history_loc, align_prob))
                        
                        # fuzzy match
                        if history_dict[src_ent]['name'][0] in hyp_ent or hyp_ent in history_dict[src_ent]['name'][0] \
                            or history_dict[src_ent]['name'][1] in hyp_ent or hyp_ent in history_dict[src_ent]['name'][1]:
                            fuzzy_ent_match += 1

                else:  # newly encountered entity
                    ids_list = [str(i) for i in align_dict[tok_idx]] if skip_cnt == 0 else [str(i) for i in tmp_hyp_ent_ids]
                    if src_ent == hyp_ent:  # the entity is remain untranslated
                        history_dict[src_ent] = {'name': [None, src_ent], 'loc': {'line': sent_idx, 'ids': ids_list, 'prob': align_prob}}
                    else:
                        history_dict[src_ent] = {'name': [hyp_ent, src_ent], 'loc': {'line': sent_idx, 'ids': ids_list, 'prob': align_prob}}


        with open(args.entity_file, 'a') as ef:
            ef.write(str(entity_list) + '\n')

    outf = open(args.out_file, 'a')
    if ent_total == 0:
        print(f'Exact: {exact_ent_match: >4} / {ent_total: <4} = {0: <.2f}%')
        print(f'Fuzzy: {fuzzy_ent_match: >4} / {ent_total: <4} = {0: <.2f}%')
        outf.write(f'Exact: {exact_ent_match: >4} / {ent_total: <4} = {0: <.2f}%\n')
        outf.write(f'Fuzzy: {fuzzy_ent_match: >4} / {ent_total: <4} = {0: <.2f}%\n')
    else:
        print(f'Exact: {exact_ent_match: >4} / {ent_total: <4} = {exact_ent_match / ent_total * 100: <.2f}%')
        print(f'Fuzzy: {fuzzy_ent_match: >4} / {ent_total: <4} = {fuzzy_ent_match / ent_total * 100: <.2f}%')
        outf.write(f'Exact: {exact_ent_match: >4} / {ent_total: <4} = {exact_ent_match / ent_total * 100: <.2f}%\n')
        outf.write(f'Fuzzy: {fuzzy_ent_match: >4} / {ent_total: <4} = {fuzzy_ent_match / ent_total * 100: <.2f}%\n')

    vote_correct, vote_total = 0, 0
    for src_ent in trans_record:
        hyp_record = trans_record[src_ent]
        vote_total += sum(hyp_record.values())
        gold_hyp_ent = max(hyp_record, key=lambda x: hyp_record[x] if x != src_ent else 0)

        for hyp_ent in hyp_record:
            if hyp_ent == src_ent:
                vote_correct += hyp_record[hyp_ent]
            elif hyp_ent in gold_hyp_ent or gold_hyp_ent in hyp_ent:
                vote_correct += hyp_record[hyp_ent]

    if vote_total == 0:
        print(f'Vote:  {vote_correct: >4} / {vote_total: <4} = {0: <.2f}%')
        outf.write(f'Vote:  {vote_correct: >4} / {vote_total: <4} = {0: <.2f}%\n')
    else:
        print(f'Vote:  {vote_correct: >4} / {vote_total: <4} = {vote_correct / vote_total * 100: <.2f}%\n')
        outf.write(f'Vote:  {vote_correct: >4} / {vote_total: <4} = {vote_correct / vote_total * 100: <.2f}%\n')
    outf.close()
    json.dump(trans_record, open(args.record_file, 'w'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
