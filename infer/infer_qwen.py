import argparse
import numpy as np
from numpy.linalg import norm
import json
from typing import Tuple, Union, List
from tqdm import tqdm
from copy import deepcopy
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os


TRANS_CNT = 0

sep_map = {'zh': '', 'ja': '', 'en': ' ', 'de': ' ', 'fr': ' '}
SRC_SEP, TGT_SEP = None, None

lang_dict = {'zh': 'Chinese', 'ja': 'Japanese', 'en': 'English', 'de': 'German', 'fr': 'French'}

model, tokenizer = None, None

def invoke_chat(prompt: str) -> str:
    
    messages = [
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response


def cosine_similarity(a: Union[np.array, List], b: Union[np.array, List]):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))


class EmbeddingDict():
    def __init__(self, total: int, recency_weight: float, similarity_weight: float, skip_context: int) -> None:
        self.embedding_list = []
        self.src_text_list = []
        self.tgt_text_list = []
        self.total = total
        assert recency_weight + similarity_weight == 10.0, "The weights should be added up to 10!"
        self.recency_weight = recency_weight / 10.0
        self.similarity_weight = similarity_weight / 10.0
    
    def insert(self, new_src: str, new_tgt: str) -> None:
        if self.total == -1 or len(self.embedding_list) < self.total:
            self.src_text_list.append(new_src)
            self.tgt_text_list.append(new_tgt)
        else:
            self.src_text_list = self.src_text_list[1:] + [new_src]
            self.tgt_text_list = self.tgt_text_list[1:] + [new_tgt]
    
    def match(self, query: str, num: int) -> Tuple[List[str]]:
        if len(self.embedding_list) <= num:
            return (self.src_text_list, self.tgt_text_list)
        query_embedding = invoke_embedding(query)
        sim_list = [cosine_similarity(query_embedding, i) for i in self.embedding_list]
        rec_list = [i / (len(self.embedding_list) - 1) for i in range(len(self.embedding_list))]
        score_list = [self.recency_weight * i + self.similarity_weight * j for i, j in zip(rec_list, sim_list)]

        idx_list = list(range(len(score_list)))
        idx_list.sort(key=lambda x: score_list[x], reverse=True)

        if self.total == -1 or len(self.embedding_list) < self.total:
            self.embedding_list.append(query_embedding)
        else:
            self.embedding_list = self.embedding_list[1:] + [query_embedding]
        return ([self.src_text_list[i] for i in idx_list[:num]], [self.tgt_text_list[i] for i in idx_list[:num]])
    

class RetrieveAgent():
    def __init__(self, total: int, recency_weight: float, similarity_weight: float, prompt_template: str, skip_context: int) -> None:
        self.src_text_list = []
        self.tgt_text_list = []
        self.total = total
        assert recency_weight + similarity_weight == 10.0, "The weights should be added up to 10!"
        self.recency_weight = recency_weight / 10.0
        self.similarity_weight = similarity_weight / 10.0
        
        self.prompt_template = prompt_template

        self.example_number = None

    def insert(self, new_src: str, new_tgt: str) -> None:
        if self.total == -1 or len(self.src_text_list) < self.total:
            self.src_text_list.append(new_src)
            self.tgt_text_list.append(new_tgt)
        else:
            self.src_text_list = self.src_text_list[1:] + [new_src]
            self.tgt_text_list = self.tgt_text_list[1:] + [new_tgt]
    
    def match(self, query: str, num: int) -> Tuple[List[str]]:
        if len(self.src_text_list) <= num:
            return (self.src_text_list, self.tgt_text_list)

        sent_list = ''
        for idx, src in enumerate(self.src_text_list):
            sent_list += f'<Sentence {idx + 1}> {src}\n'
        sent_list = sent_list.strip()

        if self.example_number is None or len(self.example_number) != num:
            random.seed(0)
            self.example_number = random.sample(list(range(max(10, num))), num)
            self.example_number.sort()
        example_num_prompt = [str(i) for i in self.example_number]
        example_num_prompt = ', '.join(example_num_prompt[:-1]) + ' and ' + example_num_prompt[-1] if num > 1 else example_num_prompt[0]
        example_list_prompt = str(self.example_number)

        prompt = self.prompt_template.format(
            top_num=num,
            sentence_list=sent_list,
            example_number=example_num_prompt,
            example_list=example_list_prompt,
            query=query
        )

        chosen_ids = invoke_chat(prompt)
        if chosen_ids is None:
            return ([], [])
        try:
            chosen_ids = eval(chosen_ids)
        except Exception as e:
            chosen_ids = []
        chosen_ids = [i for i in chosen_ids if type(i) is int and 1 <= i <= len(self.src_text_list)]
        chosen_ids.sort()
        return ([self.src_text_list[i-1] for i in chosen_ids], [self.tgt_text_list[i-1] for i in chosen_ids])


def init_memory(approach: str):
    if approach == 'embedding':
        class LongTimeMemory(EmbeddingDict):
            def __init__(self, total: int, recency_weight: float, similarity_weight: float, skip_context: int) -> None:
                super().__init__(total, recency_weight, similarity_weight, skip_context)

    elif approach == 'agent':
        class LongTimeMemory(RetrieveAgent):
            def __init__(self, total: int, recency_weight: float, similarity_weight: float, prompt_template: str, skip_context: int) -> None:
                super().__init__(total, recency_weight, similarity_weight, prompt_template, skip_context)

    else:
        print('The approach of the retriever must be "embedding" or "agent"!')

    return LongTimeMemory


def translate(
        src_lang: str, tgt_lang:str,
        src_text: str,
        rel_src_sents: List[str], rel_tgt_sents: List[str],
        src_summary: str, tgt_summary: str,
        historical_prompt: str,
        src_context: list, tgt_context: list, context_window: int,
        prompt_template: str
    ) -> dict:
    if rel_src_sents is None or len(rel_src_sents) == 0:
        rel_instances = 'N/A'
    else:
        rel_instances = ''
        for rel_src, rel_tgt in zip(rel_src_sents, rel_tgt_sents):
            rel_instances += f'<{lang_dict[src_lang]} source> {rel_src}\n<{lang_dict[tgt_lang]} translation> {rel_tgt}\n'
        rel_instances = rel_instances.strip()
    if src_summary is None:
        src_summary = 'N/A'
    if tgt_summary is None:
        tgt_summary = 'N/A'
    if historical_prompt is None or historical_prompt == '':
        historical_prompt = 'N/A'
    if src_context is None or len(src_context) == 0:
        src_context_prompt, tgt_context_prompt = 'N/A', 'N/A'
    else:
        global SRC_SEP, TGT_SEP
        src_context_prompt = SRC_SEP.join(src_context)
        tgt_context_prompt = TGT_SEP.join(tgt_context)
    
    prompt = prompt_template.format(
        src_lang=lang_dict[src_lang],
        tgt_lang=lang_dict[tgt_lang],
        src_summary=src_summary,
        tgt_summary=tgt_summary,
        rel_inst=rel_instances,
        src=src_text,
        hist_info=historical_prompt,
        src_context=src_context_prompt,
        tgt_context=tgt_context_prompt,
        context_window=context_window,
    )

    hyp = invoke_chat(prompt)
    if hyp is None:
        hyp = ''
    else:
        hyp = hyp.split('\n')[0]
    global TRANS_CNT
    if (TRANS_CNT + 1) % 10 == 0:
        print('\n\nprompt:')
        print(prompt + '\n\n')
    TRANS_CNT += 1

    return hyp


class Summary():
    def __init__(self, src_gen_template: str, tgt_gen_template: str, src_merge_template: str, tgt_merge_template: str) -> None:
        self.src_summary = None
        self.tgt_summary = None
        self.src_gen_template = src_gen_template
        self.tgt_gen_template = tgt_gen_template
        self.src_merge_template = src_merge_template
        self.tgt_merge_template = tgt_merge_template

    def set_summary(self, s_sum: str, t_sum) -> None:
        self.src_summary = s_sum
        self.tgt_summary = t_sum

    def gen_summary(self, record_list: List[dict]) -> Tuple[str]:
        src_list = [i['src'] for i in record_list]
        hyp_list = [i['hyp'] for i in record_list]
        
        src_para = SRC_SEP.join(src_list)
        hyp_para = TGT_SEP.join(hyp_list)
        
        prompt = self.src_gen_template.format(src_para=src_para)
        src_summary = invoke_chat(prompt)

        prompt = self.tgt_gen_template.format(src_para=hyp_para)
        tgt_summary = invoke_chat(prompt)

        return (src_summary, tgt_summary)

    def merge_summary(self, src_new_sum, tgt_new_sum) -> Tuple[str]:
        if self.src_summary is None:
            return (src_new_sum, tgt_new_sum)
        prompt = self.src_merge_template.format(summary_1=self.src_summary, summary_2=src_new_sum)
        src_sum = invoke_chat(prompt)

        prompt = self.tgt_merge_template.format(summary_1=self.tgt_summary, summary_2=tgt_new_sum)
        tgt_sum = invoke_chat(prompt)

        return (src_sum, tgt_sum)

    def update_summary(self, record_list: List[dict]) -> Tuple[str]:
        tmp_src_summary, tmp_tgt_summary = self.gen_summary(record_list)
        self.src_summary, self.tgt_summary = self.merge_summary(tmp_src_summary, tmp_tgt_summary)
        return (self.src_summary, self.tgt_summary)
    
    def get_summary(self) -> Tuple[str, str]:
        return (self.src_summary, self.tgt_summary)


class History():
    def __init__(self, prompt_template: str, src_lang: str, tgt_lang: str) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.entity_dict = dict()
        self.prompt_template = prompt_template

    def extract_entity(self, src: str, tgt: str) -> List[str]:
        prompt = self.prompt_template.format(
            src_lang=lang_dict[self.src_lang],
            tgt_lang=lang_dict[self.tgt_lang],
            src=src,
            tgt=tgt
        )
        new_info = invoke_chat(prompt)
        
        conflicts = list()
        if new_info is not None and new_info not in ['N/A', 'None', '', '无']:
            new_proper_noun_pairs = new_info.split(', ')
            for ent_pair in new_proper_noun_pairs:
                if len(ent_pair.split(' - ')) == 2:
                    src_ent, tgt_ent = ent_pair.split(' - ')
                    src_ent = src_ent.replace('\"', '').replace('\'', '')
                    tgt_ent = tgt_ent.replace('\"', '').replace('\'', '')
                    if self.entity_dict.get(src_ent, '') == '':
                        self.entity_dict[src_ent] = tgt_ent if tgt_ent != 'N/A' else src_ent
                    elif self.entity_dict[src_ent] != tgt_ent:
                        conflicts.append(f'"{src_ent}" - "{self.entity_dict[src_ent]}"/"{tgt_ent}"')
        return conflicts
    
    def buildin_history(self, sentence: str, only_relative: bool) -> str:
        if only_relative:
            entity_list = [ent for ent in self.entity_dict if ent in sentence]
            hist_list = [f'"{ent}" - "{self.entity_dict[ent]}"' for ent in entity_list if ent in self.entity_dict]
            hist_prompt = ', '.join(hist_list)
            return hist_prompt
        else:
            hist_list = [f'"{ent}" - "{self.entity_dict[ent]}"' for ent in self.entity_dict]
            hist_prompt = ', '.join(hist_list)
            return hist_prompt

    def get_history_dict(self) -> dict:
        return deepcopy(self.entity_dict)

    def set_history_dict(self, h_dict: dict) -> None:
        self.entity_dict = h_dict


class Context():
    def __init__(self, window_size: int) -> None:
        self.windows_size = window_size
        self.src_context = []
        self.tgt_context = []
    
    def update(self, src: str, tgt: str) -> None:
        if self.windows_size == -1:
            self.src_context.append(src)
            self.tgt_context.append(tgt)
        else:
            self.src_context = self.src_context[-(self.windows_size - 1):] + [src]
            self.tgt_context = self.tgt_context[-(self.windows_size - 1):] + [tgt]
    
    def get_context(self) -> Tuple[List[str]]:
        return (self.src_context, self.tgt_context)


def init(args, src_sum_tpl, tgt_sum_tpl, src_mer_tpl, tgt_mer_tpl, src_lang, tgt_lang):

    trans_context, long_memory, doc_summary, ent_history = None, None, None, None

    if 'context' in args.settings:
            trans_context = Context(args.context_window)

    if 'long' in args.settings:
        LongMemory = init_memory(args.retriever)
        if args.retriever == 'agent':
            with open(args.retrieve_prompt) as trf:
                long_tpl = trf.read()
            long_memory = LongMemory(args.long_window, args.recency_weight, args.similarity_weight, long_tpl, args.context_window)
        else:
            long_memory = LongMemory(args.long_window, args.recency_weight, args.similarity_weight, args.context_window)
    
    if 'summary' in args.settings:
        doc_summary = Summary(src_sum_tpl, tgt_sum_tpl, src_mer_tpl, tgt_mer_tpl)

    if 'history' in args.settings:
        with open(args.history_prompt) as thf:
            history_tpl = thf.read()
        ent_history = History(history_tpl, src_lang, tgt_lang)

    trans_records = []

    if os.path.isfile(args.output):
        with open(args.output, 'r') as f:
            trans_records = json.load(f)
        for record in trans_records:
            if doc_summary is not None and 'new_src_summary' in record:
                doc_summary.set_summary(record['new_src_summary'], record['new_tgt_summary'])

            if long_memory is not None:
                long_memory.insert(record['src'], record['hyp'])

            if ent_history is not None:
                ent_history.set_history_dict(record['entity_dict'])

            if 'context' in args.settings:
                trans_context.update(record['src'], record['hyp'])


    return trans_context, long_memory, doc_summary, ent_history, trans_records


def main():

    src_lang = args.language[:2]
    tgt_lang = args.language[-2:]

    global SRC_SEP, TGT_SEP
    SRC_SEP, TGT_SEP = sep_map[src_lang], sep_map[tgt_lang]

    with open(args.src, 'r') as sf:
        src_list = [line.strip() for line in sf]
    
    if args.ref:
        with open(args.ref, 'r') as rf:
            ref_list = [line.strip() for line in rf]
    
    with open(args.src_summary_prompt, 'r') as ssf, open(args.tgt_summary_prompt, 'r') as tsf, \
        open(args.src_merge_prompt, 'r') as smf, open(args.tgt_merge_prompt, 'r') as tmf, \
        open(args.trans_prompt, 'r') as tf:
        src_sum_tpl = ssf.read()
        tgt_sum_tpl = tsf.read()
        src_mer_tpl = smf.read()
        tgt_mer_tpl = tmf.read()
        trans_tpl = tf.read()
    

    if 'context' not in args.settings:
        args.context_window = 0

    trans_context, long_memory, doc_summary, ent_history, trans_records = init(args, src_sum_tpl, tgt_sum_tpl, src_mer_tpl, tgt_mer_tpl, src_lang, tgt_lang)

    print(f'### Resuming from {len(trans_records)} records...')

    if len(trans_records) >= len(src_list):
        exit()
    else:
        global tokenizer, model
        tokenizer = AutoTokenizer.from_pretrained(modelpath, cache_dir=modelpath)

        model = AutoModelForCausalLM.from_pretrained(
            modelpath, cache_dir=modelpath, torch_dtype="auto", device_map="auto"
        )
    for idx in tqdm(range(len(trans_records), len(src_list), 1)):
        record = dict()

        src = src_list[idx]
        ref = ref_list[idx] if args.ref else None

        long_mem_srcs, long_mem_tgts = None, None
        src_summary, tgt_summary = None, None
        hist_info = None
        src_context, tgt_context = None, None

        if 'long' in args.settings:
            long_mem_srcs, long_mem_tgts = long_memory.match(src, args.top_k)
            long_mem_srcs, long_mem_tgts = deepcopy(long_mem_srcs), deepcopy(long_mem_tgts)

        if 'summary' in args.settings:
            src_summary, tgt_summary = doc_summary.get_summary()
        
        if 'history' in args.settings:
            hist_info = ent_history.buildin_history(src, args.only_relative)
        
        if 'context' in args.settings:
            src_context, tgt_context = trans_context.get_context()

        result = translate(src_lang, tgt_lang, src, long_mem_srcs, long_mem_tgts, src_summary, tgt_summary, hist_info, src_context, tgt_context, args.context_window, trans_tpl)

        hyp = result

        record['idx'] = idx
        record['src'] = src
        if ref:
            record['ref'] = ref
        record['hyp'] = hyp

        if 'summary' in args.settings and (idx + 1) % args.summary_step == 0:
            record['new_src_summary'], record['new_tgt_summary'] = doc_summary.update_summary(trans_records[-args.summary_step:])

        if 'long' in args.settings:
            record['rel_src'] = long_mem_srcs
            record['rel_tgt'] = long_mem_tgts
            long_memory.insert(src, hyp)

        if 'history' in args.settings:
            conflict_list = ent_history.extract_entity(src, hyp)
            if args.only_relative:
                record['hist_info'] = hist_info
            record['entity_dict'] = ent_history.get_history_dict()
            if len(conflict_list) > 0:
                record['conflict'] = conflict_list

        if 'context' in args.settings:
            trans_context.update(src, hyp)

        
        trans_records.append(record)
        json.dump(trans_records, open(args.output, 'w'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', type=str)
    parser.add_argument('-s', '--src', type=str)
    parser.add_argument('-r', '--ref', type=str, default=None)
    parser.add_argument('--src_summary_prompt', type=str)
    parser.add_argument('--tgt_summary_prompt', type=str)
    parser.add_argument('--src_merge_prompt', type=str)
    parser.add_argument('--tgt_merge_prompt', type=str)
    parser.add_argument('--retrieve_prompt', type=str, default=None)
    parser.add_argument('--history_prompt', type=str, default=None)
    parser.add_argument('--trans_prompt', type=str)
    parser.add_argument('--summary_step', type=int, default=10)
    parser.add_argument('--long_window', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-rw', '--recency_weight', type=float, default=0.0)
    parser.add_argument('-sw', '--similarity_weight', type=float, default=10.0)
    parser.add_argument('--only_relative', type=bool, default=True)
    parser.add_argument('--settings', nargs='+', type=str)
    parser.add_argument('--context_window', type=int, default=3)
    parser.add_argument('--retriever', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    modelpath = args.model
    device = "cuda"

    main()
