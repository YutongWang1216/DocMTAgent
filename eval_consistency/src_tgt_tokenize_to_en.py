import jieba
import argparse
import nltk
import spacy
from tqdm.contrib import tzip


punctuations = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+。'

data_dict = {'en': 'en_core_web_sm', 'zh': 'zh_core_web_sm', 'de': 'de_core_news_sm', 'fr': 'fr_core_news_sm', 'ja': 'ja_core_news_sm', 'ko': 'ko_core_news_sm'}


def sentence_tokenize(sentence, lang, tok_tool, nlp_tool, side):
    if side == 'tgt' and lang == 'zh':
        return list(jieba.cut(sentence)), None
    else:
        if tok_tool == 'nltk':
            tok_list = nltk.word_tokenize(sentence)
            tag_list = nltk.pos_tag(tok_list)
            tag_list = [item[1] for item in tok_list]
            return tok_list, tag_list
        else:
            doc = nlp_tool(sentence)
            results = [token.text for token in doc if token.text != '|||'], [token.pos_ for token in doc if token.text != '|||']
            return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', type=str, help='source file path')
    parser.add_argument('-t', '--target_file', type=str, help='target file path')
    parser.add_argument('-l', '--language', type=str, help='language pair')
    parser.add_argument('-o', '--output', type=str, help='output file path')
    parser.add_argument('--tag_file', type=str, help='tag file path')
    parser.add_argument('--tool', type=str, choices=['nltk', 'spacy'], help='tokenizing tool')
    args = parser.parse_args()

    src_lang = args.language[:2]
    tgt_lang = args.language[-2:]

    with open(args.source_file, 'r') as sf, open(args.target_file, 'r') as tf:
        srcs = [line.strip() for line in sf]
        tgts = [line.strip() for line in tf]


    write_pairs = []
    write_tags = []

    src_nlp, tgt_nlp = None, None

    src_nlp = spacy.load(f"{data_dict[src_lang]}")
    tgt_nlp = spacy.load(f"{data_dict[tgt_lang]}")

    for src, hyp in tzip(srcs, tgts):
        if src == '' or hyp == '':
            continue
        
        src = src.replace('|||', '')
        hyp = hyp.replace('|||', '')
        
        tokenized_src, src_tags = sentence_tokenize(src, src_lang, args.tool, src_nlp, 'src')
        tokenized_hyp, tgt_tags = sentence_tokenize(hyp, tgt_lang, args.tool, tgt_nlp, 'tgt')

        write_pair = ' '.join(tokenized_src) + ' ||| ' + ' '.join(tokenized_hyp)
        write_tag = str(src_tags)

        write_pairs.append(write_pair)
        write_tags.append(write_tag)
    
    with open(args.output, 'w') as outf, open(args.tag_file, 'w') as tagf:
        outf.write('\n\n'.join(write_pairs) + '\n')
        tagf.write('\n\n'.join(write_tags) + '\n')


if __name__ == '__main__':
    main()
