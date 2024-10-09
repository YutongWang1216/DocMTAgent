
lang=$1
src_lang=${lang%%-*}
tgt_lang=${lang##*-}

# use_model=gpt35turbo
use_model=gpt4omini

summary_step=20
long_window=20
top_k=2

recency_weight=0
similarity_weight=10

context_window=3

pyfile=infer_gpt.py

out_path=results

src=/path/to/src/file
ref=/path/to/ref/file

src_summary_prompt=prompts/${lang}/src_summary_prompt.txt
tgt_summary_prompt=prompts/${lang}/tgt_summary_prompt.txt
src_merge_prompt=prompts/${lang}/src_merge_prompt.txt
tgt_merge_prompt=prompts/${lang}/tgt_merge_prompt.txt
history_prompt=prompts/${lang}/history_prompt.txt
retrieve_prompt=prompts/retrieve_prompt.txt
trans_prompt=prompts/trans_summary_long_context_history_prompt.txt

if [ ! -d $out_path ]; then
    mkdir -p $out_path
fi

output=$out_path/$use_model.json

python -u $pyfile \
    --language ${lang} \
    --src $src --ref $ref\
    --src_summary_prompt $src_summary_prompt \
    --tgt_summary_prompt $tgt_summary_prompt \
    --src_merge_prompt $src_merge_prompt \
    --tgt_merge_prompt $tgt_merge_prompt \
    --retrieve_prompt $retrieve_prompt \
    --history_prompt $history_prompt \
    --trans_prompt $trans_prompt \
    --summary_step $summary_step \
    --long_window $long_window \
    --top_k $top_k \
    --output $output \
    --recency_weight $recency_weight --similarity_weight $similarity_weight \
    --settings summary long context history \
    --context_window $context_window \
    --retriever agent \
    --model $use_model

