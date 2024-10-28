
lang=en-zh              # language pair, en<->zh,de,fr,ja supported
src_lang=${lang%%-*}
tgt_lang=${lang##*-}

# use_model=gpt35turbo
use_model=gpt4omini

summary_step=20         # update summary every $summary_step sentences, 20 recommanded
long_window=20          # long-term memory window size, 20 recommanded
top_k=2                 # long-term memory retrieve sentence number, 2 recommanded
context_window=3        # short-term memory window size, 3 recommanded

pyfile=infer_gpt.py

out_path=results

src=/path/to/src/file
ref=/path/to/ref/file

export API_BASE=
export API_KEY=

src_summary_prompt=prompts/${lang}/src_summary_prompt.txt               # prompt for source-side summary generation
tgt_summary_prompt=prompts/${lang}/tgt_summary_prompt.txt               # prompt for target-side summary generation
src_merge_prompt=prompts/${lang}/src_merge_prompt.txt                   # prompt for source-side summary merging
tgt_merge_prompt=prompts/${lang}/tgt_merge_prompt.txt                   # prompt for target-side summary merging
history_prompt=prompts/${lang}/history_prompt.txt                       # prompt for proper noun extraction
retrieve_prompt=prompts/retrieve_prompt.txt                             # prompt for short-term memory retrieval
trans_prompt=prompts/trans_summary_long_context_history_prompt.txt      # prompt for document translation

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
    --settings summary long context history \
    --context_window $context_window \
    --retriever agent \
    --model $use_model

