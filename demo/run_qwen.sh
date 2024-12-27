#!/bin/bash
source ~/.bashrc
# modify the path to your conda env
source /PATH/TO/YOUR/CONDA/bin/activate myenv # env activate

export Qwen2-72B-path=$(realpath models/Qwen2-72B-Instruct)
export Qwen2-7B-path=$(realpath models/Qwen2-7B-Instruct)

lang=${1:-"en-zh"}              # Default value is "en-zh"
use_model=${2:-"Qwen2-7B-Instruct"} # Default value is "Qwen2-7B-Instruct"
src=${3:-"/path/to/src/file"}   # Default value is "/path/to/src/file"

src_lang=${lang%%-*}
tgt_lang=${lang##*-}

# modify the path to your model path
if [ $use_model == "Qwen2-72B-Instruct" ]; then
    modelpathroot=$Qwen2-72B-path
elif [ $use_model == "Qwen2-7B-Instruct" ]; then
    modelpathroot=$Qwen2-7B-path
fi

echo "use_model=$use_model"
echo "modelpathroot=$modelpathroot"

summary_step=20         # Update summary every $summary_step sentences, 20 recommended
long_window=20          # Long-term memory window size, 20 recommended
top_k=2                 # Long-term memory retrieve sentence number, 2 recommended
context_window=3        # Short-term memory window size, 3 recommended

preprocess="preprocess.py"
infer="../infer/infer_qwen.py"
postprocess="postprocess.py"

# src=/path/to/src/file
# ref=/path/to/ref/file

src_summary_prompt=$(realpath "prompts/${lang}/src_summary_prompt.txt")   # Prompt for source-side summary generation
tgt_summary_prompt=$(realpath "prompts/${lang}/tgt_summary_prompt.txt")   # Prompt for target-side summary generation
src_merge_prompt=$(realpath "prompts/${lang}/src_merge_prompt.txt")       # Prompt for source-side summary merging
tgt_merge_prompt=$(realpath "prompts/${lang}/tgt_merge_prompt.txt")       # Prompt for target-side summary merging
history_prompt=$(realpath "prompts/${lang}/history_prompt.txt")           # Prompt for proper noun extraction
retrieve_prompt=$(realpath "prompts/retrieve_prompt_qwen.txt")            # Prompt for short-term memory retrieval
trans_prompt=$(realpath "prompts/trans_summary_long_context_history_prompt.txt") # Prompt for document translation

if [ ! -d $out_path ]; then
    mkdir -p $out_path
fi

START_TIME=$(date "+%Y%m%d_%H%M%S")

src_filename=$(basename "$src" .txt)

mkdir data/temp/$START_TIME

echo "========== Preprocessing =========="
# Preprocessing: Convert the input txt file into a one-sentence-per-line txt file
preprocess_input=$(realpath $src)
preprocess_output=$(realpath data/temp/$START_TIME/${src_filename}_preprocessed.txt)
python3 $preprocess \
    --input $preprocess_input \
    --output $preprocess_output \
    --lang ${lang}

echo "========== Inference =========="
# Translation inference: Generate a JSON file
infer_input=$(realpath $preprocess_output)
infer_output=$(realpath data/temp/$START_TIME/${src_filename}_inferred.json)
python3 $infer \
    --language ${lang} \
    --src ${infer_input} \
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
    --output $infer_output \
    --settings summary long context history \
    --context_window $context_window \
    --retriever agent \
    --model ${modelpathroot}

echo "========== Postprocessing =========="
# Postprocessing: Output the final txt file
postprocess_input=${infer_output}
postprocessed_output=$(realpath data/temp/$START_TIME/${src_filename}_postprocessed.txt)
python3 ${postprocess} \
    --input ${infer_output} \
    --output ${postprocessed_output}

cp ${postprocessed_output} data/output/${src_filename}_translated.txt
output=$(realpath data/output/${src_filename}_translated.txt)
echo "OUTPUT=$output"
