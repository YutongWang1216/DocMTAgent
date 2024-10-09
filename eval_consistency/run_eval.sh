
lang=en-zh
src_lang=${lang%%-*}
tgt_lang=${lang##*-}

src_file=/path/to/src/file
hyp_file=/path/to/hyp/file

# tool=nltk
tool=spacy

output_dir=result/

src_lang=${lang%%-*}
tgt_lang=${lang##*-}

script_path=./

if [[ $src_lang == 'en' ]]; then
    pyfile=eval_probability_en_to.py
else
    pyfile=eval_probability_to_en.py
fi

if [ ! -d "${output_dir}" ]; then
    mkdir -p $output_dir
fi

file_name=${src_file##*/}
token_file=$output_dir/$file_name.entity.token
tag_file=$output_dir/$file_name.entity.tag

align_file=$output_dir/$file_name.entity.align
word_file=$output_dir/$file_name.entity.word
prob_file=$output_dir/$file_name.entity.prob

record_file=$output_dir/$file_name.record.json
wrong_entity_file=$output_dir/$file_name.wrong_entity.json
out_file=$output_dir/$file_name.consistency
pair_file=$output_dir/$file_name.entity_pair


if [ ! -f "${token_file}" ]; then
    if [ $src_lang = 'en' ]; then
        python3 -u $script_path/src_tgt_tokenize_en_to.py -s $src_file -t $hyp_file -l $lang -o $token_file --tag_file $tag_file --tool $tool --reverse
    else
        python3 -u $script_path/src_tgt_tokenize_to_en.py -s $src_file -t $hyp_file -l $lang -o $token_file --tag_file $tag_file --tool $tool
    fi
fi

if [ ! -f "${align_file}" ]; then
    MODEL_NAME_OR_PATH=bert-base-multilingual-cased
    CUDA_VISIBLE_DEVICES=0 awesome-align \
        --data_file=$token_file \
        --model_name_or_path=$MODEL_NAME_OR_PATH \
        --extraction 'softmax' \
        --batch_size 32 \
        --output_file=$align_file \
        --output_word_file $word_file \
        --output_prob_file $prob_file
fi

sed '/^\s*$/d' $align_file > temp; mv temp $align_file
sed '/^\s*$/d' $token_file > temp; mv temp $token_file
sed '/^\s*$/d' $tag_file > temp; mv temp $tag_file
sed '/^\s*$/d' $word_file > temp; mv temp $word_file
sed '/^\s*$/d' $prob_file > temp; mv temp $prob_file

if [ ! -f "$pair_file" ]; then
    echo $file_name >> $out_file
    if [ $src_lang = 'en' ]; then
        python3 -u $script_path/$pyfile -a $align_file -t $token_file --tag_file $tag_file -p $tool --prob_file $prob_file \
            -e $pair_file -o $out_file -r $record_file -l $lang \
            --stopwords $script_path/stopwords_${src_lang}.txt --reverse
    else
        python3 -u $script_path/$pyfile -a $align_file -t $token_file --tag_file $tag_file -p $tool --prob_file $prob_file \
            -e $pair_file -o $out_file -r $record_file -l $lang \
            --stopwords $script_path/stopwords_${src_lang}.txt
    fi
fi

python3 $script_path/show_wrong_entities.py -e $pair_file -o $wrong_entity_file
