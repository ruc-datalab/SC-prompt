# 1. predict SQL structure
python src/run.py \
       --run_name t5-large \
       --model_name_or_path t5-large \
       --dataset geoquery \
       --source_prefix "question: " \
       --schema_serialization_type verbose \
       --schema_serialization_randomized false \
       --schema_serialization_with_db_id true \
       --schema_serialization_with_prompt "Translate the question into sql according to the database: " \
       --schema_serialization_with_db_content true \
       --normalize_query true \
       --target_with_db_id false \
       --metric_config both \
       --output_dir experimental_outputs/geoquery/ \
       --cache_dir transformers_cache \
       --do_train false \
       --do_eval true \
       --fp16 false \
       --per_device_eval_batch_size 2 \
       --label_smoothing_factor 0.0 \
       --learning_rate 5e-5 \
       --adafactor true \
       --adam_eps 1e-6 \
       --lr_scheduler_type constant \
       --warmup_ratio 0.0 \
       --warmup_steps 0 \
       --seed 1 \
       --logging_strategy steps \
       --logging_steps 4 \
       --metric_for_best_model exact_match \
       --greater_is_better true \
       --save_strategy steps \
       --evaluation_strategy steps \
       --predict_with_generate true \
       --num_beams 8 \
       --num_beam_groups 1 \
       --diversity_penalty 0.0 \
       --max_val_samples 182 \
       --use_constrained_decoding false \
       --use_decomposition true \
       --overwrite_output_dir true \
       --stage structure \
       --training_method PFT \
       --overwrite_cache true \
       --train_samples_ratio $1
       
# 2. predict SQL content
python src/run.py \
       --run_name t5-large \
       --model_name_or_path t5-large \
       --dataset geoquery \
       --source_prefix "question: " \
       --schema_serialization_type verbose \
       --schema_serialization_randomized false \
       --schema_serialization_with_db_id true \
       --schema_serialization_with_prompt "Translate the question into sql according to the database: " \
       --schema_serialization_with_db_content true \
       --normalize_query true \
       --target_with_db_id false \
       --metric_config both \
       --output_dir experimental_outputs/geoquery/ \
       --cache_dir transformers_cache \
       --do_train false \
       --do_eval true \
       --fp16 false \
       --per_device_eval_batch_size 2 \
       --label_smoothing_factor 0.0 \
       --learning_rate 5e-5 \
       --adafactor true \
       --adam_eps 1e-6 \
       --lr_scheduler_type constant \
       --warmup_ratio 0.0 \
       --warmup_steps 0 \
       --seed 1 \
       --logging_strategy steps \
       --logging_steps 4 \
       --metric_for_best_model exact_match \
       --greater_is_better true \
       --save_strategy steps \
       --evaluation_strategy steps \
       --predict_with_generate true \
       --num_beams 4 \
       --num_beam_groups 1 \
       --diversity_penalty 0.0 \
       --max_val_samples 182 \
       --use_constrained_decoding false \
       --use_decomposition true \
       --overwrite_output_dir true \
       --stage content \
       --training_method PFT \
       --overwrite_cache true \
       --train_samples_ratio $1