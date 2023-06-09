# 1. structure stage
# Note: To make the training process more stable, we first freeze the model to train learnable vectors.
python src/run.py \
       --run_name t5-large \
       --model_name_or_path t5-large \
       --dataset cosql \
       --source_prefix "question: " \
       --schema_serialization_type verbose \
       --schema_serialization_randomized false \
       --schema_serialization_with_db_id true \
       --schema_serialization_with_prompt "Translate the question into sql according to the database: " \
       --schema_serialization_with_db_content true \
       --normalize_query true \
       --target_with_db_id false \
       --metric_config both \
       --output_dir experimental_outputs/cosql/ \
       --cache_dir transformers_cache \
       --do_train true \
       --do_eval false \
       --fp16 false \
       --num_train_epochs 100 \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 4 \
       --gradient_accumulation_steps 16 \
       --label_smoothing_factor 0.0 \
       --learning_rate 0.1 \
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
       --num_beams 1 \
       --num_beam_groups 1 \
       --use_constrained_decoding false \
       --use_decomposition true \
       --overwrite_output_dir true \
       --stage structure \
       --training_method PT \
       --overwrite_cache true \
       --train_samples_ratio $1

python src/run.py \
       --run_name t5-large \
       --model_name_or_path t5-large \
       --dataset cosql \
       --source_prefix "question: " \
       --schema_serialization_type verbose \
       --schema_serialization_randomized false \
       --schema_serialization_with_db_id true \
       --schema_serialization_with_prompt "Translate the question into sql according to the database: " \
       --schema_serialization_with_db_content true \
       --normalize_query true \
       --target_with_db_id false \
       --metric_config both \
       --output_dir experimental_outputs/cosql/ \
       --cache_dir transformers_cache \
       --do_train true \
       --do_eval false \
       --fp16 false \
       --num_train_epochs 150 \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 2 \
       --gradient_accumulation_steps 16 \
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
       --num_beams 1 \
       --num_beam_groups 1 \
       --use_constrained_decoding false \
       --use_decomposition true \
       --overwrite_output_dir true \
       --stage structure \
       --training_method PFT \
       --overwrite_cache true \
       --train_samples_ratio $1

# content stage
python src/run.py \
       --run_name t5-large \
       --model_name_or_path t5-large \
       --dataset cosql \
       --source_prefix "question: " \
       --schema_serialization_type verbose \
       --schema_serialization_randomized false \
       --schema_serialization_with_db_id true \
       --schema_serialization_with_prompt "Translate the question into sql according to the database: " \
       --schema_serialization_with_db_content true \
       --normalize_query true \
       --target_with_db_id false \
       --metric_config both \
       --output_dir experimental_outputs/cosql/ \
       --cache_dir transformers_cache \
       --do_train true \
       --do_eval false \
       --fp16 false \
       --num_train_epochs 900 \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 4 \
       --gradient_accumulation_steps 16 \
       --label_smoothing_factor 0.0 \
       --learning_rate 0.1 \
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
       --num_beams 1 \
       --num_beam_groups 1 \
       --use_constrained_decoding false \
       --use_decomposition true \
       --overwrite_output_dir true \
       --stage content \
       --training_method PT \
       --overwrite_cache true \
       --train_samples_ratio $1

python src/run.py \
       --run_name t5-large \
       --model_name_or_path t5-large \
       --dataset cosql \
       --source_prefix "question: " \
       --schema_serialization_type verbose \
       --schema_serialization_randomized false \
       --schema_serialization_with_db_id true \
       --schema_serialization_with_prompt "Translate the question into sql according to the database: " \
       --schema_serialization_with_db_content true \
       --normalize_query true \
       --target_with_db_id false \
       --metric_config both \
       --output_dir experimental_outputs/cosql/ \
       --cache_dir transformers_cache \
       --do_train true \
       --do_eval false \
       --fp16 false \
       --num_train_epochs 100 \
       --per_device_train_batch_size 2 \
       --per_device_eval_batch_size 4 \
       --gradient_accumulation_steps 16 \
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
       --num_beams 1 \
       --num_beam_groups 1 \
       --use_constrained_decoding false \
       --use_decomposition true \
       --overwrite_output_dir true \
       --stage content \
       --training_method PFT \
       --overwrite_cache true \
       --train_samples_ratio $1

