import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .dataset import normalize, serialize_schema, combine_SC
from .args import *
from .trainer import Seq2SeqTrainer, EvalPrediction
from .process_sql import get_schema, Schema, get_sql
import os
import random
import re
import shlex

pattern_1 = re.compile('(\(|\))')
pattern_4 = re.compile('(\()')
pattern_5 = re.compile('(\))')
pattern_2 = re.compile('(,)')
pattern_3 = re.compile('(>=|<=|>|<|=)')

sql_clauses = ['select', 'from', 'where', 'group', 'having', 'order', 'limit', 'intersect', 'union', 'except']
sql_ops_space = ['>', '<', '=', 'like', '!=', '-', '+', 'between', 'and', 'or', 'not', 'in', ')', 'by', 'distinct', '>=', '<=', '<>']
sql_ops_no_space = ['count', 'avg', 'sum', 'max', 'min', '(', '!', 'desc', 'asc']
sql_marks = [',']

def lower_(
    word: str,
) -> str:
    if '"' in word or "'" in word:
        return word
    else:
        return word.lower()

def tok_process(
    toks: list,
) -> list:
    processed_tok_list = []
    i = 0
    while i < len(toks):
        if toks[i] == "``" and toks[i+2] == "''":
            temp = f'"{toks[i+1]}"'
            processed_tok_list.append(temp)
            i += 3
            continue
        else:
            processed_tok_list.append(toks[i])
            i += 1
    return [lower_(x) for x in processed_tok_list]
            


def spider_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " | " + serialized_schema.strip()


def spider_get_target(
    query: str,
    db_id: str,
    normalize_query: bool,
    target_with_db_id: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return f"{db_id} | {_normalize(query)}" if target_with_db_id else _normalize(query)


def spider_add_serialized_schema(ex: dict, mode: str, data_training_args: DataTrainingArguments) -> dict:
    serialized_schema = serialize_schema(
        question=ex["question"],
        db_path=ex["db_path"],
        db_id=ex["db_id"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    return {"serialized_schema": serialized_schema}


def spider_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    mode: Optional[str],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else "question: "
    if data_training_args.use_decomposition:
        inputs = []
        targets = []
        if data_training_args.stage == 'content':
            eval_format_list = []
            with open(data_training_args.structure_path) as f:
                info = json.load(f)
                for item in info:
                    eval_format_list.append(item['prediction'])
            print(f"load {len(eval_format_list)} eval_formats from {data_training_args.structure_path}")
        if len(batch['question']) == 1000:
            count = 0
        else:
            count = 1000

        for question, serialized_schema, db_id, query, query_toks, db_column_names in zip(batch["question"], batch["serialized_schema"], batch["db_id"], batch["query"], batch["query_toks"], batch["db_column_names"]):
            input_str = spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
            #input_str = input_str + ' | Translate '
            column_names = [x.lower() for x in db_column_names['column_name']]
            
            lex = shlex.shlex(query)
            lex.whitespace = ' '
            lex.quotes=['"', "'"]
            lex.whitespace_split = True
            query_toks = list(lex)
            query_tok_list = tok_process(query_toks)
            for idx, tok in enumerate(query_tok_list):
                if '"' in tok or "'" in tok:
                    continue
                if len(tok) > 1 and ',' in tok and tok not in column_names and query_tok_list[idx-1] not in sql_ops_space:
                    res = pattern_2.split(tok)
                    query_tok_list[idx:idx+1] = res
                if '(' in query_tok_list[idx] and ')' in query_tok_list[idx]:
                    res = pattern_1.split(query_tok_list[idx])
                    query_tok_list[idx:idx+1] = res
                elif '(' in query_tok_list[idx] and ('select' in query_tok_list[idx] or 'distinct' in query_tok_list[idx] or 'in' in query_tok_list[idx] or 'count' in query_tok_list[idx]):
                    res = pattern_4.split(query_tok_list[idx])
                    query_tok_list[idx:idx+1] = res
                elif len(query_tok_list[idx]) > 1 and ')' == query_tok_list[idx][-1]:
                    res = pattern_5.split(query_tok_list[idx])
                    query_tok_list[idx:idx+1] = res
                if ('>' in query_tok_list[idx] or '<' in query_tok_list[idx] or '>=' in query_tok_list[idx] or '<=' in query_tok_list[idx] or '=' in query_tok_list[idx]) and query_tok_list[idx][0] not in ['>', '<', '=']:
                    res = pattern_3.split(query_tok_list[idx])
                    query_tok_list[idx:idx+1] = res
            for idx, tok in enumerate(query_tok_list):
                if tok == '':
                    del query_tok_list[idx]
            
            # query_tok_list = tok_process(query_toks)
            sub_query_format_list = []
            content_label = ''
            sub_query_list = []
            select_from_record = []
            sub_query_format = ''
            sub_query = ''
            last_tok = None
            select_num = 0
            left_bracket = 0
            right_bracket = 0
            if query_tok_list[-1] == ';':
                query_tok_list = query_tok_list[:-1]
            for idx, query_tok in enumerate(query_tok_list):
                if query_tok in sql_clauses:
                    if query_tok == 'select':
                        select_num += 1
                    elif (query_tok == 'group' or query_tok == 'order') and query_tok_list[idx+1] != 'by':
                        if query_tok in column_names:
                            if idx + 1 == len(query_tok_list) or query_tok_list[idx+1] in [',', ')']:
                                sub_query_format += '[col]'
                            else:
                                sub_query_format += '[col] '
                            content_label += '[col] ' + query_tok + ' '
                            sub_query += query_tok + ' '
                        else:
                            print("error:", query_tok)
                        continue

                    if sub_query_format != '' and sub_query != '':
                        if 'select' in sub_query_format:
                            select_from_record.append(1)
                        elif 'from' in sub_query_format:
                            select_from_record.append(2)
                        else:
                            select_from_record.append(0)
                        sub_query_format_list.append(sub_query_format)
                        sub_query_list.append(sub_query)
                        sub_query_format = ''
                        sub_query = ''
                    if query_tok == 'from':
                        sub_query_format += 'from [tab]'
                        content_label += '[tab] '
                    else:
                        sub_query_format += query_tok + ' '
                    last_tok = 'sql_clauses'
                    sub_query += query_tok + ' '
                elif sub_query_format == 'from [tab]':
                    last_tok = 'from [tab]'
                    sub_query += query_tok + ' '
                    if query_tok not in [')', '(']:
                        content_label += query_tok + ' '
                    continue
                elif query_tok in sql_ops_space:
                    if query_tok == ')':
                        right_bracket += 1
                    if ((query_tok == '>' or query_tok == '<') and query_tok_list[idx+1] == '=') or (query_tok == ')' and (idx + 1 == len(query_tok_list) or query_tok_list[idx+1] == ',')):
                        # >= or <= ),
                        sub_query_format += query_tok
                        sub_query += query_tok
                    else:
                        sub_query_format += query_tok + ' '
                        sub_query += query_tok + ' '
                    last_tok = 'op'
                elif query_tok in sql_ops_no_space:
                    if query_tok == '(':
                        left_bracket += 1
                    sub_query_format += query_tok
                    sub_query += query_tok
                elif query_tok in column_names or '.' in query_tok:
                    if last_tok == 'val':
                        content_label += query_tok + ' '
                        continue
                    if idx + 1 == len(query_tok_list) or query_tok_list[idx+1] in [',', ')']:
                        sub_query_format += '[col]'
                        sub_query += query_tok
                    else:
                        sub_query_format += '[col] '
                        sub_query += query_tok + ' '
                    content_label += '[col] ' + query_tok + ' '
                    last_tok = 'col'
                elif query_tok in sql_marks:
                    sub_query_format += query_tok + ' '
                    sub_query += query_tok + ' '
                    last_tok = 'mark'                   
                else:
                    if last_tok != 'val':
                        sub_query_format += '[val] '
                        content_label += '[val] '
                    if query_tok == '``':
                        sub_query += '"'
                        content_label += '"'
                    elif query_tok == "''":
                        sub_query += '" '
                        content_label += '" '
                    elif query_tok == "'":
                        sub_query += "' "
                        content_label += "' "
                    elif last_tok == 'val' and (idx + 1 == len(query_tok_list) or query_tok_list[idx+1] in ["'", '"', '``', "''"]):
                        sub_query += query_tok
                        content_label += query_tok
                    else:
                        sub_query += query_tok + ' '
                        content_label += query_tok + ' '
                    last_tok = 'val'

            if select_num > 1 and left_bracket > right_bracket:
                sub_query_format += ')'
            sub_query_format_list.append(sub_query_format)
            sub_query_list.append(sub_query)
            if data_training_args.stage == 'structure':
                structure = normalize(' '.join(sub_query_format_list))
                inputs.append(data_training_args.schema_serialization_with_prompt + ' | ' + input_str)
                target = spider_get_target(
                    query=structure,
                    db_id=db_id,
                    normalize_query=True,
                    target_with_db_id=False,
                )
                targets.append(target)

            elif data_training_args.stage == 'content':
                if mode == 'eval':
                    input_str = data_training_args.schema_serialization_with_prompt + eval_format_list[count] + ' | ' + input_str
                else:
                    input_str = data_training_args.schema_serialization_with_prompt + ' '.join(sub_query_format_list) + ' | ' + input_str
                inputs.append(input_str)
                target = content_label
                targets.append(target)
            count += 1

    else:
        inputs = [
            spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
            for question, serialized_schema in zip(batch["question"], batch["serialized_schema"])
        ]
        targets = [
            spider_get_target(
                query=query,
                db_id=db_id,
                normalize_query=data_training_args.normalize_query,
                target_with_db_id=data_training_args.target_with_db_id,
            )
            for db_id, query in zip(batch["db_id"], batch["query"])
        ]
    print(f"{mode}: {len(inputs)}")

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class SpiderTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        metas = [
            {
                "query": x["query"],
                "question": x["question"],
                "context": context,
                "label": label,
                "db_id": x["db_id"],
                "db_path": x["db_path"],
                "db_table_names": x["db_table_names"],
                "db_column_names": x["db_column_names"],
                "db_foreign_keys": x["db_foreign_keys"],
            }
            for x, context, label in zip(examples, inputs, decoded_label_ids)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        assert len(metas) == len(predictions)
        if self.stage == 'content':
            final_pred_sqls = []
            hypotheses_path = os.path.join(self.args.output_dir, "hypotheses.json")
            if os.path.exists(hypotheses_path):
                # sentence-level check
                with open(hypotheses_path) as f:
                    hypotheses = json.load(f)
                    for idx, item in enumerate(hypotheses):
                        db_id, structure = item["structure"].split(" | ")
                        db = os.path.join(metas[idx]["db_path"], db_id, f'{db_id}.sqlite')
                        schema = Schema(get_schema(db))
                        final_pred_sql = None
                        for hypothesis in item["topk_preds"]:
                            try:
                                pred_sql = combine_SC(content=hypothesis, input="", structure=structure)
                                parse_sql = get_sql(schema, pred_sql)
                                final_pred_sql = pred_sql
                                break                           
                            except:
                                continue
                        if final_pred_sql == None:
                            # default to the first one
                            final_pred_sql = combine_SC(content=item["topk_preds"][0], input="", structure=structure)
                        final_pred_sqls.append(final_pred_sql)
                            
                os.remove(hypotheses_path)
            else:
                for pred_content, meta in zip(predictions, metas):
                    final_pred_sqls.append(combine_SC(pred_content, meta['context'])) 
            # write predict sql
            with open(f"{self.args.output_dir}/predict_sql.txt", "w") as f:
                for final_pred_sql in final_pred_sqls:
                    f.write(final_pred_sql+"\n")

            with open(f"{self.args.output_dir}/content_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"input": meta['context']}, **{"prediction": prediction}, **{"label": label}, **{"score": prediction==label}, **{"pred_sql": final_pred_sql}, **{"gold_sql": meta['query']}) for meta, prediction, final_pred_sql, label in zip(metas, predictions, final_pred_sqls, decoded_label_ids)],
                    f,
                    indent=4,
                )
            return EvalPrediction(predictions=final_pred_sqls, label_ids=decoded_label_ids, metas=metas)
        elif self.stage == 'structure':        
            for idx in range(len(predictions)):
                if 'before' in predictions[idx]:
                    predictions[idx] = predictions[idx].replace('before', '<')
                if 'after' in predictions[idx]:
                    predictions[idx] = predictions[idx].replace('after', '>')
            return EvalPrediction(predictions=predictions, label_ids=decoded_label_ids, metas=metas)
        

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        #predictions, label_ids, metas = eval_prediction
        predictions, label_ids, metas = eval_prediction
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        references = metas
        if self.stage == 'structure':
            accuracy = []
            accuracy.extend(
                (
                    pred.lower() == actual.lower()
                    for pred, actual in zip(predictions, label_ids)
                )
            )
            eval_metric = np.mean(accuracy)
            test_suite = dict()
            if eval_metric >= self.best_acc:
                with open(f"{self.args.output_dir}/structure.json", "w") as f:
                    json.dump(
                        [dict(**{"input": meta['context']}, **{"prediction": prediction}, **{"label": label}, **{"score": prediction==label}) for meta, prediction, label in zip(metas, predictions, label_ids)],
                        f,
                        indent=4,
                    )
            return {**{"exact_match": eval_metric}, **test_suite}
        elif self.stage == 'content':
            return self.metric.compute(predictions=predictions, references=references)
        else:
            raise NotImplementedError()
