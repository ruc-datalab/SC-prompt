from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers.training_args import TrainingArguments
from .bridge_content_encoder import get_database_matches
import re
import random
import json
from .args import *

@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    schemas: Dict[str, dict]


def _get_schemas(examples: Dataset) -> Dict[str, dict]:
    schemas: Dict[str, dict] = dict()
    for ex in examples:
        if ex["db_id"] not in schemas:
            schemas[ex["db_id"]] = {
                "db_table_names": ex["db_table_names"],
                "db_column_names": ex["db_column_names"],
                "db_column_types": ex["db_column_types"],
                "db_primary_keys": ex["db_primary_keys"],
                "db_foreign_keys": ex["db_foreign_keys"],
            }
    return schemas

def _prepare_train_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    data_args: DataArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:

    if data_args.dataset in ['']:
        schemas = _get_schemas_geoquery(examples=dataset)
    else:
        schemas = _get_schemas(examples=dataset)
    dataset = dataset.map(
        lambda ex: add_serialized_schema(
            ex=ex,
            mode='train'),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=True,
    )
    if data_training_args.train_samples_ratio != 1.0:
        if data_args.dataset in ['geoquery']:
            if data_training_args.train_samples_ratio == 0.094:
                indexs = [19,514,375,341,274,492,221,515,360,58,413,418,333,487,28,122,344,208,475,108,264,155,0,23,4,73,129,27,61,8,74,469,500,396,362,430,17,203,171,33,139,80,503,206,243,5,486,423,244,130]
            else:
                indexs = random.sample(range(536), int(dataset.num_rows*data_training_args.train_samples_ratio))
            print(indexs)
            print(f"use {len(set(indexs))} training samples.")
            dataset = dataset.select(indexs)
        else:
            dataset = dataset.select(range(int(dataset.num_rows*data_training_args.train_samples_ratio)))
    column_names = dataset.column_names
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
            mode='train',
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return TrainSplit(dataset=dataset, schemas=schemas)


def _prepare_eval_split(
    dataset: Dataset,
    data_training_args: DataTrainingArguments,
    data_args: DataArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:

    eval_examples = dataset
    if data_args.dataset in ['']:
        schemas = _get_schemas_geoquery(examples=dataset)
    else:
        schemas = _get_schemas(examples=eval_examples)
    eval_dataset = eval_examples.map(
        lambda ex: add_serialized_schema(
            ex=ex,
            mode='eval'),
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=False,
    )
    if data_training_args.max_val_samples is not None:
        eval_dataset = eval_dataset.select(range(data_training_args.max_val_samples))
    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
            mode='eval',
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return EvalSplit(dataset=eval_dataset, examples=eval_examples, schemas=schemas)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None
    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_training_args=data_training_args,
            data_args=data_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )
    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_args=data_args,
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )
    if training_args.do_predict:
        eval_split = _prepare_eval_split(
            dataset_dict["test"],
            data_args=data_args,
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=pre_process_function,
        )
    schemas = {
        **(train_split.schemas if train_split is not None else {}),
        **(eval_split.schemas if eval_split is not None else {}),
        **(test_split_schemas if test_splits is not None else {}),
    }

    return DatasetSplits(
        train_split=train_split, 
        eval_split=eval_split, 
        test_splits=test_splits, 
        schemas=schemas
    )


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))


def serialize_schema(
    question: str,
    db_path: str,
    db_id: str,
    db_column_names: Dict[str, str],
    db_table_names: List[str],
    schema_serialization_type: str = "peteshaw",
    schema_serialization_randomized: bool = False,
    schema_serialization_with_db_id: bool = True,
    schema_serialization_with_db_content: bool = False,
    normalize_query: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "database: {db_id}. "
        table_sep = ". "
        table_str = "table: {table}. columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError
    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                string = column_str_with_values.format(column=column_name_str, values=value_sep.join(matches))
                return string
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    
    reorder_tables = []
    for table in tables:
        if '(' in table:
            reorder_tables = [table] + reorder_tables
        else:
            reorder_tables.append(table)
            
    tables = reorder_tables
    #if schema_serialization_randomized:
    #    random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = 'database: ' + table_sep.join(tables)

    return serialized_schema

def combine_SC(content, input, structure=None):
    if structure == None:
        structure = input.replace("Translate the question into sql according to the database:", "")
        end_index = structure.index(" | question:")
        structure = structure[:end_index]
    col_num = structure.count('[col]')
    tab_num = structure.count('[tab]')
    val_num = structure.count('[val]')
    if (content.count('[col]') != col_num) or (content.count('[tab]') != tab_num) or (content.count('[val]') != val_num):
        return structure

    content_dict = {"[col]": [], "[tab]": [], "[val]": []}
    tok = None
    temp_str = ''
    i = 0
    while i < len(content):
        if content[i] == '[' and i+4 < len(content) and content[i+4] == ']' and (content[i:i+5] in ['[col]', '[tab]', '[val]']):
            if tok != None:
                content_dict[tok].append(temp_str.strip())
            tok = content[i:i+5]
            temp_str = ''
            i += 6
            continue
        temp_str += content[i]
        i += 1
    if tok != None:
        content_dict[tok].append(temp_str.strip())

    pred_sql = structure
    # replace [col]
    end_index = 0
    for i in range(col_num):
        begin_index = pred_sql[end_index:].index('[col]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[col]'][i] + pred_sql[begin_index+5:]
        end_index = begin_index + len(content_dict['[col]'][i]) + 1
    
    # replace [tab]
    end_index = 0
    for i in range(tab_num):
        begin_index = pred_sql[end_index:].index('[tab]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[tab]'][i] + pred_sql[begin_index+5:]
        end_index = begin_index + len(content_dict['[tab]'][i]) + 1
    
    # replace [val]
    end_index = 0
    for i in range(val_num):
        begin_index = pred_sql[end_index:].index('[val]') + end_index
        pred_sql = pred_sql[:begin_index] + content_dict['[val]'][i] + pred_sql[begin_index+5:]
        end_index = begin_index + len(content_dict['[val]'][i]) + 1
    if pred_sql[0] == ' ':
        pred_sql = pred_sql[1:]
    return pred_sql

