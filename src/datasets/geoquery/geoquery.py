# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Geoquery: Learning to Parse Database Queries Using Inductive Logic Programming"""

import json
import os
from typing import List, Generator, Any, Dict, Tuple
from third_party.spider.preprocess.get_tables import dump_db_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{data-geography-original
  dataset   = {Geography, original},
  author    = {John M. Zelle and Raymond J. Mooney},
  title     = {Learning to Parse Database Queries Using Inductive Logic Programming},
  booktitle = {Proceedings of the Thirteenth National Conference on Artificial Intelligence - Volume 2},
  year      = {1996},
  pages     = {1050--1055},
  location  = {Portland, Oregon},
  url       = {http://dl.acm.org/citation.cfm?id=1864519.1864543},
}
"""

_DESCRIPTION = """\
Geoquery contains 880 queries and a database of U.S. geography.
"""

_HOMEPAGE = ""

_LICENSE = "CC BY-SA 4.0"

_URL = "geoquery.zip"

def normalize_alias(
    sql: str,
    table_names: List[str],
) -> str:
    alias_format = 'T{count}'
    count = 1
    for tab in table_names+['DERIVED_FIELD', 'DERIVED_TABLE']:
        tab = tab.upper()
        for idx in ['0','1','2','3','4','5','6']:
            old_alias = tab+'alias'+idx
            if old_alias in sql:
                new_alias = alias_format.format(count=count)
                sql = sql.replace(old_alias, new_alias)
                count += 1
    return sql
    
class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="Geoquery",
            version=VERSION,
            description="880 queries and a database of U.S. geography",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()
        self.include_train_others: bool = kwargs.pop("include_train_others", False)

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "query_toks": datasets.features.Sequence(datasets.Value("string")),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_filepath = dl_manager.download_and_extract(url_or_urls=_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "geoquery/train.json")],
                    "db_path": os.path.join(downloaded_filepath, "geoquery/database"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": [os.path.join(downloaded_filepath, "geoquery/test.json")],
                    "db_path": os.path.join(downloaded_filepath, "geoquery/database"),
                },
            ),
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        print(f'db_path={db_path}')
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                geoquery = json.load(f)
                for idx, sample in enumerate(geoquery):
                    db_id = 'geo'
                    if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_db_json_schema(
                            db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                        )
                    schema = self.schema_cache[db_id]
                    sample['sql'] = normalize_alias(sample["sql"], schema["table_names_original"])
                    yield idx, {
                        "query": sample['sql'],
                        "query_toks": sample["sql"].split(),
                        "question": sample["query"],
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_table_names": schema["table_names_original"],
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in schema["column_names_original"]
                        ],
                        "db_column_types": schema["column_types"],
                        "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                        "db_foreign_keys": [
                            {"column_id": column_id, "other_column_id": other_column_id}
                            for column_id, other_column_id in schema["foreign_keys"]
                        ],
                    }

