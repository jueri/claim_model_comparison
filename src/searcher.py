# -*- coding: utf-8 -*-
"""Functions related to the passage retrieval model.

This model contains functions to construct a searchable index and search 
the index with pyserini.

Example:
    A pandas DataFrame can be converted to a json file, ready to index, 
    using this function:

        $ from searcher import convert_data, create_index, build_query
        $ 
        $ convert_data(data_sample["Sentence"], data_path=PYSERINI_PATH)


    An index can be created with:

        $ searcher = searcher = create_index(data_path=PYSERINI_PATH, index_path=INDEX_PATH, language="english")

    A query to search the index can be created using the `build_query()` 
    function:

        $ query = build_query()

"""

import json
import os
import shutil

import pandas as pd  # type: ignore
from pyserini.index.__main__ import JIndexCollection  # type: ignore
from pyserini.search import SimpleSearcher  # type: ignore
from pyserini.search import querybuilder
from typing import List


def convert_data(passages: pd.DataFrame, data_path: str):
    """Create indexable data for pyserini index in the pyserini directory.

    Args:
        segments (list[str]): List with segments to index.
    """
    data = []
    for idx, text in passages.items():
        data.append({"id": idx, "contents": text})

    try:
        os.mkdir(data_path)
    except OSError as error:
        print(error)

    with open(os.path.join(data_path, "data.json"), "w") as writer:
        json.dump(data, writer, ensure_ascii=False)


def create_index(data_path: str, index_path: str, language: str = "english"):
    """Create a pyserini index, index the data and return a searcher object.

    Args:
        data_path (str, optional): Path to the data to index. Defaults to os.path.join(".", "pyserini").
        index_path (str, optional): Path to store the index at. Defaults to os.path.join(".", "pyserini", "index").
        language (str, optional): Language of the data. Defaults to german.

    Returns:
        [type]: [description]
    """
    args = [
        "-collection",
        "JsonCollection",
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        "1",
        "-input",
        data_path,
        "-index",
        index_path,
        "-storePositions",
        "-storeDocvectors",
        "-storeRaw",
    ]

    try:
        os.mkdir(index_path)
    except OSError as error:
        print(error)
        shutil.rmtree(index_path)
        os.mkdir(index_path)

    JIndexCollection.main(args)
    searcher = SimpleSearcher(index_path)
    searcher.set_language(language)
    return searcher


def build_query(should: List[str], must: List[str]):
    """Create an anserini query with boolean conditions.

    Args:
        should (list[str]): List of terms the resulds should have.
        must (list[str]): List of terms the results have to have.

    Returns:
        Anserini query: Compiled query to search with.
    """
    condition_should = querybuilder.JBooleanClauseOccur["should"].value
    condition_must = querybuilder.JBooleanClauseOccur["must"].value

    boolean_query_builder = querybuilder.get_boolean_query_builder()

    for term in should:
        encoded_term = querybuilder.get_term_query(term)
        boolean_query_builder.add(encoded_term, condition_should)

    if must:
        for term in must:
            encoded_term = querybuilder.get_term_query(term)
            boolean_query_builder.add(encoded_term, condition_must)

    return boolean_query_builder.build()
