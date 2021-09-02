import requests
import streamlit as st

from pydantic import BaseModel, Field
from opyrator.components import outputs
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import normalize_whitespace, fix_bad_unicode
from typing import Dict


preprocessor = Preprocessor()
preprocessor.pipe(normalize_whitespace)
preprocessor.pipe(fix_bad_unicode)


class TextGenerationInput(BaseModel):
    context: str = Field(
        title="Input Context",
        description="Context to analysis sentiment.",
    )


def process(context: str) -> Dict:
    query_param = st.experimental_get_query_params()
    model_url = query_param['api'][0]
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.post(url=model_url, headers=headers, json={"sentence" : preprocessor.run(context)})
    predictions = response.json()

    return predictions


def xangle_sentiment_analysis(input: TextGenerationInput) -> outputs.ClassificationOutput:
    context = input.context

    predictions = process(context)

    return outputs.ClassificationOutput(
        __root__=[
            outputs.ScoredLabel(label=label, score=score)
            for label, score in predictions.items()
        ]
    )

