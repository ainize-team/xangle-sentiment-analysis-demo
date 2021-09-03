import requests
import streamlit as st

from pydantic import BaseModel, Field
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import normalize_whitespace, fix_bad_unicode


preprocessor = Preprocessor()
preprocessor.pipe(normalize_whitespace)
preprocessor.pipe(fix_bad_unicode)


class TextGenerationInput(BaseModel):
    context: str = Field(
        title="Input Context",
        description="Context to analysis sentiment.",
    )


class SentimentAnalysisOutput(BaseModel):
    output: str


def process(context: str) -> str:
    query_param = st.experimental_get_query_params()
    if 'api' in query_param:
        api_url = query_param['api'][0]
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        try:
            response = requests.post(url=api_url, headers=headers, json={"sentence": preprocessor.run(context)})
            results = response.json()
            
        except:
            results = 'Endpoint API Internal error occurs.'
    else:
        results = 'There is no endpoint API in Query String.'

    return results


def xangle_sentiment_analysis(input: TextGenerationInput) -> SentimentAnalysisOutput:
    context = input.context

    results = process(context)

    return SentimentAnalysisOutput(output=results)
