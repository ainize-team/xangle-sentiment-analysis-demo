import requests
import streamlit as st

from pydantic import BaseModel, Field, HttpUrl
from opyrator.components import outputs
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import normalize_whitespace, fix_bad_unicode


preprocessor = Preprocessor()
preprocessor.pipe(normalize_whitespace)
preprocessor.pipe(fix_bad_unicode)


class TextGenerationInput(BaseModel):
    model_url: HttpUrl = Field(
        '',
        title="Model URL",
        description="URL to request API.",
    )

    context: str = Field(
        title="Input Context",
        description="Context to analysis sentiment.",
    )


def process(model_url, context):
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    response = requests.post(url=model_url, headers=headers, json={"sentence" : preprocessor.run(context)})
    predictions = response.json()

    return predictions


def plot_bar_chart(predictions):
    import plotly.express as px

    predictions = [{'label': k, 'score': v} for k, v in predictions.items()]

    fig = px.bar(
        predictions,
        x="label",
        y="score",
    )

    st.plotly_chart(fig)


def xangle_sentiment_analysis(input: TextGenerationInput) -> outputs.ClassificationOutput:
    """Input Model URL on sidebar."""
    context = input.context
    model_url = input.model_url

    predictions = process(model_url, context)

    plot_bar_chart(predictions)
