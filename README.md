# Text Analytics for Crude Oil Market Summaries

## Introduction
This repository contains a sample dataset and PyTorch code for the paper entitled **"Crude Oil Price Movement and Return Prediction via Text Analytics on Market Summaries"** (to be submitted to Energy Economics Journal).

## Requirements

## Tasks and Model Architecture:
1. Price movement prediction: UP, DOWN, NO_CHANGE(FLAT) as multiclass classification. \
We propose a vanilla BERT-based **BERTForSequenceClassification** head for this classification task. The extracted span vectors are fed into the model; BERT's **BERTForSequenceClassification** head is equipped with a \textit{sigmoid} activation function to predict one of the three classes: UP, DOWN, FLAT.

2. Return prediction (percentage of price change) as regression analysis. \
we use the BERT-based model with **BERTForSequenceClassification** head for the regression task by setting the num_class = 1. The model predicts a single scalar value as output. We use two common losses for price prediction: root mean squared error (RMSE) and mean absolute percentage error (MAPE).
 
## Comparison with other Text Processing Techniques
1. Non-event based

2. Event-based

## Repository Contents


