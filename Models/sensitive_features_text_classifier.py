from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd
from pathlib import Path

def train_sensitive_features_text_classifier():
    dataset = load_dataset("csv", data_files=str(Path("bias_identification/sensitive_features_dataset.csv")), split="train")
    dataset_dict = dataset.train_test_split(test_size=0.2, seed=0)
    model_path = "google-bert/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    id2label = {0: 'Not Sensitive', 1: 'Sensitive'}
    label2id = {'Not Sensitive': 0, 'Sensitive': 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label, label2id=label2id)

    # freeze all base model parameters
    for name, param in model.base_model.named_parameters():
        param.requires_grad = False

    # unfreeze base model pooling layers
    for name, param in model.base_model.named_parameters():
        if 'pooler' in name:
            param.requires_grad = True

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)
    
    # Apply label conversion to integers
    dataset_dict = dataset_dict.map(lambda x: {'labels': label2id[x['label']]}, remove_columns=['label'])
    
    tokenized_data = dataset_dict.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # load metrics
    accuracy = evaluate.load("accuracy")
    auc_score = evaluate.load("roc_auc")

    def compute_metrics(eval_pred):
        # get predictions
        predictions, labels = eval_pred
        
        # apply softmax to get probabilities
        probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)
        # use probabilities of the positive class for ROC AUC
        positive_class_probs = probabilities[:, 1]
        # compute auc
        auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'],3)
        
        # predict most probable class
        predicted_classes = np.argmax(predictions, axis=1)
        # compute accuracy
        acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'],3)
        
        return {"Accuracy": acc, "AUC": auc}
    
    # hyperparameters
    lr = 2e-4
    batch_size = 8
    num_epochs = 10

    training_args = TrainingArguments(
        output_dir="bert-sensitive-features-classifier",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # apply model to validation dataset
    predictions = trainer.predict(tokenized_data["test"])

    # Extract the logits and labels from the predictions object
    logits = predictions.predictions
    labels = predictions.label_ids

    # Use your compute_metrics function
    metrics = compute_metrics((logits, labels))
    print(metrics)
    


def classify_sensitive_features(features_descriptions, features):
    # Reload the trained model and tokenizer
    model_path = "bert-sensitive-features-classifier/checkpoint-830"  # This should be the path where your trained model is saved
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Define the label mappings (same as before)
    id2label = {0: 'Not Sensitive', 1: 'Sensitive'}
    label2id = {'Not Sensitive': 0, 'Sensitive': 1}
    # Function to preprocess your new text
    def preprocess_new_data(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # New data (replace with your own input texts)
    new_texts = features_descriptions
    # Preprocess the new data
    inputs = preprocess_new_data(new_texts)
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    # Get predicted classes (0 or 1)
    predicted_classes = torch.argmax(logits, dim=1)
    # Convert the predicted class ids to the corresponding label names
    predicted_labels = [id2label[label.item()] for label in predicted_classes]
    features_sensitivity = {}
    for i in range(len(features)):
        features_sensitivity[features[i]] = predicted_labels[i]
    return features_sensitivity


# train_sensitive_features_text_classifier()
# classify_sensitive_features(features_descriptions=['1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
# 'percentage of lower status of the population.'],features=['B', 'LSTAT'], )