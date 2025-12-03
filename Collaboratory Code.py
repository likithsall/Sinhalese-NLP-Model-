# Install necessary libraries
!pip install transformers datasets
!pip install huggingface_hub

# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

# Log in to your Hugging Face account
# You can find your access token at https://huggingface.co/settings/tokens
login(token="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")

# Load the Sinhala dataset from Hugging Face
dataset = load_dataset("oscar-corpus/OSCAR-2301","si_meta/si_meta_part_1.jsonl.zst")

# Load a pre-trained tokenizer and model for Sinhala
model_name = "ai4bharat/indic-bert"  # You can choose another model if preferred
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Modify num_labels based on your task

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test sets
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("sinhala_model")
