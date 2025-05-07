from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np

def evaluar_modelo_guardado(ruta_modelo, test_dataset):
    """
    Evalúa un modelo previamente guardado con métricas: Accuracy, F1, Precision y Recall.

    Parámetros:
    - ruta_modelo: Ruta del modelo/tokenizer guardado (en Google Drive o local).
    - test_dataset: Dataset de prueba ya tokenizado (Hugging Face Dataset).

    Return:
    - Diccionario con métricas de evaluación.
    """
    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)

    # Cargar métricas
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average='weighted')["f1"],
            "precision": precision.compute(predictions=preds, references=labels, average='weighted')["precision"],
            "recall": recall.compute(predictions=preds, references=labels, average='weighted')["recall"]
        }

    # Configuración del Trainer para solo evaluación
    training_args = TrainingArguments(
        output_dir="./eval_results",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        logging_dir="./eval_logs",
        report_to="none"
    )

    # Crear Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )

    # Acceder a 'labels' desde test_dataset
    predictions = trainer.predict(test_dataset)
    labels = test_dataset['label']

    # Pasar 'labels' directamente a compute_metrics
    metricas = compute_metrics((predictions.predictions, labels))

    # Imprimir métricas
    print("\n Métricas del modelo evaluado:")
    for clave, valor in metricas.items():
        print(f"{clave}: {valor:.4f}")

    return metricas
