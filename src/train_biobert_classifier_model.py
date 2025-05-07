from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
import os
import json

def train_biobert_classifier_modelo_3(tok_train_dataset, tok_eval_dataset, num_labels=2, output_dir="./results_biobert_model_3", model_checkpoint="dmis-lab/biobert-base-cased-v1.1", seed=42):
    """
    Entrena un modelo BioBERT para una tarea de clasificación de texto.

    Parámetros:
    - tokenized_dataset: DatasetDict ya tokenizado.
    - num_labels: número de clases (por ejemplo, 2 para clasificación binaria).
    - output_dir: ruta donde se guardará el modelo.
    - model_checkpoint: checkpoint del modelo base.
    - seed: semilla aleatoria para reproducibilidad.

    Return:
    - trainer: objeto Trainer con el modelo entrenado.
    """

    # Carga el modelo BioBERT adaptado a clasificación
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir="./results_biobert_model_3",    # Carpeta de salida donde se guardan los resultados
        eval_strategy="steps",                     # Evaluación cada cierto número de pasos
        eval_steps=500,                            # Frecuencia de evaluación
        logging_steps=100,                         # Frecuencia de logs
        save_steps=1000,                           # Guardar checkpoints cada 1000 pasos
        per_device_train_batch_size=32,            # Tamaño de batch durante el entrenamiento
        per_device_eval_batch_size=64,             # Tamaño de batch durante la evaluación
        num_train_epochs=3,                        # Número de épocas de entrenamiento. Se quedá en 3
        learning_rate=2e-5,                        # Learning rate recomendado para BERT
        weight_decay=0.01,                         # Regularización L2
        warmup_steps=500,                          # Calentamiento del scheduler
        logging_dir="./logs",                      # Directorio para logs de TensorBoard u otros
        save_total_limit=3,                        # Cuántos checkpoints mantener en la carpeta de salida
        load_best_model_at_end=True,               # Cargar el mejor modelo al final
        metric_for_best_model="eval_loss",         # Métrica usada para determinar el mejor modelo, ya que se necesita que generalice mejor
        seed=seed,                                   # Semilla para reproducibilidad
        save_strategy="steps",                     # Guardar checkpoints regularmente durante el entrenamiento
        report_to="none",                          # Deshabilitar reporting si no usas TensorBoard o plataformas
        fp16=True,                                 # Usar mixed precision si la GPU lo soporta
        auto_find_batch_size=True,                 # Detecta automáticamente el mejor batch size en caso de problemas de memoria
        greater_is_better=False,                   # porque menor pérdida es mejor
        lr_scheduler_type="linear",                # Usar un scheduler lineal

    )

    # Carga las métricas de evaluación desde evaluate
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(eval_pred):
        """
        Calcula métricas de evaluación: accuracy, F1 (ponderado), precisión y recall.

        Parámetro:
        - eval_pred: tupla (logits, etiquetas verdaderas)

        Retorna:
        - Diccionario con métricas.
        """
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)

        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average='weighted')["f1"],
            "precision": precision.compute(predictions=preds, references=labels, average='weighted')["precision"],
            "recall": recall.compute(predictions=preds, references=labels, average='weighted')["recall"]
        }

    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train_dataset,
        eval_dataset=tok_eval_dataset,
        tokenizer=AutoTokenizer.from_pretrained(model_checkpoint),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Si no mejora en 3 evaluaciones va a interrumpoir el entrenamiento
    )

    # Entrenar el modelo
    trainer.train()

    # Guardar el mejor modelo al final del entrenamiento
    trainer.save_model("./saved_model_3")  # Guarda el modelo completo en la carpeta especificada
    trainer.save_state()  # Guarda el estado del entrenamiento (incluye configuraciones y más)

    metrics_path = os.path.join(output_dir, f"eval_metrics_seed_{seed}.json")

    # Evaluar y guardar métricas
    metrics = trainer.evaluate()
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


    return trainer
