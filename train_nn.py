"""Deep learning training script adapted from notebooks/eda_and_training.ipynb."""

import json
import joblib
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("EEG-Eye-State.csv")
# Saving as .keras is the modern standard for Keras 3 / TF 2.x
MODEL_PATH = Path("models/model_dl.keras") 
META_PATH = Path("models/metadata_dl.json")
SCALER_PATH = Path("models/scaler_dl.pkl")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["eyeDetection"])
    y = df["eyeDetection"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(X_train, X_val, X_test):
    """Scales data using StandardScaler, required for Neural Networks."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_model(hp):
    """Defines the model architecture for Keras Tuner."""
    model = Sequential()
    
    # Input layer matches the number of features (14 for EEG data)
    model.add(Dense(
        units=hp.Int("units_input", min_value=32, max_value=256, step=32),
        activation="relu",
    ))
    
    # Tuning number of hidden layers
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(Dense(
            units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
            activation="relu"
        ))
        # Optional dropout for regularization
        if hp.Boolean("dropout"):
            model.add(Dropout(rate=0.25))

    # Output layer for binary classification
    model.add(Dense(1, activation="sigmoid"))

    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def tune_deep_learning(X_train, y_train, X_val, y_val):
    """Runs Keras Tuner to find best hyperparameters."""
    tuner = kt.Hyperband(
        build_model,
        objective="val_accuracy",
        max_epochs=50,
        factor=3,
        directory="models/kt_dir",
        project_name="eeg_eye_state"
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    print("Starting Keras Tuner hyperparameter search...")
    tuner.search(
        X_train, y_train, 
        epochs=50, 
        validation_data=(X_val, y_val), 
        callbacks=[early_stopping],
        verbose=1
    )
    
    return tuner


def final_evaluation(model, X_test, y_test):
    # Predict probabilities
    proba = model.predict(X_test).flatten()
    # Convert probabilities to binary predictions
    pred = (proba > 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "classification_report": classification_report(y_test, pred, digits=4),
    }
    return metrics


def save_artifacts(model, feature_names, best_params, scaler):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save Keras model
    model.save(MODEL_PATH)
    
    # Save Scaler
    joblib.dump(scaler, SCALER_PATH)
    
    meta = {
        "features": list(feature_names),
        "target": "eyeDetection",
        "best_params": best_params,
        "framework": "tensorflow"
    }
    with META_PATH.open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    
    # 1. Split Data
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)

    # 2. Preprocess (Scale) Data - Critical for Deep Learning
    X_train, X_val, X_test, scaler = preprocess_data(X_train_raw, X_val_raw, X_test_raw)

    # 3. Tune Model
    tuner = tune_deep_learning(X_train, y_train, X_val, y_val)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nBest params:")
    print(best_hps.values)

    # 4. Retrain Best Model
    # Build the model with the best hp
    best_model = tuner.hypermodel.build(best_hps)
    
    # Combine train and val for final training (optional, but standard practice)
    # Note: For DL, we often keep validation split to monitor overfitting, 
    # but here we follow the pattern of retraining on more data.
    X_full = np.concatenate((X_train, X_val))
    y_full = pd.concat([y_train, y_val])
    
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    
    print("\nRetraining best model on full training set...")
    best_model.fit(
        X_full, y_full,
        epochs=50,
        callbacks=[early_stopping],
        verbose=1
    )

    # 5. Evaluate
    metrics = final_evaluation(best_model, X_test, y_test)
    print("\nFinal metrics:")
    print("Accuracy:", metrics["accuracy"])
    print("ROC AUC:", metrics["roc_auc"])
    print(metrics["classification_report"])

    # 6. Save
    save_artifacts(best_model, X_train_raw.columns, best_hps.values, scaler)
    print(f"Saved model to {MODEL_PATH}")
