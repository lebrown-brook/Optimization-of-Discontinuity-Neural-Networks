import tensorflow as tf
import numpy as np
from nnlayers import DiscontinuityDense
import os

# Pfad zum Modell
model_path = 'deltaNN/acetone_example/modelB835_2022.h5'

print(f"Lade Modell von: {model_path}")

# Custom Objects für das Laden
custom_objects = {
    'DiscontinuityDense': DiscontinuityDense,
    'Zeros': tf.keras.initializers.Zeros
}

try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("Modell erfolgreich geladen.\n")

    print("-" * 60)
    print(f"{'Layer Name':<25} | {'Type':<20} | {'Input Tensors (Type & Content)'}")
    print("-" * 60)

    for layer in model.layers:
        # Zugriff auf inbound nodes (interne Keras API, die in nnanalyzers verwendet wird)
        if hasattr(layer, '_inbound_nodes') and len(layer._inbound_nodes) > 0:
            node = layer._inbound_nodes[0]

            # input_tensors abrufen
            if hasattr(node, 'input_tensors'):
                input_t = node.input_tensors
            elif hasattr(node, 'inputs'): # Neuere Keras Versionen nutzen manchmal .inputs
                 input_t = node.inputs
            else:
                input_t = "N/A"

            # Analyse des Inhalts
            input_info = ""
            if isinstance(input_t, list):
                input_info = f"LIST (len={len(input_t)}): "
                for t in input_t:
                    input_info += f"[{t.name}, shape={t.shape}] "
            else:
                input_info = f"SINGLE OBJECT: {type(input_t)}"

            print(f"{layer.name:<25} | {layer.__class__.__name__:<20} | {input_info}")
        else:
             print(f"{layer.name:<25} | {layer.__class__.__name__:<20} | No inbound nodes (InputLayer?)")

    print("-" * 60)
    print("\nERKLÄRUNG:")
    print("Wenn 'Input Tensors' eine LISTE ist (z.B. len=1), dann ist das der Grund für den Fehler.")
    print("Der Layer erwartet oft den Tensor direkt, nicht die Liste, die ihn enthält.")

except Exception as e:
    print(f"Fehler beim Laden oder Inspizieren: {e}")

