import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os # <-- IMPORTACIÓN NECESARIA

def generate_predictions(model, test_loader, config, species_cols):
    """
    Genera predicciones (probabilidades) para el conjunto de prueba
    y las formatea en un DataFrame de submission.
    """
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    model.eval() 
    
    all_probabilities = []
    all_ids = []
    
    print("\nComenzando la generación de predicciones en el set de prueba...")
    
    with torch.no_grad():
        for inputs, ids in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            
            all_probabilities.append(probabilities)
            all_ids.extend(ids)
            
    # Concatenar todos los resultados
    final_probabilities = np.concatenate(all_probabilities, axis=0)
    
    # 2. Cargar el formato de submission para asegurar el orden de las columnas
    submission_format = pd.read_csv(config['paths']['submission_format'])

    # Crear el DataFrame de resultados
    results_df = pd.DataFrame(final_probabilities, columns=species_cols)
    results_df.insert(0, 'id', all_ids) 

    # 3. Formatear y guardar
    submission_df = submission_format[['id']].merge(results_df, on='id', how='left')
    submission_df = submission_df.fillna(0) 

    submission_output_path = os.path.join(os.path.dirname(config['paths']['model_output']), 'submission.csv')
    submission_df.to_csv(submission_output_path, index=False)
    
    print(f"Archivo de submission generado y guardado en: {submission_output_path}")
    
    return submission_df