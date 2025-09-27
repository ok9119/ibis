# -*- coding: utf-8 -*-

import pandas as pd
import pysam
from pathlib import Path
import joblib

from ibis_utils import complement, slice_and_dice

def process_experiment_data(experiment_type, template_path, fasta_path, output_dir):
    """
    Get predictions for PBM, HTS or SMS experiments

    Parameters:
    -----------
    experiment_type : 'PBM', 'HTS' или 'SMS'
    template_path : str
    fasta_path (test sequences): str
    output_dir : str
    """

    template = pd.read_csv(template_path, sep='\t')

    # Fetch test sequences
    test_sequences = pysam.FastaFile(fasta_path)
    id_column = 'tag'
    template['seq'] = template[id_column].apply(lambda x: test_sequences.fetch(reference=x))

    # Remove adapter sequences
    if experiment_type == 'PBM':
        template['seq'] = template['seq'].str.replace('CCTGTGTGAAATTGTTATCCGCTCT', '')
    elif experiment_type == 'HTS':
        pass
    elif experiment_type == 'SMS':
        pass
    else:
      raise ValueError("Wrong type of experiment")

    # Tokenize sequences
    template['seq'] = template['seq'].apply(slice_and_dice)

    # Predict
    X_test = template['seq']
    predictions = pd.DataFrame(clf_fin.predict_proba(X_test.astype('str')),
                              columns=clf_fin.classes_)

    target_columns = {
    'HTS': ['USF3', 'ZBED2', 'MYF6', 'SALL3', 'CAMTA1', 'ZNF367', 'ZNF648', 'ZNF518B', 'ZBED5', 'ZNF251', 'ZNF493', 'ZNF20', 'LEUTX', 'PRDM13', 'ZNF395'],
    'SMS': ['USF3', 'ZBED2', 'CAMTA1', 'ZNF367', 'ZNF648', 'ZBED5', 'ZNF251', 'ZNF493', 'PRDM13', 'ZNF395'],
    'PBM': ['USF3', 'ZBED2', 'MYF6', 'ZBED5', 'LEUTX']}.get(experiment_type)

    for col_name in target_columns:
      template[col_name] = predictions[col_name]


    # Save the results
    output_columns = [id_column] + target_columns
    output_filename = f'{experiment_type}_predictions.tsv'
    output_path = Path(output_dir) / output_filename

    template[output_columns].to_csv(output_path, sep='\t', float_format='%.5f', index=False)

    return template[output_columns]


def process_all_experiments(model, base_dir):

    experiments = {
        'PBM': {
            'template': f'{base_dir}/submission_template_example/PBM_aaa_template.tsv',
            'fasta': f'{base_dir}/PBM_participants.fasta'
        },
        'HTS': {
            'template': f'{base_dir}/submission_template_example/HTS_aaa_template.tsv',
            'fasta': f'{base_dir}/HTS_participants.fasta'
        },
        'SMS': {
            'template': f'{base_dir}/submission_template_example/SMS_aaa_template.tsv',
            'fasta': f'{base_dir}/SMS_participants.fasta'
        }
    }

    results = {}
    for exp_type, data in experiments.items():
        try:
            results[exp_type] = process_experiment_data(
                experiment_type=exp_type,
                template_path=data['template'],
                fasta_path=data['fasta'],
                output_dir=f'{base_dir}/predictions/'
            )
        except Exception as e:
            print(f"An error occured while processing {exp_type} data: {e}")

    return results

# Load model and run:
if __name__ == "__main__":
    filename = './models/g2a_tfidf_lr_knn_cb.joblib.pkl'
    clf_fin = joblib.load(filename)
    base_directory = './data/ibis_test'
    process_all_experiments(clf_fin, base_directory)

