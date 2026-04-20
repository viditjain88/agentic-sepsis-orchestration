import os
import shutil
import pandas as pd
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.config import Config
from medcat.preprocessors.cleaners import NameDescriptor

SEPSIS_CONCEPTS = [
    {"cui": "C0243026", "name": "Sepsis", "type": "Disease or Syndrome", "semantic_type": "T047", "synonyms": ["Sepsis", "Septicemia", "septic"]},
    {"cui": "C1090680", "name": "Severe Sepsis", "type": "Disease or Syndrome", "semantic_type": "T047", "synonyms": ["Severe Sepsis", "severe sepsis"]},
    {"cui": "C0151744", "name": "Septic Shock", "type": "Disease or Syndrome", "semantic_type": "T047", "synonyms": ["Septic Shock", "septic shock"]},
    {"cui": "C0039082", "name": "Systemic Inflammatory Response Syndrome", "type": "Disease or Syndrome", "semantic_type": "T047", "synonyms": ["Systemic Inflammatory Response Syndrome", "SIRS", "sirs"]},
    {"cui": "C0020649", "name": "Hypotension", "type": "Finding", "semantic_type": "T033", "synonyms": ["Hypotension", "hypotensive", "low blood pressure"]},
    {"cui": "C0001125", "name": "Lactic Acidosis", "type": "Disease or Syndrome", "semantic_type": "T047", "synonyms": ["Lactic Acidosis", "elevated lactate", "high lactate", "Lactate is elevated"]}
]

def setup_medcat_poc():
    print("Setting up MedCAT for PoC...")

    cdb_dir = 'output/medcat_models/cdb'
    vocab_dir = 'output/medcat_models/vocab'

    if os.path.exists(cdb_dir):
        shutil.rmtree(cdb_dir)
    if os.path.exists(vocab_dir):
        shutil.rmtree(vocab_dir)

    os.makedirs(cdb_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs('output/medcat_models', exist_ok=True)

    config = Config()
    config.general.spacy_model = 'en_core_web_md'

    cdb = CDB(config=config)

    for concept in SEPSIS_CONCEPTS:
        cui = concept['cui']
        name = concept['name']
        sem_type = concept['semantic_type']

        names_dict = {}
        for syn in concept['synonyms']:
            tokens = tuple(syn.lower().split())
            snames = tokens
            names_dict[syn] = NameDescriptor(snames=snames, tokens=tokens, raw_name=syn, is_upper=syn.isupper())

        cdb._add_concept(
            cui=cui,
            names=names_dict,
            ontologies={'UMLS'},
            name_status='A',
            type_ids={sem_type},
            description=name
        )

    cdb.save(cdb_dir)

    vocab = Vocab()
    vocab.save(vocab_dir)

    print("MedCAT models generated and saved to output/medcat_models/")

if __name__ == '__main__':
    setup_medcat_poc()
