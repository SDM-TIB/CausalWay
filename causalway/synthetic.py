"""Synthetic 3-class clinic KG generator (notebook 02 / run_synthetic.py).

Generates a T-Box + A-Box with three classes (Patient, Hospital, Therapy) and
two object properties (``treatedAt``, ``receives``), sampled from a known
ground-truth ontological causal graph.  A discrete (categorical) variant is
written to TTL for the round-trip; a continuous linear-SEM variant is produced
in-memory for NOTEARS / DAGMA / LiNGAM.

The ground-truth DAG is the 11-edge graph from the plan (Plan 1, §7.2), with
two edges traversing ``treatedAt`` in the *inverse* direction and two traversing
``receives``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import XSD

NS = Namespace("http://causalkg.example.org/synthetic/")

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
CLASSES = {
    "Patient": ("Patient", "A patient undergoing treatment at a clinic, with demographic and clinical covariates."),
    "Hospital": ("Hospital", "A healthcare facility where patients receive treatment."),
    "Therapy": ("Therapy", "A therapeutic regimen administered to a patient."),
}

# (domain class, property local name, rdfs:label, rdfs:comment)
DATA_PROPERTIES = [
    ("Patient", "age", "Age", "Patient age group at diagnosis."),
    ("Patient", "smoking", "Smoking", "Smoking status of the patient, a strong driver of disease progression."),
    ("Patient", "geneticRisk", "Genetic Risk", "Genetic susceptibility of the patient."),
    ("Patient", "tumorStage", "Tumor Stage", "Disease stage, influenced by age, smoking, genetics and environmental exposure."),
    ("Patient", "survival", "Survival", "Overall survival category, influenced by disease stage, care quality and treatment toxicity."),
    ("Hospital", "region", "Region", "Geographic region of the hospital."),
    ("Hospital", "airPollution", "Air Pollution", "Local air-pollution level, which depends on the region."),
    ("Hospital", "careQuality", "Care Quality", "Quality of care provided by the hospital."),
    ("Therapy", "drugClass", "Drug Class", "Pharmacological class of the administered drug."),
    ("Therapy", "dosage", "Dosage", "Dosage level of the therapy, chosen from the patient's disease stage."),
    ("Therapy", "toxicity", "Toxicity", "Observed toxicity, driven by drug class and dosage."),
]

# (property local name, domain class, range class, rdfs:label, rdfs:comment)
OBJECT_PROPERTIES = [
    ("treatedAt", "Patient", "Hospital", "treated at",
     "The hospital where the patient is treated."),
    ("receives", "Patient", "Therapy", "receives",
     "The therapy regimen the patient receives."),
]

# Ground-truth ontological causal edges as (cause node name, effect node name).
GROUND_TRUTH_EDGES = [
    ("Patient.age", "Patient.tumorStage"),
    ("Patient.smoking", "Patient.tumorStage"),
    ("Patient.geneticRisk", "Patient.tumorStage"),
    ("Patient.tumorStage", "Patient.survival"),
    ("Hospital.region", "Hospital.airPollution"),
    ("Hospital.airPollution", "Patient.tumorStage"),
    ("Hospital.careQuality", "Patient.survival"),
    ("Patient.tumorStage", "Therapy.dosage"),
    ("Therapy.drugClass", "Therapy.toxicity"),
    ("Therapy.dosage", "Therapy.toxicity"),
    ("Therapy.toxicity", "Patient.survival"),
]


def _node_name(domain: str, prop: str) -> str:
    return f"{domain}.{prop}"


# Canonical order of the 11 data-property node names (matches build_nodes sort).
DATA_NODE_NAMES = sorted(
    [_node_name(d, p) for (d, p, _, _) in DATA_PROPERTIES]
)


# --------------------------------------------------------------------------- #
# Discrete sampling helpers
# --------------------------------------------------------------------------- #
AGE_VALUES = ["Young", "Middle", "Old"]
SMOKING_VALUES = ["No", "Yes"]
GEN_VALUES = ["Low", "High"]
REGION_VALUES = ["North", "South", "East", "West"]
AIR_VALUES = ["Low", "Medium", "High"]
CARE_VALUES = ["Low", "Medium", "High"]
STAGE_VALUES = ["I", "II", "III", "IV"]
SURVIVAL_VALUES = ["Short", "Medium", "Long"]
DRUG_VALUES = ["A", "B", "C"]
DOSAGE_VALUES = ["Low", "Medium", "High"]
TOX_VALUES = ["Low", "High"]

_age_score = {"Young": 0, "Middle": 1, "Old": 2}
_smoking_score = {"No": 0, "Yes": 1}
_gen_score = {"Low": 0, "High": 1}
_region_score = {"North": 0, "South": 1, "East": 2, "West": 3}
_air_score = {"Low": 0, "Medium": 1, "High": 2}
_care_score = {"Low": 0, "Medium": 1, "High": 2}
_stage_score = {"I": 0, "II": 1, "III": 2, "IV": 3}
_drug_score = {"A": 0, "B": 1, "C": 2}
_dosage_score = {"Low": 0, "Medium": 1, "High": 2}
_tox_score = {"Low": 0, "High": 1}


def _quantize(score: float, values: list, thresholds: list) -> str:
    idx = int(np.digitize(score, thresholds))
    return values[min(idx, len(values) - 1)]


@dataclass
class SyntheticData:
    graph: Graph
    truth_edges: list = field(default_factory=lambda: list(GROUND_TRUTH_EDGES))
    continuous_df: Optional[pd.DataFrame] = None
    multiplicity: dict = field(default_factory=dict)
    n_patients: int = 0
    n_hospitals: int = 0
    n_therapies: int = 0

    def write_ttl(self, path: str) -> None:
        self.graph.serialize(destination=path, format="turtle")


def generate(
    n_patients: int = 500,
    n_hospitals: int = 20,
    max_therapies: int = 4,
    seed: int = 42,
) -> SyntheticData:
    """Sample the synthetic KG and return both variants plus the ground truth."""
    rng = np.random.default_rng(seed)
    graph = Graph()
    graph.bind("kg", NS)
    _write_tbox(graph)

    # ---- hospitals ----
    hospital_region = rng.choice(REGION_VALUES, size=n_hospitals)
    hospital_air = [_sample_air(rng, r) for r in hospital_region]
    hospital_care = [_sample_care(rng) for _ in range(n_hospitals)]

    region_cont = rng.normal(0, 1, size=n_hospitals)
    air_cont = 0.8 * region_cont + rng.normal(0, 0.6, size=n_hospitals)
    care_cont = rng.normal(0, 1, size=n_hospitals)

    hospital_iris = []
    for h in range(n_hospitals):
        iri = NS[f"hospital_{h}"]
        hospital_iris.append(iri)
        graph.add((iri, RDF.type, NS.Hospital))
        graph.add((iri, NS.region, Literal(hospital_region[h])))
        graph.add((iri, NS.airPollution, Literal(hospital_air[h])))
        graph.add((iri, NS.careQuality, Literal(hospital_care[h])))

    # ---- patients ----
    patients = []  # (iri, hospital_idx, age, smoking, gen, stage, survival)

    for p in range(n_patients):
        iri = NS[f"patient_{p}"]
        hosp_idx = int(rng.integers(0, n_hospitals))
        age = str(rng.choice(AGE_VALUES))
        smoking = str(rng.choice(SMOKING_VALUES))
        gen = str(rng.choice(GEN_VALUES))
        stage = _sample_stage(rng, age, smoking, gen, hospital_air[hosp_idx])
        patients.append([iri, hosp_idx, age, smoking, gen, stage, None])

        graph.add((iri, RDF.type, NS.Patient))
        graph.add((iri, NS.age, Literal(age)))
        graph.add((iri, NS.smoking, Literal(smoking)))
        graph.add((iri, NS.geneticRisk, Literal(gen)))
        graph.add((iri, NS.tumorStage, Literal(stage)))
        graph.add((iri, NS.treatedAt, hospital_iris[hosp_idx]))

    # ---- therapies (assign after patients so stage is known) ----
    n_therapies = 0
    therapy_count = rng.integers(1, max_therapies + 1, size=n_patients)

    # continuous SEM generation (flat-join granularity)
    age_cont = rng.normal(0, 1, size=n_patients)
    smoking_cont = rng.binomial(1, 0.5, size=n_patients).astype(float)
    gen_cont = rng.binomial(1, 0.3, size=n_patients).astype(float)
    stage_cont = (
        0.35 * age_cont + 0.7 * smoking_cont + 0.6 * gen_cont
        + 0.6 * air_cont[np.array([p[1] for p in patients])]
        + rng.normal(0, 0.5, size=n_patients)
    )

    cont_cols = {name: [] for name in DATA_NODE_NAMES}
    tox_sum = np.zeros(n_patients)
    n_therapies_per_patient = np.zeros(n_patients)

    for p in range(n_patients):
        iri, hosp_idx, age, smoking, gen, stage, _ = patients[p]
        for t in range(int(therapy_count[p])):
            t_iri = NS[f"therapy_{n_therapies}"]
            n_therapies += 1
            drug = str(rng.choice(DRUG_VALUES))
            dosage = _sample_dosage(rng, stage)
            tox = _sample_tox(rng, drug, dosage)
            graph.add((t_iri, RDF.type, NS.Therapy))
            graph.add((t_iri, NS.drugClass, Literal(drug)))
            graph.add((t_iri, NS.dosage, Literal(dosage)))
            graph.add((t_iri, NS.toxicity, Literal(tox)))
            graph.add((iri, NS.receives, t_iri))

            tox_sum[p] += _tox_score[tox]
            n_therapies_per_patient[p] += 1

            # continuous flat-join row
            drug_cont = float(rng.normal(0, 1))
            dosage_cont = 0.8 * stage_cont[p] + rng.normal(0, 0.4)
            tox_cont = 0.5 * drug_cont + 0.8 * dosage_cont + rng.normal(0, 0.4)
            row = {
                _node_name("Patient", "age"): age_cont[p],
                _node_name("Patient", "smoking"): smoking_cont[p],
                _node_name("Patient", "geneticRisk"): gen_cont[p],
                _node_name("Patient", "tumorStage"): stage_cont[p],
                _node_name("Hospital", "region"): region_cont[hosp_idx],
                _node_name("Hospital", "airPollution"): air_cont[hosp_idx],
                _node_name("Hospital", "careQuality"): care_cont[hosp_idx],
                _node_name("Therapy", "drugClass"): drug_cont,
                _node_name("Therapy", "dosage"): dosage_cont,
                _node_name("Therapy", "toxicity"): tox_cont,
            }
            for k, v in row.items():
                cont_cols[k].append(v)

    # survival depends on aggregate therapy toxicity (mean tox score per patient)
    for p in range(n_patients):
        iri, hosp_idx, age, smoking, gen, stage, _ = patients[p]
        m = tox_sum[p] / n_therapies_per_patient[p] if n_therapies_per_patient[p] > 0 else 0.0
        survival = _sample_survival(rng, stage, hospital_care[hosp_idx], m)
        patients[p][6] = survival
        graph.add((iri, NS.survival, Literal(survival)))

    # survival (continuous) per patient -> repeated on each flat-join row
    survival_cont = (
        -0.5 * stage_cont
        + 0.5 * care_cont[np.array([p[1] for p in patients])]
        - 1.0 * (tox_sum / np.maximum(n_therapies_per_patient, 1))
        + rng.normal(0, 0.4, size=n_patients)
    )
    # distribute patient-level continuous values into the flat-join columns
    cont_cols[_node_name("Patient", "survival")] = []
    for p in range(n_patients):
        for _ in range(int(therapy_count[p])):
            cont_cols[_node_name("Patient", "survival")].append(survival_cont[p])

    continuous_df = pd.DataFrame(cont_cols)[DATA_NODE_NAMES]

    multiplicity = {
        "Patient": float(n_therapies / n_patients),
        "Hospital": float(n_patients / n_hospitals),
        "Therapy": float(n_therapies / n_therapies) if n_therapies else 0.0,
    }

    return SyntheticData(
        graph=graph,
        continuous_df=continuous_df,
        multiplicity=multiplicity,
        n_patients=n_patients,
        n_hospitals=n_hospitals,
        n_therapies=n_therapies,
    )


# --------------------------------------------------------------------------- #
# Discrete CPT-like sampling
# --------------------------------------------------------------------------- #
def _sample_air(rng, region):
    s = _region_score[region] * 0.8 + rng.normal(0, 0.4)
    return _quantize(s, AIR_VALUES, [0.8, 1.6])


def _sample_care(rng):
    s = rng.normal(1.5, 0.7)
    return _quantize(s, CARE_VALUES, [0.7, 1.5])


def _sample_stage(rng, age, smoking, gen, air):
    s = (
        _age_score[age] * 0.35
        + _smoking_score[smoking] * 0.7
        + _gen_score[gen] * 0.6
        + _air_score[air] * 0.6
        + rng.normal(0, 0.4)
    )
    return _quantize(s, STAGE_VALUES, [0.6, 1.4, 2.2])


def _sample_dosage(rng, stage):
    s = _stage_score[stage] * 0.8 + rng.normal(0, 0.4)
    return _quantize(s, DOSAGE_VALUES, [0.7, 1.5])


def _sample_tox(rng, drug, dosage):
    s = _drug_score[drug] * 0.5 + _dosage_score[dosage] * 0.8 + rng.normal(0, 0.4)
    return "High" if s > 1.2 else "Low"


def _sample_survival(rng, stage, care, mean_tox_score):
    s = (
        -_stage_score[stage] * 0.5
        + _care_score[care] * 0.5
        - mean_tox_score * 1.0
        + rng.normal(1.9, 0.4)
    )
    return _quantize(s, SURVIVAL_VALUES, [1.0, 2.0])


# --------------------------------------------------------------------------- #
# TTL writer
# --------------------------------------------------------------------------- #
def _write_tbox(graph: Graph) -> None:
    for name, (label, comment) in CLASSES.items():
        c = NS[name]
        graph.add((c, RDF.type, OWL.Class))
        graph.add((c, RDFS.label, Literal(label)))
        graph.add((c, RDFS.comment, Literal(comment)))

    for domain, prop, label, comment in DATA_PROPERTIES:
        p = NS[prop]
        graph.add((p, RDF.type, OWL.DatatypeProperty))
        graph.add((p, RDFS.domain, NS[domain]))
        graph.add((p, RDFS.range, XSD.string))
        graph.add((p, RDFS.label, Literal(label)))
        graph.add((p, RDFS.comment, Literal(comment)))

    for prop, domain, range_, label, comment in OBJECT_PROPERTIES:
        p = NS[prop]
        graph.add((p, RDF.type, OWL.ObjectProperty))
        graph.add((p, RDFS.domain, NS[domain]))
        graph.add((p, RDFS.range, NS[range_]))
        graph.add((p, RDFS.label, Literal(label)))
        graph.add((p, RDFS.comment, Literal(comment)))


def truth_adjacency(node_names: list) -> np.ndarray:
    """Ground-truth adjacency matrix over the given node-name order."""
    idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    adj = np.zeros((n, n), dtype=int)
    for cause, effect in GROUND_TRUTH_EDGES:
        if cause in idx and effect in idx:
            adj[idx[cause], idx[effect]] = 1
    return adj
