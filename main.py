"""
Simulation multi-agent de la chaine pharmaceutique francaise - version amelioree.

Objectifs:
- politique de commande explicite pour la pharmacie et le grossiste
- production adaptative du fabricant
- prevision naive sur historique glissant
- export de datasets pour l'apprentissage ML / XGBoost
- compatibilite avec un dashboard Streamlit dans /dashboard
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    plt = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

LAMBDA_HEBDO_FR = 0.0014
DELTA_FR = 0.20
DUREE_DISRUPTION_MIN_FR = 6
DUREE_DISRUPTION_MAX_FR = 52

DELAI_FABRICANT_GROSSISTE = 4
DELAI_GROSSISTE_PHARMACIE = 1

FUTURE_HORIZON_WEEKS = 4
HISTORIQUE_PREVISION = 8


class Fabricant:
    """Produit selon une capacite adaptee a la demande observee."""

    def __init__(self, capacite_nominale, lam=LAMBDA_HEBDO_FR, delta=DELTA_FR):
        self.capacite_nominale = capacite_nominale
        self.lam = lam
        self.delta = delta
        self.en_disruption = False
        self.duree_restante = 0

    def produire(self, demande_observee):
        if not self.en_disruption and random.random() < self.lam:
            self.en_disruption = True
            self.duree_restante = random.randint(
                DUREE_DISRUPTION_MIN_FR,
                DUREE_DISRUPTION_MAX_FR,
            )

        capacite_theorique = min(demande_observee * 1.10, self.capacite_nominale)

        if self.en_disruption:
            production = capacite_theorique * (1 - self.delta)
            self.duree_restante -= 1
            if self.duree_restante <= 0:
                self.en_disruption = False
                self.duree_restante = 0
        else:
            production = capacite_theorique

        return production


class FileDelai:
    """File simple pour modeliser un delai fixe de transport ou de transmission."""

    def __init__(self, delai):
        self.delai = delai
        self.file = deque([0.0] * delai)

    def envoyer(self, quantite):
        self.file.append(float(quantite))

    def recevoir(self):
        return float(self.file.popleft())

    def niveau(self):
        return float(sum(self.file))


class Grossiste:
    """Agent intermediaire avec regle de commande simple."""

    def __init__(self, stock_initial, seuil_critique):
        self.stock = float(stock_initial)
        self.seuil_critique = float(seuil_critique)
        self.en_rupture = False

    def calcul_commande(self, demande_prevue):
        stock_cible = self.seuil_critique * 8
        return max(0.0, stock_cible - self.stock + demande_prevue)

    def recevoir(self, quantite):
        self.stock += float(quantite)

    def servir(self, demande):
        livre = min(self.stock, demande)
        self.stock -= livre
        self.en_rupture = self.stock < self.seuil_critique
        return livre


class Pharmacie:
    """Dernier maillon avant le patient, avec politique de commande cible."""

    def __init__(self, stock_initial, seuil_critique):
        self.stock = float(stock_initial)
        self.seuil_critique = float(seuil_critique)
        self.en_rupture = False
        self.demande_non_servie_cumulee = 0.0

    def calcul_commande(self, demande_prevue):
        stock_cible = self.seuil_critique * 4
        return max(0.0, stock_cible - self.stock + demande_prevue)

    def recevoir(self, quantite):
        self.stock += float(quantite)

    def vendre(self, demande):
        vendu = min(self.stock, demande)
        manque = demande - vendu
        self.stock -= vendu
        self.en_rupture = self.stock <= 0
        self.demande_non_servie_cumulee += manque
        taux_service = 1.0 if demande <= 0 else vendu / demande
        return vendu, manque, taux_service


def demande_patient(d0, sigma):
    return max(0.0, random.gauss(d0, sigma))


def prevision_naive(historique_demandes, d0):
    if not historique_demandes:
        return float(d0)
    return float(sum(historique_demandes) / len(historique_demandes))


def _initialiser_historiques():
    return {
        "periode": [],
        "capacite_fabricant": [],
        "disruption_fabricant": [],
        "duree_disruption_restante": [],
        "demande_patient": [],
        "demande_prevue": [],
        "commande_pharmacie": [],
        "commande_grossiste": [],
        "livraison_grossiste": [],
        "livraison_pharmacie": [],
        "demande_non_servie": [],
        "taux_service": [],
        "stock_grossiste": [],
        "stock_pharmacie": [],
        "stock_transit_fg": [],
        "stock_transit_gp": [],
        "rupture_grossiste": [],
        "rupture_pharmacie": [],
    }


def lancer_simulation(n_periodes=104, d0=100, sigma=15, graine=42):
    random.seed(graine)

    fabricant = Fabricant(capacite_nominale=130)
    grossiste = Grossiste(
        stock_initial=d0 * DELAI_FABRICANT_GROSSISTE * 1.5,
        seuil_critique=d0 * 2,
    )
    pharmacie = Pharmacie(
        stock_initial=d0 * DELAI_GROSSISTE_PHARMACIE * 1.5,
        seuil_critique=d0 * 0.5,
    )

    file_fab_gros = FileDelai(DELAI_FABRICANT_GROSSISTE)
    file_gros_pharma = FileDelai(DELAI_GROSSISTE_PHARMACIE)
    file_commande_gros_fab = FileDelai(DELAI_FABRICANT_GROSSISTE)

    historique_demandes = deque(maxlen=HISTORIQUE_PREVISION)
    hist = _initialiser_historiques()

    for t in range(n_periodes):
        demande_prevue = prevision_naive(historique_demandes, d0)
        commande_pharmacie = pharmacie.calcul_commande(demande_prevue)
        commande_grossiste = grossiste.calcul_commande(demande_prevue)

        file_gros_pharma.envoyer(commande_pharmacie)
        file_commande_gros_fab.envoyer(commande_grossiste)

        livraison_grossiste = file_fab_gros.recevoir()
        grossiste.recevoir(livraison_grossiste)

        commande_arrivee = file_gros_pharma.recevoir()
        livraison_pharmacie = grossiste.servir(commande_arrivee)
        pharmacie.recevoir(livraison_pharmacie)

        demande_observee_fab = file_commande_gros_fab.recevoir()
        demande_pour_production = max(demande_prevue, demande_observee_fab)
        capacite = fabricant.produire(demande_pour_production)
        file_fab_gros.envoyer(capacite)

        D = demande_patient(d0, sigma)
        vendu, manque, taux_service = pharmacie.vendre(D)
        historique_demandes.append(D)

        hist["periode"].append(t)
        hist["capacite_fabricant"].append(capacite)
        hist["disruption_fabricant"].append(fabricant.en_disruption)
        hist["duree_disruption_restante"].append(fabricant.duree_restante)
        hist["demande_patient"].append(D)
        hist["demande_prevue"].append(demande_prevue)
        hist["commande_pharmacie"].append(commande_pharmacie)
        hist["commande_grossiste"].append(commande_grossiste)
        hist["livraison_grossiste"].append(livraison_grossiste)
        hist["livraison_pharmacie"].append(livraison_pharmacie)
        hist["demande_non_servie"].append(manque)
        hist["taux_service"].append(taux_service)
        hist["stock_grossiste"].append(grossiste.stock)
        hist["stock_pharmacie"].append(pharmacie.stock)
        hist["stock_transit_fg"].append(file_fab_gros.niveau())
        hist["stock_transit_gp"].append(file_gros_pharma.niveau())
        hist["rupture_grossiste"].append(grossiste.en_rupture)
        hist["rupture_pharmacie"].append(pharmacie.en_rupture)

    rupture_grossiste = hist["rupture_grossiste"]
    target = []
    for i in range(len(rupture_grossiste)):
        future_window = rupture_grossiste[i + 1 : i + FUTURE_HORIZON_WEEKS + 1]
        target.append(int(any(future_window)))
    hist["target_rupture_4_semaines"] = target

    return hist, fabricant, grossiste, pharmacie


def hist_to_dataframe(hist, simulation_id=None):
    df = pd.DataFrame(hist)
    if simulation_id is not None:
        df["simulation"] = simulation_id
    return df


def sauvegarder_dataset_single_run(hist, output_path=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = DATA_DIR / "dataset_simulation_ml.csv"
    df = hist_to_dataframe(hist)
    df.to_csv(output_path, index=False)
    return output_path


def generer_dataset_monte_carlo(
    n_simulations=1000,
    n_periodes=104,
    d0=100,
    sigma=15,
    output_path=None,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = DATA_DIR / "dataset_xgboost.csv"

    frames = []
    for i in range(n_simulations):
        hist, _, _, _ = lancer_simulation(
            n_periodes=n_periodes,
            d0=d0,
            sigma=sigma,
            graine=i,
        )
        frames.append(hist_to_dataframe(hist, simulation_id=i))

    dataset = pd.concat(frames, ignore_index=True)
    dataset.to_csv(output_path, index=False)
    return dataset, output_path


def afficher(hist, output_path=None):
    if plt is None:
        print("matplotlib n'est pas installe: generation du graphique ignoree.")
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = IMAGES_DIR / "resultats_simulation_v3.png"

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(hist["periode"], hist["demande_patient"], color="#7A3E9D", linewidth=1.3)
    axes[0].plot(hist["periode"], hist["demande_prevue"], color="#2B7A78", linewidth=1.1)
    axes[0].set_ylabel("Demande")
    axes[0].set_title("Simulation pharmaceutique multi-agent")
    axes[0].legend(["Demande observee", "Demande prevue"], fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist["periode"], hist["stock_grossiste"], color="#1D5FA6", linewidth=1.3)
    axes[1].plot(hist["periode"], hist["stock_transit_fg"], color="#4C9F70", linewidth=1.0)
    axes[1].set_ylabel("Grossiste")
    axes[1].legend(["Stock grossiste", "Transit FG"], fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(hist["periode"], hist["stock_pharmacie"], color="#6D28D9", linewidth=1.3)
    axes[2].plot(hist["periode"], hist["stock_transit_gp"], color="#F59E0B", linewidth=1.0)
    axes[2].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Pharmacie")
    axes[2].legend(["Stock pharmacie", "Transit GP"], fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot(hist["periode"], hist["taux_service"], color="#1A6B3C", linewidth=1.3)
    axes[3].set_ylabel("Taux service")
    axes[3].set_xlabel("Periode (semaine)")
    axes[3].set_ylim(0, 1.05)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Graphique sauvegarde : {output_path}")


def _print_resume(hist, pharmacie):
    nb_rupture_g = sum(hist["rupture_grossiste"])
    nb_rupture_p = sum(hist["rupture_pharmacie"])
    nb_disrupt = sum(hist["disruption_fabricant"])
    n = len(hist["periode"])

    print(f"Semaines de disruption fabricant : {nb_disrupt}/{n} ({nb_disrupt/n*100:.1f}%)")
    print(f"Semaines en rupture grossiste     : {nb_rupture_g}/{n} ({nb_rupture_g/n*100:.1f}%)")
    print(f"Semaines en rupture pharmacie     : {nb_rupture_p}/{n} ({nb_rupture_p/n*100:.1f}%)")
    print(f"Demande patient non servie (cumul): {pharmacie.demande_non_servie_cumulee:.0f} unites")


def main():
    parser = argparse.ArgumentParser(description="Simulation multi-agent pharmaceutique.")
    parser.add_argument("--n-periodes", type=int, default=104)
    parser.add_argument("--d0", type=float, default=100)
    parser.add_argument("--sigma", type=float, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Genere le dataset monte carlo pour le ML.",
    )
    parser.add_argument(
        "--mc-simulations",
        type=int,
        default=1000,
        help="Nombre de simulations pour le dataset ML.",
    )
    args = parser.parse_args()

    print("=== Simulation pharmaceutique multi-agent ===\n")
    hist, fabricant, grossiste, pharmacie = lancer_simulation(
        n_periodes=args.n_periodes,
        d0=args.d0,
        sigma=args.sigma,
        graine=args.seed,
    )
    _print_resume(hist, pharmacie)
    afficher(hist)
    sauvegarder_dataset_single_run(hist)

    if args.generate_dataset:
        print("\nGeneration du dataset monte carlo...")
        _, output_path = generer_dataset_monte_carlo(
            n_simulations=args.mc_simulations,
            n_periodes=args.n_periodes,
            d0=args.d0,
            sigma=args.sigma,
        )
        print(f"Dataset sauvegarde : {output_path}")


if __name__ == "__main__":
    main()
