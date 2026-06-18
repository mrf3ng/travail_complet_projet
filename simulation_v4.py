"""
Simulation multi-agent de la chaine pharmaceutique francaise - VERSION 4
Projet de recherche S8 - Centrale Lille

PERIMETRE :
Maillon Fabricant -> Grossiste -> Pharmacie de ville (circuit ville),
identifie comme le plus critique dans la modelisation theorique (point de
rupture R2 au niveau du grossiste).

CHANGEMENT PAR RAPPORT A LA V3 :
La v3 presentait une limite assumee : le stock du grossiste croissait de
facon continue (~25-30 unites/semaine), car le fabricant produisait
systematiquement sa capacite nominale (130), superieure a la demande
moyenne (100), SANS AUCUN mecanisme reliant sa production a la demande
reelle ou au stock du grossiste. Ce n'etait pas un simple "effet de bord"
mais une consequence arithmetique directe et garantie du parametrage
(surplus structurel = capacite_nominale - D0 = 30/semaine, quelle que
soit la disruption ou la variance de la demande).

Cette version corrige le probleme en implementant la regle de production
ADAPTATIVE deja identifiee comme piste dans la v3, inspiree de ShortageSim
(qi = min(D/n, ci)) : le fabricant ne produit plus une quantite fixe, mais
ajuste sa production a un signal de demande lisse (moyenne mobile
exponentielle de la demande recue par le grossiste), plafonne par sa
capacite disponible (qui chute en cas de disruption).

  demande_lissee(t) = alpha * demande_observee(t) + (1 - alpha) * demande_lissee(t-1)
  production(t)     = min(demande_lissee(t), capacite_disponible(t))

Consequence directe verifiee par simulation (20 graines, voir notes du
projet) : le stock du grossiste se stabilise autour d'un niveau quasi
constant au lieu de diverger, et reste borne meme sur un horizon de 10
ans, sans qu'aucune derive lente n'apparaisse.

AUTRE CHANGEMENT : le seuil critique du grossiste a ete recalibre. Avec
la regle de production fixe de la v3, seuil_critique = D0 * 2 (= 200)
n'avait jamais ete atteint car le stock grossissait sans cesse. Avec la
regle adaptative, le stock du grossiste oscille naturellement autour de
~180 (ecart-type ~38) en l'absence de disruption -- un seuil a 200 etait
donc presque TOUJOURS franchi par le bas, ce qui aurait fait apparaitre
une fausse alerte permanente (jusqu'a 75% du temps en "rupture" meme sans
aucune disruption du fabricant). Le seuil a ete abaisse a D0 * 0.8 (= 80)
pour ne signaler un etat critique que lors d'un veritable ecart par
rapport au regime normal (typiquement en cas de disruption prolongee du
fabricant), conformement a l'intention initiale du modele.

PARAMETRES CALIBRES SUR DONNEES FRANCAISES (ANSM / Open Medic 2021-2023) :
  lambda = 0.14%/semaine (probabilite de disruption du fabricant)
  delta  = 20% (perte de capacite en cas de disruption, valeur ShortageSim
           conservee en l'absence de donnee francaise directement
           comparable)
  duree disruption = 6 a 52 semaines, moyenne 28 semaines (apres nettoyage
           des valeurs aberrantes du fichier ANSM -- dates de fin
           anterieures aux dates de debut, et durees superieures a 2 ans
           probablement liees a des fiches non mises a jour)

LIMITES ASSUMEES (inchangees par rapport a la v3, toujours valables) :
  - lambda est estime sur un tres petit echantillon (3 evenements sur 42
    observations categorie x annee), donc cette valeur est fragile
    statistiquement.
  - delta = 20% est une valeur reprise de ShortageSim (contexte
    americain), non calibree specifiquement sur le marche francais.
  - Les seuils de stock (capacite, stocks de securite) restent des choix
    de modelisation raisonnables mais non calibres sur des donnees
    reelles de stock interne (donnees non disponibles publiquement --
    secret commercial), comme explique dans le rapport.
  - Meme apres correction, le systeme reste relativement robuste face a
    une disruption isolee (le stock grossiste joue son role de tampon).
    Les vraies "ruptures" severes restent rares sur un horizon de 2 ans
    avec les parametres actuels -- point a surveiller si la simulation
    doit servir a generer des donnees synthetiques pour le volet ML, et
    a discuter avec l'encadrant (cf. point 2 du planning : enrichir ou
    non les donnees reelles avec des donnees synthetiques).
"""

import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque


LAMBDA_HEBDO_FR = 0.0014
DELTA_FR = 0.20
DUREE_DISRUPTION_MIN_FR = 6
DUREE_DISRUPTION_MAX_FR = 52

DELAI_FABRICANT_GROSSISTE = 4   # semaines
DELAI_GROSSISTE_PHARMACIE = 1   # semaine

ALPHA_LISSAGE_DEMANDE = 0.3      # poids du dernier signal de demande dans la moyenne mobile exponentielle


class Fabricant:
    """Produit chaque semaine selon une regle ADAPTATIVE (v4) : la production suit
    un signal de demande lisse (moyenne mobile exponentielle), plafonne par la
    capacite disponible (qui chute en cas de disruption). Inspiree de la regle
    qi = min(D/n, ci) de ShortageSim.

    Avant (v3) : production = capacite_nominale fixe, sans lien avec la demande
    reelle -> surplus structurel garanti si capacite_nominale > D0 (cf. notes v4
    en tete de fichier). Desormais corrige.
    """
    def __init__(self, capacite_nominale, demande_initiale, lam=LAMBDA_HEBDO_FR,
                 delta=DELTA_FR, alpha=ALPHA_LISSAGE_DEMANDE):
        self.capacite_nominale = capacite_nominale
        self.lam = lam
        self.delta = delta
        self.alpha = alpha
        self.en_disruption = False
        self.duree_restante = 0
        self.demande_lissee = demande_initiale  # amorce du signal a t=0

    def produire(self, demande_observee):
        # Mise a jour du signal de demande (simplification : le fabricant recoit
        # directement la demande arrivee au grossiste comme signal ; un modele plus
        # fin pourrait ajouter un delai d'information supplementaire ici).
        self.demande_lissee = self.alpha * demande_observee + (1 - self.alpha) * self.demande_lissee

        if self.en_disruption:
            self.duree_restante -= 1
            if self.duree_restante <= 0:
                self.en_disruption = False
        elif random.random() < self.lam:
            self.en_disruption = True
            self.duree_restante = random.randint(DUREE_DISRUPTION_MIN_FR, DUREE_DISRUPTION_MAX_FR)

        capacite_dispo = self.capacite_nominale * (1 - self.delta) if self.en_disruption else self.capacite_nominale
        return min(self.demande_lissee, capacite_dispo)


class FileDelai:
    """File d'attente simple pour modeliser un delai de transport fixe entre 2 agents."""
    def __init__(self, delai):
        self.delai = delai
        self.file = deque([0] * delai)  # initialement vide (pas de flux en transit au demarrage)

    def envoyer(self, quantite):
        self.file.append(quantite)

    def recevoir(self):
        return self.file.popleft()


class Grossiste:
    """Recoit le flux du fabricant (apres delai), sert la pharmacie selon sa demande."""
    def __init__(self, stock_initial, seuil_critique):
        self.stock = stock_initial
        self.seuil_critique = seuil_critique
        self.en_rupture = False

    def recevoir(self, quantite):
        self.stock += quantite

    def servir(self, demande):
        livre = min(self.stock, demande)
        self.stock -= livre
        self.en_rupture = self.stock < self.seuil_critique
        return livre


class Pharmacie:
    """Recoit le flux du grossiste (apres delai), vend au patient."""
    def __init__(self, stock_initial, seuil_critique):
        self.stock = stock_initial
        self.seuil_critique = seuil_critique
        self.en_rupture = False
        self.demande_non_servie_cumulee = 0

    def recevoir(self, quantite):
        self.stock += quantite

    def vendre(self, demande):
        vendu = min(self.stock, demande)
        manque = demande - vendu
        self.stock -= vendu
        self.en_rupture = self.stock <= 0
        self.demande_non_servie_cumulee += manque
        return vendu


def demande_patient(D0, sigma):
    return max(0, random.gauss(D0, sigma))


def lancer_simulation(n_periodes=104, D0=100, sigma=15, graine=42):
    random.seed(graine)

    fabricant = Fabricant(capacite_nominale=130, demande_initiale=D0)
    file_fab_gros = FileDelai(DELAI_FABRICANT_GROSSISTE)
    grossiste = Grossiste(stock_initial=D0 * DELAI_FABRICANT_GROSSISTE * 1.5,
                           seuil_critique=D0 * 0.8)
    file_gros_pharma = FileDelai(DELAI_GROSSISTE_PHARMACIE)
    pharmacie = Pharmacie(stock_initial=D0 * DELAI_GROSSISTE_PHARMACIE * 1.5,
                          seuil_critique=D0 * 0.5)

    hist = {k: [] for k in ['periode', 'capacite_fabricant', 'disruption_fabricant',
                             'stock_grossiste', 'stock_pharmacie', 'demande_patient',
                             'rupture_grossiste', 'rupture_pharmacie']}

    # Signal de demande transmis au fabricant : la demande recue par le grossiste
    # a la periode precedente (amorce a D0 pour la toute premiere periode).
    demande_recue_grossiste_precedente = D0

    for t in range(n_periodes):
        # 1. Le fabricant ajuste sa production au signal de demande, puis produit
        #    -> part dans le "tuyau" vers le grossiste
        capacite = fabricant.produire(demande_recue_grossiste_precedente)
        file_fab_gros.envoyer(capacite)
        arrivee_chez_grossiste = file_fab_gros.recevoir()
        grossiste.recevoir(arrivee_chez_grossiste)

        # 2. La pharmacie exprime sa demande au grossiste -> part dans le "tuyau"
        D = demande_patient(D0, sigma)
        file_gros_pharma.envoyer(D)               # la demande d'aujourd'hui part vers le grossiste
        demande_arrivee_au_grossiste = file_gros_pharma.recevoir()  # la demande d'il y a "delai" semaines arrive
        livre_a_pharmacie = grossiste.servir(demande_arrivee_au_grossiste)
        pharmacie.recevoir(livre_a_pharmacie)

        # 3. La pharmacie vend au patient (demande du jour meme, deja en stock ou pas)
        pharmacie.vendre(D)

        # 4. Le signal transmis au fabricant pour la PROCHAINE periode est la
        #    demande qui vient d'arriver au grossiste cette periode-ci.
        demande_recue_grossiste_precedente = demande_arrivee_au_grossiste

        hist['periode'].append(t)
        hist['capacite_fabricant'].append(capacite)
        hist['disruption_fabricant'].append(fabricant.en_disruption)
        hist['stock_grossiste'].append(grossiste.stock)
        hist['stock_pharmacie'].append(pharmacie.stock)
        hist['demande_patient'].append(D)
        hist['rupture_grossiste'].append(grossiste.en_rupture)
        hist['rupture_pharmacie'].append(pharmacie.en_rupture)

    return hist, fabricant, grossiste, pharmacie


def afficher(hist):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(hist['periode'], hist['capacite_fabricant'], color='#1A6B3C', linewidth=1.3)
    axes[0].set_ylabel('Production\nfabricant')
    axes[0].set_title('Simulation Grossiste-Pharmacie (v4) - production adaptative, parametres calibres France')
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist['periode'], hist['stock_grossiste'], color='#1D5FA6', linewidth=1.3, label='Stock grossiste')
    axes[1].set_ylabel('Stock\ngrossiste')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(hist['periode'], hist['stock_pharmacie'], color='#6D28D9', linewidth=1.3, label='Stock pharmacie')
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1, label='Rupture (stock=0)')
    axes[2].set_ylabel('Stock\npharmacie')
    axes[2].set_xlabel('Periode (semaine)')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/simulation/resultats_simulation_v4.png', dpi=150)
    print("Graphique sauvegarde : resultats_simulation_v4.png")


if __name__ == '__main__':
    print("=== Simulation v4 (production adaptative, stock grossiste stabilise) ===\n")
    hist, fabricant, grossiste, pharmacie = lancer_simulation()

    nb_rupture_g = sum(hist['rupture_grossiste'])
    nb_rupture_p = sum(hist['rupture_pharmacie'])
    nb_disrupt = sum(hist['disruption_fabricant'])

    print(f"Semaines de disruption fabricant : {nb_disrupt}/104 ({nb_disrupt/104*100:.1f}%)")
    print(f"Semaines en rupture grossiste     : {nb_rupture_g}/104 ({nb_rupture_g/104*100:.1f}%)")
    print(f"Semaines en rupture pharmacie     : {nb_rupture_p}/104 ({nb_rupture_p/104*100:.1f}%)")
    print(f"Demande patient non servie (cumul): {pharmacie.demande_non_servie_cumulee:.0f} unites")
    print(f"Stock grossiste final             : {grossiste.stock:.0f} (debut : {hist['stock_grossiste'][0]:.0f})")

    afficher(hist)