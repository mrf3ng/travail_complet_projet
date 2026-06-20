# Rapport scientifique

## Titre

**Modelisation multi-agent de la chaine pharmaceutique francaise et prediction des ruptures de stock par machine learning**

---

## Resume

Ce projet etudie la dynamique des ruptures de stock dans une chaine pharmaceutique a travers deux briques principales:

1. une simulation multi-agent de la chaine d'approvisionnement;
2. une chaine de prediction supervisee pour anticiper une rupture a court horizon.

La version actuellement implantee du depot se concentre sur un noyau simple et interpretable de la chaine **fabricant -> grossiste repartiteur -> pharmacie de ville**. La simulation est inspiree de la logique de `simulation_v4.py`: la production du fabricant suit une demande lissee, les delais logistiques sont modelises explicitement, et les stocks du grossiste et de la pharmacie evoluent au fil des periodes. Cette trajectoire unique sert ensuite de base a la generation de donnees tabulaires exploitables pour l'apprentissage automatique.

Le projet comprend egalement un module de machine learning capable d'entrainer un classifieur binaire de rupture a 4 semaines, en privilegant `XGBoost` quand la dependance est disponible. En complement, une interface Streamlit permet de generer le dataset, d'entrainer le modele, de charger un modele sauvegarde et de visualiser les predictions sur une simulation.

---

## 1. Contexte scientifique et industriel

Les ruptures de medicaments constituent un probleme majeur pour les systemes de sante. Elles perturbent les parcours de soins, degradent la continuite therapeutique et imposent des adaptations couteuses aux differents acteurs de la chaine.

Dans le contexte francais, le role du grossiste repartiteur est central: il se situe entre le fabricant et les pharmacies de ville, et absorbe une partie des chocs grace a son stock tampon. Ce maillon est donc un point d'observation pertinent pour etudier la propagation des tensions d'approvisionnement.

Le projet s'inscrit dans une demarche de modelisation appliquee: comprendre comment un incident amont se propage, quelles variables internes signalent une fragilisation de la chaine, et comment transformer ces trajectoires en un probleme previsible par machine learning.

---

## 2. Objectifs du projet

Les objectifs operationnels sont les suivants:

1. modeliser la chaine pharmaceutique sous forme d'agents interagissants;
2. simuler les flux physiques sur plusieurs semaines;
3. produire des donnees structurees a partir de la simulation;
4. entrainer un classifieur de rupture a horizon 4 semaines;
5. visualiser les resultats dans un dashboard interactif;
6. conserver une architecture reproductible et simple a maintenir.

---

## 3. Perimetre de la version actuelle

L'architecture logicielle actuelle est organisee autour de plusieurs fichiers:

- `main.py`: simulation, generation de datasets, export des graphiques;
- `ml_utils.py`: preparation des donnees, entrainement, sauvegarde et prediction;
- `dashboard/app.py`: interface Streamlit;
- `app.py`: point d'entree pratique pour Streamlit;
- `data/`: datasets generes;
- `images/`: figures produites par la simulation;
- `models/`: modele sauvegarde;
- `logs/`: journal du dashboard.

La version actuelle ne cherche pas encore a reproduire toute la complexite du modele theorique complet. Elle concentre l'effort sur le segment le plus utile pour la dynamique des ruptures et pour l'enrichissement du dataset.

---

## 4. Modelisation multi-agent

### 4.1. Agents consideres

La simulation actuelle comporte trois agents:

- **Fabricant**: produit le medicament et peut subir une disruption;
- **Grossiste repartiteur**: recoit la production avec un delai et alimente la pharmacie;
- **Pharmacie de ville**: recoit le flux du grossiste et sert la demande patient.

Cette version ne contient pas d'agent regulateur explicite, ni de circuit hospitalier. Le modele a ete volontairement recentre pour garder une base robuste et lisible.

### 4.2. Demande patient

La demande hebdomadaire est modelisee par une variable aleatoire gaussienne tronquee a zero:

$$
D_t = \max(0, \mathcal{N}(D_0, \sigma^2))
$$

ou:

- $D_t$ est la demande observee a la semaine $t$;
- $D_0$ est la demande moyenne;
- $\sigma$ est l'ecart-type de la demande.

Cette hypothese permet de representer une demande fluctuante tout en restant physiquement realiste.

### 4.3. Lissage de la demande

Le fabricant ne repond pas directement a une observation ponctuelle. Il maintient un signal de demande lisse, mis a jour par moyenne mobile exponentielle:

$$
\tilde{D}_t = \alpha D_t + (1-\alpha)\tilde{D}_{t-1}
$$

ou $\alpha$ est le coefficient de lissage.

Cette logique est l'un des points qui rapproche le depot actuel de l'esprit de `ShortageSim`: on ne se limite pas a une production fixe, on repond a un signal agrege plutot qu'a une valeur instantanee.

### 4.4. Production du fabricant

Le fabricant dispose d'une capacite nominale $C_{nom}$. Sa production est limitee par la demande lissee et par la capacite disponible en cas de disruption:

$$
Q_t = \min(\tilde{D}_t, C_t)
$$

avec:

$$
C_t =
\begin{cases}
C_{nom} & \text{si aucune disruption n'est active} \\
(1-\delta)C_{nom} & \text{si une disruption est active}
\end{cases}
$$

La version actuelle reste volontairement simple: elle ne simule pas de politique de commande complexe, mais elle elimine le probleme de derive structurelle du stock qui existait dans la version precedente.

### 4.5. Modelisation de la disruption

La disruption du fabricant est traitee comme un evenement stochastique:

$$
Z_t \sim \mathcal{B}(\lambda)
$$

avec:

- $Z_t = 1$ si une disruption debute a la periode $t$;
- $\lambda$ la probabilite hebdomadaire de disruption.

Une fois declenchee, la disruption dure un nombre aleatoire de semaines borne entre un minimum et un maximum. Cette duree est suivie par un compteur interne.

### 4.6. Delais logistiques

Deux delais fixes sont modelises:

- $d_{FG}$: delai fabricant -> grossiste;
- $d_{GP}$: delai grossiste -> pharmacie.

Les flux sont transportes au moyen de files de delai. Cela permet de representer l'inertie logistique de la chaine:

$$
A_t^{FG} = Q_{t-d_{FG}}
$$

$$
A_t^{GP} = D_{t-d_{GP}}
$$

### 4.7. Grossiste repartiteur

Le grossiste recoit le flux fabricant avec retard, puis sert la demande arrivee a la semaine courante. Son stock suit la relation:

$$
S_t^G = S_{t-1}^G + A_t^{FG} - L_t^G
$$

ou $L_t^G$ est la quantite servie a la pharmacie.

Le grossiste est considere en rupture lorsque son stock passe sous un seuil critique. Dans la version actuelle, ce seuil est plus bas que dans la version initiale, afin de signaler un vrai decrochage plutot qu'un simple niveau de stock modere.

### 4.8. Pharmacie

La pharmacie recoit la livraison du grossiste puis vend au patient final. Son stock evolue selon:

$$
S_t^P = S_{t-1}^P + A_t^{GP} - V_t
$$

avec:

$$
V_t = \min(S_t^P, D_t)
$$

La demande non servie vaut:

$$
M_t = D_t - V_t
$$

et le taux de service est:

$$
\tau_t =
\begin{cases}
1 & \text{si } D_t \le 0 \\
\frac{V_t}{D_t} & \text{sinon}
\end{cases}
$$

La pharmacie est en rupture si son stock devient nul ou negatif.

---

## 5. Construction du jeu de donnees

La simulation n'est pas seulement un outil d'illustration. Elle produit aussi un dataset tabulaire exploitable pour le machine learning.

### 5.1. Variables enregistrees

Pour chaque periode simulee, le projet enregistre notamment:

- la periode;
- la capacite du fabricant;
- le signal de demande lisse du fabricant;
- l'etat de disruption;
- la duree restante de disruption;
- la demande patient;
- la demande non servie;
- le taux de service;
- le stock du grossiste;
- le stock de la pharmacie;
- les indicateurs de rupture.

Ces variables capturent a la fois:

- l'etat instantane du systeme;
- les signes de fragilite;
- les consequences operationnelles d'un incident amont.

### 5.2. Cible de classification

La cible du modele ML est une rupture future a 4 semaines. Formellement:

$$
y_t = \mathbb{1}\left(\exists k \in \{1,\dots,4\},\; \mathbb{1}_{rupture,t+k}^G = 1\right)
$$

Le probleme est ainsi transforme en classification binaire:

- $y_t = 1$: une rupture du grossiste est attendue dans les 4 semaines;
- $y_t = 0$: aucune rupture n'est detectee a cet horizon.

### 5.3. Generation Monte Carlo

Pour diversifier les trajectoires, la simulation est repetee sur plusieurs graines aleatoires. Les simulations sont concatenees afin d'obtenir un dataset global:

$$
\mathcal{D} = \bigcup_{i=1}^{N}\mathcal{D}_i
$$

Cette approche rend le jeu de donnees plus riche en configurations de demande, de disruptions et de transitions de rupture.

---

## 6. Modele de machine learning

### 6.1. Variables explicatives

Le modele apprend a partir de variables qui decrivent l'etat du systeme:

- capacite du fabricant;
- demande lissee du fabricant;
- disruption fabricant;
- duree restante de disruption;
- demande patient;
- demande non servie;
- taux de service;
- stock du grossiste;
- stock de la pharmacie;
- indicateurs de rupture.

### 6.2. Architecture d'apprentissage

Le projet privilegie `XGBoost` si la dependance est disponible. Ce choix est adapte a des donnees tabulaires heterogenees et a des relations non lineaires.

Si `xgboost` n'est pas installe, le projet bascule automatiquement vers un modele logistique simple implemente en numpy. Cette solution de secours permet de conserver un pipeline fonctionnel dans des environnements contraints.

### 6.3. Fonction de decision

Le modele renvoie:

- une probabilite de rupture future;
- une classe binaire associee a un seuil de 0.5.

On note:

$$
p_t = \mathbb{P}(y_t = 1 \mid x_t)
$$

et:

$$
\hat{y}_t = \mathbb{1}(p_t \ge 0.5)
$$

### 6.4. Mesures d'evaluation

Le rapport du modele est resume par:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Dans le contexte des ruptures, le recall est particulierement important, car manquer une rupture reelle est plus couteux qu'emettre une alerte un peu trop tot.

---

## 7. Structure logicielle et reproductibilite

### 7.1. Organisation du projet

L'organisation actuelle du projet se repartit comme suit:

- `main.py` centralise la simulation et l'export des donnees;
- `ml_utils.py` isole le traitement ML;
- `dashboard/app.py` fournit l'interface Streamlit;
- `app.py` sert de point d'entree pratique pour lancer le dashboard.

Cette separation facilite:

- la lecture du code;
- la maintenance;
- l'extension future;
- la reutilisation des fonctions de simulation.

### 7.2. Reproductibilite

La simulation utilise une graine aleatoire explicite:

$$
\text{seed} = s
$$

Cette graine permet de reproduire une trajectoire donnee, ce qui est utile a la fois pour le debug, pour la generation de datasets et pour les demonstrations de soutenance.

---

## 8. Lecture du dashboard Streamlit

Le dashboard est l'interface d'exploitation du projet. Il permet de generer les donnees, d'entrainer un modele et d'explorer les resultats.

### 8.1. Generation du dataset

Depuis la barre laterale, l'utilisateur peut definir:

- le nombre de simulations Monte Carlo;
- le nombre de periodes par simulation;
- la demande moyenne;
- la variabilite de la demande.

En cliquant sur **Generer le dataset**, la simulation produit un fichier CSV dans `data/`.

### 8.2. Entrainement du modele

Le bouton **Entrainer le modele** lance l'apprentissage sur le dataset courant. Le dashboard affiche ensuite:

- Accuracy;
- Precision;
- Recall;
- F1;
- les valeurs TP, TN, FP, FN.

Si `xgboost` est disponible, il est utilise en priorite. Sinon, le mode de secours logistique est applique automatiquement.

### 8.3. Analyse des simulations

L'onglet **Predictions sur simulation** permet de:

- selectionner une trajectoire Monte Carlo;
- visualiser la probabilite de rupture predite;
- comparer les stocks du grossiste et de la pharmacie;
- comparer la demande observee et le signal de production lisse.

Cette vue est importante pour comprendre visuellement la facon dont un incident amont se traduit dans l'etat du systeme.

#### Lecture du menu de simulation

Le menu deroulant **Simulation** liste les identifiants des trajectoires Monte Carlo disponibles dans le dataset. Chaque valeur correspond a une simulation complete, generee avec une graine differente. Le choix d'une simulation permet donc d'explorer une trajectoire precise parmi l'ensemble des scenarii simules.

#### Lecture du graphe de probabilite de rupture

Le graphe **Probabilite de rupture predite a 4 semaines** trace la variable `proba_rupture` produite par le modele.

- Axe horizontal: la periode, exprimee en **semaines**;
- Axe vertical: une **probabilite** comprise entre 0 et 1.

Interpretation:

- une valeur proche de 0 indique un risque faible de rupture a 4 semaines;
- une valeur proche de 1 indique un risque eleve;
- une hausse progressive de la courbe signale souvent une fragilisation de la chaine avant la rupture observable.

#### Lecture du graphe des stocks principaux

Le graphe **Stocks principaux** compare les stocks du grossiste et de la pharmacie.

- Axe horizontal: la periode, en **semaines**;
- Axe vertical: des **unites de stock**.

Interpretation:

- une baisse durable du stock du grossiste traduit une tension aval qui peut preceder une rupture;
- un stock pharmacie proche de zero signifie que le patient risque de ne plus etre servi;
- le stock grossiste est souvent l'indicateur le plus utile pour anticiper une rupture future.

#### Lecture du graphe demande et signal de production

Le graphe **Demande et signal de production** compare plusieurs courbes si elles sont disponibles dans le dataset:

- `demande_patient`: la demande reelle observee;
- `demande_prevue`: la prevision naive, si elle est enregistree;
- `demande_lissee_fabricant`: le signal lisse utilise par le fabricant.

- Axe horizontal: la periode, en **semaines**;
- Axe vertical: des **unites de demande ou de production**.

Interpretation:

- `demande_patient` represente le comportement reel du marche;
- `demande_prevue` donne une lecture lisse de la demande attendue;
- `demande_lissee_fabricant` montre comment le fabricant repond au signal qu'il recoit;
- si ce signal suit bien la demande reelle, la production s'ajuste mieux aux besoins;
- si le signal reagi avec retard ou trop fortement, on peut observer des ecarts de stock.

#### Lecture du tableau final

Le tableau en fin de page permet de comparer la prediction du modele avec l'etat reel de la simulation.

Les colonnes principales sont:

- `periode`: la semaine consideree;
- `proba_rupture`: la probabilite predite par le modele;
- `prediction_rupture`: la classe binaire finale, avec seuil a 0.5;
- `rupture_grossiste`: la rupture observee dans la simulation;
- `stock_grossiste` et `stock_pharmacie`: les niveaux de stock reels.

Ce tableau est utile pour verifier si le modele anticipe correctement les situations de tension et pour comprendre a quel moment la probabilite de rupture commence a augmenter.

### 8.4. Prediction manuelle

L'onglet **Prediction manuelle** permet de saisir un scenario artificiel ou de reprendre une ligne de simulation comme base.

Cette fonctionnalite est utile pour:

- tester des cas limites;
- comprendre l'effet d'une variable isolee;
- montrer l'usage concret du classifieur;
- preparer une demonstration en soutenance.

---

## 9. Interpretation des resultats

L'objectif n'est pas uniquement de produire des courbes, mais de comprendre le comportement de la chaine.

### 9.1. Robustesse de la chaine

Si le grossiste conserve un stock tampon suffisant malgre une disruption du fabricant, cela signifie que la chaine absorbe correctement le choc. Le stock joue alors son role d'amortisseur.

### 9.2. Signaux avant-coureurs

Les indicateurs les plus utiles pour anticiper une rupture sont:

- la baisse continue du stock grossiste;
- l'augmentation de la demande non servie;
- la baisse du taux de service;
- la presence d'une disruption fabricant;
- la divergence entre demande observee et signal de production lisse.

### 9.3. Lecture metier

Du point de vue operationnel:

- un grossiste tendu devient le point de fragilite principal;
- une pharmacie a stock nul ne protege plus le patient final;
- un modele utile doit detecter le risque avant la rupture visible.

---

## 10. Limites actuelles et perspectives

### 10.1. Limites actuelles

Le systeme actuel fournit une base solide, mais il conserve plusieurs limites:

- la simulation reste centree sur trois agents;
- la demande patient est volontairement simple;
- les politiques de commande restent peu elaborees;
- le modele de prediction est construit sur des donnees simulees;
- la comparaison entre plusieurs familles de modeles n'est pas encore integree.

### 10.2. Perspectives de travail

Les evolutions naturelles du projet sont:

1. enrichir la simulation avec d'autres agents;
2. complexifier la politique de reapprovisionnement;
3. comparer plusieurs modeles de prediction;
4. ajouter des analyses de sensibilite;
5. mieux distinguer les signaux precoces et les ruptures effectives;
6. rapprocher encore davantage la simulation du cadre theorique de reference.

---

## 11. Conclusion

Le projet propose une base coherente et reproductible pour etudier la dynamique des ruptures dans la chaine pharmaceutique francaise. La simulation multi-agent constitue le socle dynamique du systeme, tandis que le module de machine learning transforme les trajectoires simulees en un probleme predictif exploitable. Le dashboard Streamlit renforce la lisibilite du projet en permettant de generer les donnees, d'entrainer un modele et de visualiser les predictions.

Dans sa version actuelle, le depot repond deja a plusieurs objectifs majeurs:

- formaliser le probleme;
- simuler une chaine critique de la supply chain pharmaceutique;
- produire des donnees structurees;
- entrainer un modele de rupture;
- offrir une visualisation interpretable;
- conserver une base simple a faire evoluer.

Ce rapport peut donc servir de base methodologique pour la version finale du projet et pour la soutenance.

---

## Annexe A. Arborescence fonctionnelle

```text
.
|-- app.py
|-- main.py
|-- ml_utils.py
|-- dashboard/
|   `-- app.py
|-- data/
|-- images/
|-- logs/
`-- models/
```

## Annexe B. Placeholders de figures

Les figures suivantes peuvent etre integrees au rapport final:

- `images/resultats_simulation_v3.png`
- captures du dashboard Streamlit
- exemples de prediction sur simulation
- exemples de prediction manuelle
