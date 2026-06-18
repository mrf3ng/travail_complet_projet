# Rapport scientifique

## Titre

**Modélisation multi-agent de la chaîne pharmaceutique française et prévision des ruptures de stock par machine learning**

---

## Résumé

Ce projet étudie la dynamique des ruptures de stock dans une chaîne pharmaceutique en combinant deux approches complémentaires :

1. une **modélisation multi-agent** de la chaîne d’approvisionnement ;
2. une **chaîne de prévision supervisée** visant à détecter précocement les ruptures.

Le travail actuel implémente une simulation de la chaîne **fabricant -> grossiste répartiteur -> pharmacie**, avec prise en compte des délais logistiques, de la variabilité de la demande patient, et des interruptions de production du fabricant. À partir de cette simulation, un ensemble de données est généré pour entraîner un modèle de classification binaire capable de prédire une rupture à horizon de quatre semaines.

L’application finale comprend un **dashboard Streamlit** qui permet de générer les données, d’entraîner le modèle, de charger un modèle sauvegardé, d’explorer des simulations et de tester des scénarios manuellement. Le projet fournit ainsi une base expérimentale cohérente pour analyser la résilience de la chaîne et la capacité d’un modèle ML à anticiper les ruptures.

---

## 1. Contexte scientifique et industriel

Les ruptures de médicaments constituent un problème croissant pour les systèmes de santé. Elles affectent l’accès aux traitements, dégradent la continuité des soins et imposent des adaptations coûteuses à l’ensemble des acteurs de la chaîne.

Dans le cas français, la structure de distribution présente une particularité importante : le **grossiste répartiteur** est un maillon obligatoire entre le fabricant et les pharmacies de ville. Ce point est central, car une défaillance au niveau du grossiste peut se propager immédiatement à plusieurs pharmacies simultanément.

Le projet s’inscrit dans un cadre de recherche appliquée sur :

- la compréhension des mécanismes de propagation des ruptures ;
- la modélisation de la chaîne d’approvisionnement ;
- l’anticipation des ruptures par apprentissage automatique ;
- l’aide à la décision via un outil de visualisation interactif.

---

## 2. Objectifs du projet

Les objectifs opérationnels sont les suivants :

1. **modéliser la chaîne pharmaceutique** sous forme d’agents interagissants ;
2. **simuler les flux physiques et informationnels** sur plusieurs semaines ;
3. **générer des données structurées** exploitables pour un modèle de machine learning ;
4. **entraîner un classifieur** de rupture à horizon court ;
5. **visualiser et interpréter** les résultats dans un dashboard ;
6. **préparer une base scientifique** claire, reproductible et présentable en soutenance.

---

## 3. Périmètre de la version actuelle

La structure logicielle actuelle du projet est organisée de manière modulaire :

- `main.py` : simulation, génération des datasets, export des graphiques ;
- `ml_utils.py` : préparation des données, entraînement, sauvegarde et prédiction ;
- `dashboard/app.py` : application Streamlit interactive ;
- `app.py` : point d’entrée du dashboard ;
- `data/` : datasets générés ;
- `images/` : figures de simulation ;
- `models/` : modèle entraîné sauvegardé ;
- `logs/` : journalisation du dashboard.

Cette architecture permet de distinguer clairement :

- la logique de simulation ;
- la logique d’apprentissage ;
- la logique d’interface utilisateur.

---

## 4. Modélisation multi-agent

### 4.1. Agents considérés

Dans la version actuellement implémentée, la chaîne simulée comporte trois agents principaux :

- **Fabricant** : produit le médicament ;
- **Grossiste répartiteur** : reçoit la production, gère un stock intermédiaire et sert la pharmacie ;
- **Pharmacie** : vend au patient final.

Le cadre conceptuel plus large du projet vise une chaîne pharmaceutique complète, mais la version courante constitue un noyau robuste centré sur le segment le plus critique pour la propagation des ruptures.

### 4.2. Demande patient

La demande hebdomadaire du patient est modélisée par une variable aléatoire tronquée à zéro :

$$
D_t = \max(0, \mathcal{N}(D_0, \sigma^2))
$$

où :

- \(D_t\) est la demande observée à la semaine \(t\) ;
- \(D_0\) est la demande moyenne ;
- \(\sigma\) est l’écart-type de la demande.

Cette formulation capture l’idée d’une consommation fluctuante, tout en évitant les valeurs négatives non physiques.

### 4.3. Prévision naïve de la demande

La simulation utilise une prévision simple à partir d’une fenêtre glissante :

$$
\hat{D}_t =
\begin{cases}
D_0 & \text{si aucun historique n’est disponible} \\
\frac{1}{m}\sum_{i=1}^{m} D_{t-i} & \text{sinon}
\end{cases}
$$

avec \(m\) la taille de l’historique utilisé.

Cette prévision joue un double rôle :

- elle rend le comportement d’achat plus réaliste ;
- elle alimente la construction des variables explicatives du dataset.

### 4.4. Production du fabricant

Le fabricant dispose d’une capacité nominale \(C_{\text{nom}}\). Sa production théorique est ajustée à la demande observée :

$$
C_t^{\text{th}} = \min(1.1 \cdot \hat{D}_t,\; C_{\text{nom}})
$$

Le facteur \(1.1\) introduit une marge de sécurité modérée.

En cas de disruption, la production effective est réduite :

$$
C_t =
\begin{cases}
C_t^{\text{th}} & \text{si aucune disruption n’est active} \\
(1-\delta) \, C_t^{\text{th}} & \text{si une disruption est active}
\end{cases}
$$

où \(\delta\) représente la perte de capacité liée à l’incident.

### 4.5. Modélisation de la disruption

La disruption du fabricant est modélisée comme un processus stochastique de type Bernoulli :

$$
Z_t \sim \mathcal{B}(\lambda)
$$

avec :

- \(Z_t = 1\) si une disruption débute à la semaine \(t\) ;
- \(\lambda\) la probabilité hebdomadaire de disruption.

Une fois activée, la disruption dure un nombre aléatoire de semaines \(L\) :

$$
L \sim \mathcal{U}\{L_{\min}, L_{\max}\}
$$

La variable d’état de la disruption évolue ensuite avec un compteur de durée restante.

### 4.6. Délais logistiques

Deux délais sont pris en compte :

- \(d_{FG}\) : délai fabricant -> grossiste ;
- \(d_{GP}\) : délai grossiste -> pharmacie.

Les flux reçus à la semaine \(t\) correspondent donc à des envois antérieurs :

$$
A_t^{FG} = C_{t-d_{FG}}
$$

$$
A_t^{GP} = Q_{t-d_{GP}}
$$

où \(Q_t\) est la quantité expédiée par le grossiste vers la pharmacie.

Cette représentation par files de retard permet de simuler les effets d’inertie propres à une chaîne d’approvisionnement.

### 4.7. Grossiste répartiteur

Le grossiste joue le rôle d’amortisseur entre la production et la demande de détail. Son stock évolue selon :

$$
S_t^{G} = S_{t-1}^{G} + A_t^{FG} - Q_t
$$

La quantité servie au système aval est limitée par le stock disponible :

$$
Q_t = \min(S_t^{G}, \; \text{demande reçue})
$$

Le grossiste est déclaré en rupture lorsque son stock passe sous un seuil critique :

$$
\mathbb{1}_{\text{rupture},t}^{G} =
\begin{cases}
1 & \text{si } S_t^{G} < s_G^{\text{crit}} \\
0 & \text{sinon}
\end{cases}
$$

### 4.8. Pharmacie

La pharmacie reçoit les livraisons du grossiste, puis sert la demande patient. Son stock suit :

$$
S_t^{P} = S_{t-1}^{P} + A_t^{GP} - V_t
$$

avec :

$$
V_t = \min(S_t^{P}, D_t)
$$

La demande non servie vaut :

$$
M_t = D_t - V_t
$$

et le taux de service est donné par :

$$
\tau_t =
\begin{cases}
1 & \text{si } D_t \le 0 \\
\frac{V_t}{D_t} & \text{sinon}
\end{cases}
$$

La pharmacie est en rupture si son stock devient nul ou négatif :

$$
\mathbb{1}_{\text{rupture},t}^{P} = \mathbb{1}(S_t^{P} \le 0)
$$

---

## 5. Construction du jeu de données

La simulation n’est pas seulement un outil d’illustration. Elle sert aussi à produire un jeu de données tabulaire exploitable pour le machine learning.

### 5.1. Variables enregistrées

Pour chaque semaine simulée, le projet enregistre notamment :

- la période ;
- la capacité du fabricant ;
- l’état de disruption et la durée restante ;
- la demande patient ;
- la demande prévue ;
- la commande de la pharmacie ;
- la commande du grossiste ;
- les livraisons reçues ;
- la demande non servie ;
- le taux de service ;
- les stocks du grossiste et de la pharmacie ;
- les stocks en transit ;
- les indicateurs de rupture.

Ces variables reflètent à la fois :

- l’état du système ;
- les signaux d’alerte ;
- les conséquences des perturbations.

### 5.2. Cible de classification

La variable cible du modèle ML est définie comme une rupture future à 4 semaines. Formellement :

$$
y_t = \mathbb{1}\left(\exists k \in \{1,\dots,4\},\; \mathbb{1}_{\text{rupture},t+k}^{G}=1\right)
$$

Cette formulation transforme le problème en classification binaire :

- \(y_t = 1\) : une rupture du grossiste est attendue dans les 4 semaines ;
- \(y_t = 0\) : aucune rupture n’est détectée à cet horizon.

### 5.3. Génération Monte Carlo

Pour diversifier les scénarios, la simulation est répétée plusieurs fois avec des graines différentes. Le dataset final est obtenu par concaténation :

$$
\mathcal{D} = \bigcup_{i=1}^{N} \mathcal{D}_i
$$

où \(N\) est le nombre de simulations Monte Carlo.

Cette approche augmente la variété des trajectoires observées et améliore la robustesse de l’entraînement.

---

## 6. Modèle de machine learning

### 6.1. Variables explicatives

Le modèle apprend à partir d’un ensemble de variables explicatives représentant l’état du système :

- capacité du fabricant ;
- disruption fabricant ;
- durée restante de disruption ;
- demande patient ;
- demande prévue ;
- commandes ;
- livraisons ;
- demande non servie ;
- taux de service ;
- stocks ;
- indicateurs de rupture.

Ces variables sont cohérentes avec le problème métier, car elles décrivent à la fois la pression de la demande, l’état de l’offre et les signes de fragilité de la chaîne.

### 6.2. Architecture d’apprentissage

Le projet privilégie `XGBoost` lorsque la dépendance est disponible. Ce choix est pertinent pour :

- des données tabulaires ;
- des relations non linéaires ;
- une bonne performance sur des signaux hétérogènes ;
- une interprétabilité relative via l’importance des variables.

En solution de secours, un modèle logistique simple est prévu afin de conserver un pipeline fonctionnel même dans un environnement léger.

### 6.3. Fonction de décision

Le classifieur renvoie une probabilité de rupture :

$$
p_t = \mathbb{P}(y_t=1 \mid x_t)
$$

et une décision binaire :

$$
\hat{y}_t = \mathbb{1}(p_t \geq 0.5)
$$

où \(x_t\) désigne le vecteur de variables explicatives à la semaine \(t\).

### 6.4. Mesures d’évaluation

Le rapport du modèle est évalué par :

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

avec :

- \(TP\) : vrais positifs ;
- \(TN\) : vrais négatifs ;
- \(FP\) : faux positifs ;
- \(FN\) : faux négatifs.

Dans le contexte de la rupture de stock, le **recall** est particulièrement important : manquer une rupture réelle est souvent plus pénalisant qu’annoncer une alerte un peu trop tôt.

---

## 7. Structure logicielle et reproductibilité

### 7.1. Organisation du projet

L’organisation actuelle facilite la maintenance et la compréhension du code :

- `main.py` centralise la simulation et la génération des données ;
- `ml_utils.py` isole le bloc de machine learning ;
- `dashboard/app.py` encapsule l’interface utilisateur ;
- `app.py` sert de point d’entrée pratique pour Streamlit.

Cette séparation des responsabilités améliore :

- la lisibilité ;
- la testabilité ;
- la réutilisation ;
- la capacité à faire évoluer le projet.

### 7.2. Reproductibilité

La simulation utilise une graine aléatoire contrôlée :

$$
\text{seed} = s
$$

Ce mécanisme permet de reproduire les trajectoires de simulation et les résultats associés, ce qui est essentiel dans un cadre scientifique.

---

## 8. Lecture du dashboard Streamlit

Le dashboard constitue l’interface d’exploitation du projet. Il est conçu pour permettre une lecture intuitive des simulations et des prédictions.

### 8.1. Génération du dataset

Depuis la barre latérale, l’utilisateur peut définir :

- le nombre de simulations Monte Carlo ;
- le nombre de périodes par simulation ;
- la demande moyenne ;
- la variabilité de la demande.

En cliquant sur **Générer le dataset**, le système lance la simulation et sauvegarde les données dans `data/dataset_xgboost.csv`.

**Interprétation scientifique** :
- un dataset plus large accroît la diversité des situations vues à l’entraînement ;
- les extrêmes de demande sont mieux représentés ;
- la capacité de généralisation du modèle peut s’améliorer.

![Capture - Génération du dataset](images/placeholder_dashboard_generation.png)

### 8.2. Entraînement du modèle

Le bouton **Entraîner le modèle** lance l’apprentissage sur le dataset disponible. Le dashboard affiche ensuite :

- `Accuracy`
- `Precision`
- `Recall`
- `F1`
- ainsi que les comptes \(TP, TN, FP, FN\).

**Interprétation** :
- une `Accuracy` élevée indique une bonne performance globale ;
- une `Precision` élevée signifie peu de fausses alertes ;
- un `Recall` élevé signifie que les ruptures réelles sont bien détectées ;
- un bon `F1` traduit un compromis équilibré entre précision et sensibilité.

![Capture - Métriques d’entraînement](images/placeholder_dashboard_metrics.png)

### 8.3. Analyse des simulations

L’onglet **Predictions sur simulation** permet de :

- sélectionner une simulation Monte Carlo ;
- visualiser la probabilité de rupture prédite ;
- comparer les ruptures observées et les stocks.

Les courbes présentées sont essentielles pour interpréter la dynamique du système.

#### Courbe de probabilité de rupture

Cette courbe montre \(p_t\), la probabilité préditе de rupture à 4 semaines.

**Lecture** :
- une montée progressive de la probabilité signale une fragilisation du système ;
- un pic de probabilité peut précéder une baisse de stock ou une propagation d’un incident amont ;
- une probabilité durablement faible indique un système stable.

![Capture - Probabilité de rupture](images/placeholder_dashboard_proba.png)

#### Courbe des stocks

Le dashboard affiche les stocks du grossiste et de la pharmacie.

**Lecture** :
- si le stock du grossiste décroît fortement, le risque de rupture augmente ;
- si le stock de la pharmacie chute à zéro, la demande patient n’est plus servie ;
- si le grossiste conserve un stock tampon suffisant, le système absorbe mieux les chocs.

![Capture - Stocks et transit](images/placeholder_dashboard_stocks.png)

#### Courbe demande observée vs demande prévue

Cette visualisation compare la demande réelle à la prévision naïve.

**Lecture** :
- si la demande réelle s’écarte fortement de la prévision, le système peut devenir moins bien approvisionné ;
- plus la courbe prévue suit la demande réelle, plus la politique de commande est cohérente.

![Capture - Demande observée et prévue](images/placeholder_dashboard_demandes.png)

### 8.4. Prédiction manuelle

L’onglet **Prediction manuelle** permet de saisir un scénario artificiel en choisissant les valeurs des variables explicatives.

Cette fonctionnalité est utile pour :

- tester des cas extrêmes ;
- comprendre l’influence d’une variable isolée ;
- simuler une situation métier particulière ;
- démontrer l’usage du modèle en soutenance.

**Interprétation** :
- si les stocks sont faibles et qu’une rupture est déjà visible, la probabilité de rupture future augmente mécaniquement ;
- si la demande est élevée mais la capacité et les stocks restent confortables, la probabilité doit rester plus faible.

![Capture - Prédiction manuelle](images/placeholder_dashboard_manual.png)

---

## 9. Interprétation des résultats

L’objectif n’est pas uniquement de produire des courbes, mais de comprendre le fonctionnement du système.

### 9.1. Robustesse de la chaîne

Si la chaîne maintient un taux de service proche de 1 malgré une perturbation du fabricant, cela signifie que :

- le stock intermédiaire joue son rôle d’amortisseur ;
- les délais logistiques sont absorbés ;
- la politique de réapprovisionnement est suffisante.

### 9.2. Signaux avant-coureurs

Les signaux les plus utiles pour anticiper une rupture sont :

- la baisse continue du stock grossiste ;
- la hausse de la demande non servie ;
- la diminution du taux de service ;
- l’activation d’une disruption fabricant ;
- la déviation entre demande observée et demande prévue.

### 9.3. Lecture métier

Du point de vue opérationnel :

- un grossiste fortement sollicité peut devenir le point de fragilité principal ;
- une pharmacie avec stock nul ne peut plus protéger le patient final ;
- un modèle prédictif utile doit détecter la rupture avant la rupture visible.

---

## 10. Limites actuelles et perspectives

### 10.1. Limites actuelles

Le système actuel fournit une base solide, mais il conserve plusieurs limites :

- la simulation couvre principalement le sous-système fabricant-grossiste-pharmacie ;
- les données utilisées pour le ML sont issues de la simulation, et non encore d’une base réelle fusionnée ;
- la prévision de demande reste volontairement simple ;
- le modèle de classification reste centré sur une rupture à horizon 4 semaines ;
- la comparaison explicite entre plusieurs familles de modèles n’est pas encore intégrée dans l’application.

### 10.2. Perspectives de travail

Les évolutions naturelles du projet sont :

1. enrichir la simulation avec d’autres agents de la chaîne ;
2. intégrer progressivement des données réelles ;
3. comparer plusieurs modèles de prévision et de classification ;
4. améliorer la stratégie de commande ;
5. ajouter des métriques de robustesse et de coût ;
6. développer des analyses de sensibilité ;
7. utiliser le dashboard comme support de décision.

---

## 11. Conclusion

Le projet propose une approche cohérente et progressive de la modélisation de la chaîne pharmaceutique. La simulation multi-agent constitue le socle dynamique du système, tandis que le module de machine learning transforme les trajectoires simulées en un problème prédictif exploitable. L’interface Streamlit renforce la lisibilité du projet en permettant d’observer en temps réel les comportements du système, d’entraîner un modèle et de tester des scénarios.

En l’état, le projet répond déjà à plusieurs attentes majeures d’un travail de recherche appliquée :

- formalisation du problème ;
- modélisation mathématique ;
- génération et structuration des données ;
- apprentissage supervisé ;
- visualisation interprétable ;
- reproductibilité et mise en valeur des résultats.

Ce rapport peut ainsi servir de base scientifique et méthodologique pour la soutenance finale.

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

Les figures ci-dessous sont prévues pour l’intégration finale dans le rapport :

- `images/placeholder_dashboard_generation.png`
- `images/placeholder_dashboard_metrics.png`
- `images/placeholder_dashboard_proba.png`
- `images/placeholder_dashboard_stocks.png`
- `images/placeholder_dashboard_demandes.png`
- `images/placeholder_dashboard_manual.png`
