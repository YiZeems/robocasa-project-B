# Planning deadline, runs et rapport — Projet RL RoboCasa

---

# 15. Deadline projet et stratégie de livraison

## Deadline

Le projet doit être terminé pour :

```text
Jeudi 7 mai 2026 à 23h59
```

À partir du lundi 4 mai 2026, il reste environ :

```text
3 jours et demi
```

L'objectif n'est donc pas de chercher l'entraînement parfait, mais de produire :

```text
1. Des runs exploitables.
2. Des résultats comparables.
3. Un rapport d'avancement solide.
4. Une analyse honnête des limites.
5. Une proposition claire d'amélioration.
```

---

# 16. Objectif final réaliste

À la deadline, il faut avoir :

```text
1. Un environnement RoboCasa fonctionnel sur la tâche ouvrir une porte.
2. Un run SAC principal terminé.
3. Un run PPO baseline terminé ou partiellement terminé.
4. Des courbes reward / success rate / episode length.
5. Une analyse de stagnation ou surentraînement.
6. Un rapport d'avancement propre avec protocole, résultats, limites et perspectives.
```

Le projet peut être considéré réussi même si l'agent n'est pas parfait, à condition de montrer :

```text
- une méthode rigoureuse ;
- des choix justifiés ;
- des métriques claires ;
- des résultats comparables ;
- une analyse honnête des échecs ou limites.
```

---

# 17. Priorité absolue avant la deadline

Ne pas partir sur des choix trop coûteux.

## À privilégier

```text
observations état bas niveau
+
reward dense
+
SAC principal
+
PPO baseline
```

## À éviter sauf obligation

```text
RGB pur
PPO depuis pixels
gros modèle vision
OpenPI / GR00T
trop de méthodes en parallèle
tuning massif d'hyperparamètres
```

---

# 18. Planning recommandé jusqu'au jeudi 7 mai

## Lundi 4 mai — Stabilisation et lancement SAC

### Objectif du jour

Avoir un run SAC qui tourne correctement.

### Tâches

```text
1. Vérifier que la tâche ouvrir une porte fonctionne.
2. Vérifier le reset.
3. Vérifier la condition de succès.
4. Vérifier que la reward augmente dans le bon sens.
5. Lancer un SAC debug 100k–300k steps.
6. Si le debug marche, lancer un SAC long 3M steps pendant la nuit.
```

### Run 0 — SAC debug

```yaml
algo: SAC
timesteps: 300_000
observation: state
eval_freq: 25_000
n_eval_episodes: 20
```

Critère pour continuer :

```text
reward moyen augmente
ou
success_rate commence à dépasser 5–10 %
```

Si après 300k steps tout reste à zéro, ne pas lancer de gros run. Il faut d'abord corriger :

```text
reward
observations
action space
reset distribution
success condition
```

### Run 1 — SAC principal

```yaml
algo: SAC
timesteps: 3_000_000
learning_rate: 3e-4
batch_size: 256
buffer_size: 1_000_000
gamma: 0.99
tau: 0.005
ent_coef: auto
eval_freq: 25_000
n_eval_episodes: 50
```

Temps estimé :

```text
2–10 h si observations état
5–16 h si simulation plus lente
```

---

## Mardi 5 mai — Analyse SAC et lancement PPO

### Objectif du jour

Obtenir un premier résultat exploitable et lancer une baseline PPO.

### Analyse du run SAC

Regarder :

```text
- success rate train
- success rate validation
- reward moyen
- episode length
- door angle final
- meilleur checkpoint
- moment de stagnation
```

### Décision après SAC

| Résultat SAC | Action |
|---|---|
| Success > 60–70 % | Garder ce run comme résultat principal |
| Success 20–60 % | Tuner légèrement SAC |
| Success < 10 % | Corriger reward / observation / action space |
| Reward stable à zéro | Problème environnement ou reward |
| Train haut mais validation bas | Surentraînement / manque de généralisation |

### Run 2 — SAC variante tuning

Si SAC apprend mais pas assez :

```yaml
algo: SAC
timesteps: 2_000_000
learning_rate: 1e-4
batch_size: 512
buffer_size: 1_000_000
gamma: 0.99
ent_coef: auto
```

ou :

```yaml
algo: SAC
timesteps: 2_000_000
learning_rate: 3e-4
batch_size: 512
ent_coef: auto_0.2
```

Ne pas faire plus de deux variantes SAC avant la deadline.

### Run 3 — PPO baseline

À lancer mardi après-midi ou mardi soir :

```yaml
algo: PPO
timesteps: 5_000_000
n_envs: 16
n_steps: 1024
batch_size: 256
n_epochs: 10
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
eval_freq: 25_000
n_eval_episodes: 50
```

Temps estimé :

```text
4–16 h
```

Si PPO est trop lent :

```yaml
timesteps: 3_000_000
n_envs: 8
```

---

## Mercredi 6 mai — Finalisation des résultats

### Objectif du jour

Avoir les résultats définitifs ou quasi définitifs.

### Méthodes à comparer

```text
SAC principal
SAC variante
PPO baseline
```

### Critères de comparaison

```text
- meilleur validation success rate
- vitesse d'apprentissage
- stabilité
- reward final
- généralisation sur seeds non vus
- nombre de steps avant stagnation
- temps de run
```

### Tableau de résultats à produire

| Méthode | Steps | Best success validation | Step du meilleur checkpoint | Reward final | Stagnation | Temps de run |
|---|---:|---:|---:|---:|---|---:|
| SAC | 3M | XX % | X steps | X | oui/non | X h |
| SAC tuned | 2M | XX % | X steps | X | oui/non | X h |
| PPO | 3M/5M | XX % | X steps | X | oui/non | X h |

### Graphiques minimaux

Produire au moins :

```text
1. Reward moyen vs steps
2. Success rate validation vs steps
3. Episode length vs steps
4. Comparaison SAC/PPO
```

Si possible :

```text
5. Door opening angle final vs steps
```

---

## Jeudi 7 mai — Rapport et rendu

### Objectif du jour

Ne plus lancer de gros run après jeudi matin.

Jeudi doit être réservé à :

```text
- finalisation du rapport ;
- insertion des graphes ;
- écriture de l'analyse ;
- formulation des limites ;
- conclusion ;
- vérification de la reproductibilité ;
- export final.
```

Lancer seulement de petits runs de confirmation si nécessaire :

```text
100k–300k steps maximum
```

Ne pas lancer de nouveau gros run jeudi après-midi.

---

# 19. Plan de runs optimal

## Version réaliste et propre

```text
Run 0 : SAC debug — 300k steps
Run 1 : SAC principal — 3M steps
Run 2 : SAC tuned — 2M steps
Run 3 : PPO baseline — 3M à 5M steps
```

C'est suffisant pour un bon rapport.

---

## Version si le projet prend du retard

Si l'environnement prend trop de temps à stabiliser :

```text
Run 0 : SAC debug — 100k
Run 1 : SAC principal — 1M à 2M
Run 2 : PPO court — 1M à 2M
```

Dans le rapport, présenter cela comme :

```text
résultats préliminaires
+
analyse d'avancement
+
protocole prévu pour runs longs
```

---

## Version si tout fonctionne très bien

Si SAC marche dès le lundi :

```text
Run 1 : SAC 3M
Run 2 : SAC tuned 3M
Run 3 : PPO 5M
Run 4 : BC + SAC si démonstrations disponibles
```

Mais seulement si les démonstrations sont déjà propres et directement exploitables.

---

# 20. Structure du rapport final

## Plan recommandé

```text
1. Introduction
2. Objectif du projet
3. Environnement RoboCasa
4. Tâche atomique : ouvrir une porte
5. Méthodes testées
6. Reward shaping
7. Hyperparamètres
8. Protocole expérimental
9. Résultats
10. Analyse
11. Surentraînement / stagnation
12. Limites
13. Perspectives
14. Conclusion
```

---

## Message principal à défendre

```text
La méthode SAC a été choisie comme méthode principale car elle est adaptée au contrôle continu et plus sample-efficient que PPO. PPO a été utilisé comme baseline stable. La performance a été évaluée non seulement par le reward, mais surtout par le success rate sur des seeds de validation non vues. Le surentraînement est détecté par l'écart entre train success et validation success, ainsi que par la stagnation du validation success malgré l'augmentation du reward d'entraînement.
```

---

# 21. Hyperparamètres finaux à utiliser

## SAC principal

```yaml
algo: SAC
total_timesteps: 3_000_000
learning_rate: 3e-4
buffer_size: 1_000_000
batch_size: 256
gamma: 0.99
tau: 0.005
train_freq: 1
gradient_steps: 1
learning_starts: 10_000
ent_coef: auto
policy_network: [256, 256]
q_network: [256, 256]
eval_freq: 25_000
n_eval_episodes: 50
```

---

## SAC tuned

```yaml
algo: SAC
total_timesteps: 2_000_000
learning_rate: 1e-4
buffer_size: 1_000_000
batch_size: 512
gamma: 0.99
tau: 0.005
ent_coef: auto
eval_freq: 25_000
n_eval_episodes: 50
```

---

## PPO baseline

```yaml
algo: PPO
total_timesteps: 5_000_000
n_envs: 16
n_steps: 1024
batch_size: 256
n_epochs: 10
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
eval_freq: 25_000
n_eval_episodes: 50
```

---

# 22. Critères de décision

## Si SAC marche bien

```text
SAC devient méthode principale.
PPO sert de baseline comparative.
Conclusion : SAC est plus efficace sur cette tâche de contrôle continu.
```

---

## Si PPO marche mieux

```text
PPO devient meilleure méthode empirique dans cette configuration.
Conclusion : PPO est plus stable pour cette configuration précise, même s'il est moins sample-efficient théoriquement.
```

---

## Si aucune méthode ne marche bien

Ce n'est pas forcément un échec.

Conclusion possible :

```text
Les résultats indiquent que la difficulté principale vient probablement de l'exploration, d'un reward shaping insuffisant, d'une représentation d'état incomplète, ou d'une distribution de reset trop difficile. Le projet montre la nécessité d'utiliser des démonstrations, du curriculum learning ou une méthode imitation learning + RL fine-tuning.
```

---

# 23. À ne pas faire avant jeudi

Éviter absolument :

```text
- changer de tâche en cours de route ;
- passer au RGB si l'état fonctionne ;
- lancer 10 méthodes différentes ;
- tuner trop d'hyperparamètres ;
- attendre jeudi pour écrire le rapport ;
- lancer un run énorme jeudi après-midi ;
- juger uniquement avec le reward train ;
- présenter le dernier checkpoint au lieu du meilleur checkpoint validation.
```

---

# 24. Planning horaire conseillé

## Lundi soir

```text
18h–20h : debug environnement + reward
20h–22h : SAC debug 100k–300k
22h–nuit : SAC 3M
```

## Mardi

```text
Matin : analyse SAC
Après-midi : SAC tuned ou correction reward
Soir/nuit : PPO 3M–5M
```

## Mercredi

```text
Matin : analyse PPO
Après-midi : graphes + tableau comparatif
Soir : rédaction rapport 60–70 %
Nuit : dernier run court si nécessaire
```

## Jeudi

```text
Matin : finalisation résultats
Après-midi : rédaction finale
Soir : relecture, figures, conclusion, export PDF
Avant 23h59 : rendu
```

---

# 25. Estimation du temps de run sur RTX 4070 + i5-13600K + 64 Go RAM

## Hypothèses

```text
GPU : RTX 4070 12 Go
CPU : Intel i5-13600K
RAM : 64 Go DDR4
Tâche : ouvrir une porte
Environnement : RoboCasa / robosuite / MuJoCo
```

Le bottleneck est souvent :

```text
simulation MuJoCo
+
rendu caméra
+
nombre d'environnements parallèles
```

La RTX 4070 est utile pour les réseaux profonds, les CNN et les méthodes vision, mais pour les observations état bas niveau, le CPU et la simulation dominent souvent.

---

## Vitesses probables

| Configuration | Vitesse probable |
|---|---:|
| Observations état, sans rendu RGB lourd | 100–400 env steps/s |
| Observations RGB caméra | 10–60 env steps/s |
| RGB haute résolution + plusieurs caméras | 5–25 env steps/s |

---

## Temps par méthode

| Méthode | Observation | Steps / epochs jusqu'à stagnation | Temps estimé | Recommandation |
|---|---|---:|---:|---|
| SAC | État | 2M–5M steps | 5–16 h | Meilleur RL pur |
| PPO | État | 5M–10M steps | 8–30 h | Baseline stable |
| TD3 | État | 3M–8M steps | 8–30 h | À tester après SAC |
| SAC / DrQ-v2 | RGB | 5M–20M steps | 2–14 jours | Meilleur RL depuis pixels |
| PPO | RGB | 10M–20M+ steps | 4–15 jours | Pas recommandé |
| BC | État | 50–200 epochs | 30 min – 4 h | Excellent si démos |
| BC | RGB | 50–200 epochs | 3–24 h | Bon point de départ |
| Diffusion Policy | État/RGB | 50–300 epochs | 2 h – 4 jours | Très bon avec démos |
| BC + SAC | État | BC + 500k–2M RL steps | 2–8 h | Meilleur compromis |
| Diffusion + SAC/DrQ | RGB | prétrain + fine-tune | 1–5 jours | Meilleur si vision |

---

# 26. Recommandation finale pour la deadline

## Meilleur compromis temps / résultat

```text
1. SAC état — 3M steps
2. SAC tuned — 2M steps
3. PPO état — 3M à 5M steps
```

## Ne pas dépasser

```text
3 gros runs principaux
```

## Métrique principale

```text
validation_success_rate
```

## Checkpoint final

```text
meilleur checkpoint validation
et non le dernier checkpoint
```

---

# 27. Résumé ultra-opérationnel

```text
Deadline : jeudi 7 mai 2026, 23h59

Méthode principale : SAC
Baseline : PPO
Observation : état bas niveau
Runs principaux : 2 ou 3 maximum
Pas de RGB sauf obligation
Pas de gros modèle
Pas de tuning excessif

Run 0 : SAC debug 100k–300k
Run 1 : SAC principal 3M
Run 2 : SAC tuned 2M
Run 3 : PPO 3M–5M

Évaluation : toutes les 25k steps
Métrique principale : validation success rate
Checkpoint final : meilleur checkpoint validation
Surentraînement : train success haut mais validation/test stagnant ou en baisse

Rapport :
- protocole
- méthodes
- hyperparamètres
- reward shaping
- résultats
- courbes
- stagnation / surentraînement
- limites
- perspectives
```
