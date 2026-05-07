# Rapport de projet - Apprentissage par renforcement pour RoboCasa

- **Module :** IA705 - Apprentissage pour la robotique
- **Formation :** Mastère Spécialisé Intelligence Artificielle Multimodale, Telecom Paris
- **Projet :** ouverture d'une porte de placard avec PandaOmron dans RoboCasa
- **Date :** 7 mai 2026
- **Dépôt :** https://github.com/YiZeems/robocasa-project-B

## 1. Introduction

Ce projet étudie une tâche de manipulation robotique simple à décrire mais difficile à apprendre : ouvrir une porte de placard dans RoboCasa. L'agent part de zéro, sans démonstration, et doit apprendre une politique de contrôle continue pour le bras Panda d'un robot PandaOmron.

L'objectif initial était de comparer SAC et PPO sur une tâche atomique RoboCasa, puis d'identifier les signes de surentraînement. En pratique, le projet a surtout mis en évidence une difficulté plus fondamentale : avant même de comparer les algorithmes, il faut obtenir un signal d'apprentissage fiable sur une tâche de contact. Les runs réalisés montrent les échecs successifs, les corrections apportées, puis l'amélioration obtenue avec HER.

La question de recherche peut donc être formulée ainsi :

> Comment stabiliser l'apprentissage d'une politique SAC sur une tâche d'ouverture de porte RoboCasa, malgré une reward rare, un espace d'action continu et des risques de reward hacking ?

## 2. Environnement et tâche

RoboCasa est un benchmark de manipulation en cuisine construit sur RoboSuite et MuJoCo. Il propose des scènes variées, des objets procéduraux et plusieurs tâches atomiques, dont `OpenCabinet`.

Dans ce projet, le robot est PandaOmron : un bras Franka Panda monté sur une base mobile Omron. La base n'est pas contrôlée. Elle reste fixe, face au meuble, afin de réduire le problème à la manipulation du bras. Ce choix garde le projet dans le périmètre d'une tâche atomique et évite de mélanger navigation et manipulation.

| Élément | Valeur utilisée |
|---|---|
| Tâche RoboCasa | `OpenCabinet` / ouverture de porte |
| Robot | PandaOmron, base fixe |
| Simulateur | MuJoCo via RoboSuite et RoboCasa |
| Horizon | 500 steps |
| Fréquence de contrôle | 10 Hz |
| Observations | État bas niveau, pas d'image RGB |
| Actions | Contrôle continu du bras |
| Succès strict | porte ouverte au-delà du seuil défini dans la config |

Les observations RGB n'ont pas été utilisées. Elles auraient ajouté un problème de représentation visuelle à un projet déjà limité par l'exploration et le contact. Le choix d'observations d'état rend l'expérience plus lisible pour analyser les algorithmes RL.

## 3. Formulation RL

La tâche est modélisée comme un MDP :

| Composant | Définition dans le projet |
|---|---|
| État | positions/vitesses articulaires, pose de l'effecteur, distance à la poignée, angle de porte |
| Action | commande continue du bras Panda |
| Transition | dynamique physique MuJoCo |
| Reward | reward façonnée, puis variante HER |
| Horizon | 500 steps |
| Discount | `gamma = 0.99` |

Le critère important n'est pas la reward cumulée seule. Une politique peut obtenir une reward correcte en restant proche de la poignée sans ouvrir la porte. Le taux de succès et l'angle réel de la porte sont donc les métriques qui décident si la politique résout vraiment la tâche.

## 4. Méthodes

### SAC

SAC a été choisi comme méthode principale car il est adapté aux espaces d'action continus et réutilise les transitions via un replay buffer. Cette propriété est précieuse en robotique simulée : chaque trajectoire coûte du temps, et le signal utile peut être rare.

Les premières configurations SAC utilisaient l'auto-tuning de l'entropie. Ce choix, théoriquement confortable, s'est révélé instable ici : le coefficient d'entropie `alpha` chutait vers zéro, la politique devenait trop déterministe et les critics divergeaient.

La correction décisive a été de fixer `ent_coef = 0.1`. Cette modification n'a pas suffi à résoudre la tâche, mais elle a rendu l'entraînement beaucoup plus sain : actor loss stable, critic loss contrôlée et exploration maintenue.

### PPO

PPO était prévu comme baseline comparative. Il est plus stable et plus simple à régler, mais moins sample-efficient car il est on-policy. Dans l'état actuel des résultats, la comparaison SAC/PPO n'est pas encore vraiment concluante : aucun run n'atteint un succès strict stable. Comparer finement les deux algorithmes avant d'obtenir au moins une politique qui réussit serait prématuré.

### HER

HER, Hindsight Experience Replay, a été introduit après les échecs du reward shaping dense seul. L'idée est de relabelliser des trajectoires échouées comme si l'objectif avait été l'angle de porte effectivement atteint. Ainsi, même une petite ouverture peut devenir un exemple positif pour le critic.

HER v1, avec reward sparse pure, n'a pas suffi : l'agent ne recevait aucun guidage pour aller vers la poignée. HER v2, en revanche, combine HER avec une reward dense modérée. C'est le premier run qui produit une ouverture visible.

## 5. Reward shaping

La reward naïve favorise l'approche de la poignée et le succès final. Sur cette tâche, cela crée un risque immédiat : l'agent peut apprendre à rester près de la poignée sans agir sur la porte. On observe alors une reward correcte mais un taux de succès nul.

La reward façonnée ajoute plusieurs composantes :

```text
R = approach + progress + success
    - action_reg
    - stagnation
    - wrong_dir
    - oscillation
```

| Composante | Rôle |
|---|---|
| `approach` | guider l'effecteur vers la poignée |
| `progress` | récompenser l'augmentation de l'angle de porte |
| `success` | donner un bonus clair lorsque le seuil est atteint |
| `action_reg` | éviter des actions trop brutales |
| `stagnation` | pénaliser le fait de rester proche sans progresser |
| `wrong_dir` | éviter de refermer la porte |
| `oscillation` | éviter d'exploiter des mouvements aller-retour |

Deux mécanismes ont été particulièrement utiles :

- **High-watermark progress** : l'agent n'est récompensé que s'il dépasse le meilleur angle déjà atteint dans l'épisode.
- **Gating de l'approche** : l'approche ne doit pas devenir l'objectif final ; elle sert seulement à guider le début du comportement.

Malgré ces protections, le reward shaping dense seul n'a pas suffi. Il stabilise l'apprentissage, mais il ne résout pas le cold start : l'agent ne découvre pas assez souvent le contact utile avec la porte pour apprendre une politique robuste.

## 6. Protocole expérimental

Les runs ont été lancés de manière incrémentale. Chaque échec a été utilisé comme diagnostic pour modifier la configuration suivante.

| Run | Objectif |
|---|---|
| SAC v1 | configuration SAC initiale avec entropie auto |
| SAC v2 | tentative de corriger l'entropie auto |
| SAC v3 | entropie fixe, stabilité SAC |
| SAC v3 curriculum | tâche légèrement facilitée |
| SAC HER v1 | HER sparse pur |
| SAC HER v2 | HER avec reward dense hybride |

Les évaluations sont déterministes et séparées de l'entraînement. Le checkpoint à conserver est le meilleur checkpoint validation (`best_model.zip`), pas le modèle final. Cette distinction est importante en RL : un modèle peut se dégrader après son meilleur point.

Les métriques suivies sont :

| Métrique | Rôle |
|---|---|
| `val_success_rate` | succès strict, métrique principale |
| `val_door_angle_max_mean` | meilleure ouverture atteinte pendant l'épisode |
| `val_door_angle_final_mean` | ouverture finale, utile pour détecter une porte qui se referme |
| `val_return_mean` | tendance d'apprentissage, à interpréter avec prudence |
| `val_approach_frac_mean` | détection du hover-hacking |
| `train/ent_coef` | stabilité de l'exploration SAC |
| `train/critic_loss` | santé des critics |
| `train/actor_loss` | qualité du signal envoyé à l'actor |

## 7. Résultats

| Run | Steps | Succès validation | Meilleur angle max | Diagnostic |
|---|---:|---:|---:|---|
| SAC v1 | 500k | 0 % | 0.014 rad | `ent_coef` s'effondre, critic loss très élevée |
| SAC v2 | 900k | 0 % | 0.017 rad | même problème d'entropie, seulement retardé |
| SAC v3 | 400k | 0 % | 0.012 rad | entraînement stable mais aucun contact utile |
| SAC v3 curriculum | 500k | 0 % | 0.039 rad | léger progrès, puis régression |
| SAC HER v1 | 200k | 0 % | 0.000 rad | reward sparse pure trop difficile |
| SAC HER v2 | 300k | 0 % | 0.133 rad | meilleure ouverture, puis début de hover-hacking |

Le meilleur résultat est obtenu par SAC HER v2 au checkpoint 200k : `val_door_angle_max_mean = 0.133 rad`. La porte s'ouvre réellement, mais pas suffisamment ni assez longtemps pour compter comme succès strict en fin d'épisode.

Ce résultat est important même s'il ne valide pas totalement la tâche. Il montre que le problème n'est pas simplement un bug d'entraînement : l'agent peut produire une interaction physique utile avec la porte, mais le signal d'objectif reste trop fragile pour apprendre à maintenir l'ouverture.

## 8. Analyse des runs

### SAC v1 et v2 : échec de l'auto-tuning d'entropie

Les deux premiers runs échouent pour une raison similaire. Le coefficient d'entropie descend vers zéro. L'exploration disparaît trop tôt et les Q-values deviennent instables. La critic loss monte jusqu'à des valeurs pathologiques, ce qui rend les gradients de l'actor inutilisables.

La tentative v2, avec `ent_coef="auto_0.1"` et une entropie cible modifiée, ne règle pas le fond du problème. Elle décale le crash sans l'éviter.

### SAC v3 : stabilité retrouvée, mais cold start

Avec `ent_coef = 0.1` fixe, SAC devient beaucoup plus stable. Les métriques internes cessent d'exploser. Pourtant, la porte ne s'ouvre pas. Le diagnostic change donc : le problème principal n'est plus la stabilité de SAC mais l'exploration du contact.

L'agent n'observe presque jamais une transition où la porte bouge de façon significative. Le replay buffer ne contient donc pas de signal assez fort pour apprendre que certaines actions valent mieux que d'autres.

### Curriculum : amélioration légère mais insuffisante

Le run curriculum abaisse le seuil et réduit la variabilité du spawn. On observe un petit pic de performance autour de 300k steps, mais il ne se transforme pas en convergence. Le replay buffer reste dominé par des transitions sans succès.

Cette expérience suggère que faciliter seulement le seuil de succès ne suffit pas. Il faut aussi fournir au critic des exemples positifs, même partiels.

### HER : premier vrai progrès

HER v1, en sparse pur, ne marche pas : sans reward dense, l'agent n'est pas guidé vers la poignée. HER v2 corrige cela en combinant reward dense et relabellisation HER.

Le résultat est le meilleur du projet : une ouverture maximale moyenne de 0.133 rad à 200k steps. Ensuite, le comportement dérive vers une forme de hover-hacking : l'agent apprend à rester proche de la poignée plutôt qu'à pousser franchement et maintenir l'ouverture.

## 9. Difficultés rencontrées

### Instabilité des wrappers

Les wrappers RoboCasa/RoboSuite ne se comportent pas toujours comme des environnements Gymnasium standard. Un adaptateur dédié a été nécessaire pour stabiliser les observations et rendre l'environnement compatible avec Stable-Baselines3.

### Multiprocessing et rendu vidéo

L'entraînement utilise `SubprocVecEnv`, mais le rendu OpenGL n'est pas fiable dans les workers parallèles. La génération vidéo a donc été séparée de l'entraînement : un script rejoue le meilleur épisode dans un environnement single-worker avec rendu offscreen.

### Reward hacking

Le hover-hacking est apparu à plusieurs reprises. C'est un résultat intéressant du projet : une reward dense peut donner l'impression que l'agent progresse alors que le comportement réel reste mauvais. Les métriques `approach_frac`, `door_angle_max` et les vidéos sont indispensables pour éviter cette erreur d'interprétation.

### Reprise automatique

Des runs longs peuvent être interrompus. Le projet inclut une reprise automatique depuis checkpoint, mais il faut parfois forcer un nouveau run avec `--no-auto-resume` pour éviter de repartir d'une configuration déjà condamnée.

## 10. Limites

Les limites principales sont les suivantes :

| Limite | Conséquence |
|---|---|
| Succès strict toujours à 0 % | la comparaison SAC/PPO reste incomplète |
| Seed unique | variance non mesurée |
| Observations d'état uniquement | pas de conclusion sur l'apprentissage visuel |
| Pas de sim-to-real | résultats valables seulement en simulation |
| Pas d'imitation learning | exploration plus difficile depuis zéro |
| HER encore partiel | ouverture visible mais non maintenue |

### Limites matérielles et temps de calcul

Une partie importante de l'échec à obtenir un succès strict vient aussi des contraintes matérielles du projet. La tâche RoboCasa est coûteuse : chaque worker charge une simulation MuJoCo complète, les épisodes durent jusqu'à 500 steps, et SAC ajoute plusieurs mises à jour réseau pour chaque step collecté. Même avec 12 workers, un run utile demande plusieurs centaines de milliers à plusieurs millions de steps.

Dans ces conditions, le projet n'a pas permis de lancer une exploration expérimentale complète :

| Contrainte | Effet concret sur le projet |
|---|---|
| Temps de calcul limité avant la deadline | runs arrêtés dès qu'un diagnostic solide apparaissait, sans pouvoir tester toutes les variantes longues |
| Un seul GPU / une seule machine principale | impossibilité de lancer plusieurs configurations longues en parallèle |
| RAM consommée par RoboCasa et `SubprocVecEnv` | nombre de workers limité et debug plus fragile, surtout sous WSL2 |
| Coût des runs multi-seeds | résultats obtenus principalement avec `seed=0`, donc variance non estimée |
| Coût des ablations reward/HER | impossible de tester proprement toutes les combinaisons de `w_approach`, `w_progress`, curriculum et HER |
| Génération vidéo séparée | validation qualitative plus lente, car le rendu ne peut pas être fait directement dans les workers |

Ces contraintes n'expliquent pas tout : les runs ont aussi révélé de vrais problèmes algorithmiques, notamment le crash de l'entropie SAC, le cold start et le hover-hacking. En revanche, elles expliquent pourquoi le projet s'est arrêté au meilleur résultat partiel (`door_angle_max = 0.133 rad`) au lieu d'aller jusqu'à une politique robuste avec succès strict. Il aurait probablement fallu plusieurs cycles supplémentaires de runs longs, avec HER v3, curriculum plus progressif et plusieurs seeds, pour confirmer ou infirmer la capacité de l'agent à réussir régulièrement.

Ces limites ne sont pas des détails de présentation : elles définissent ce que l'on peut honnêtement conclure. Le projet montre une progression expérimentale et un diagnostic solide, mais pas encore une politique qui réussit la tâche de manière fiable.

## 11. Perspectives

Les suites les plus crédibles sont :

1. Supprimer ou réduire fortement la reward d'approche dans HER v3 pour éviter le hover-hacking.
2. Récompenser davantage le maintien de l'ouverture, pas seulement l'angle maximal atteint.
3. Lancer plusieurs seeds dès qu'un premier succès strict apparaît.
4. Ajouter un curriculum plus structuré : porte déjà entrouverte, puis difficulté croissante.
5. Utiliser des démonstrations RoboCasa pour pré-entraîner la politique par Behavior Cloning, puis fine-tuner avec SAC/HER.
6. Tester PPO uniquement après avoir stabilisé une configuration de tâche qui produit des succès.

## 12. Conclusion

Le projet n'aboutit pas encore à une politique qui ouvre la porte de manière fiable au sens strict du benchmark. En revanche, il produit une analyse utile des obstacles rencontrés en robot learning :

- SAC avec auto-tuning d'entropie peut devenir instable sur cette configuration.
- Une reward dense mal équilibrée peut favoriser le hover-hacking.
- La stabilité des losses ne suffit pas si le replay buffer ne contient aucun contact utile.
- HER permet enfin d'obtenir une ouverture visible, mais il faut encore corriger l'objectif pour maintenir cette ouverture.

Le résultat le plus solide est donc un pipeline expérimental fonctionnel et une trajectoire de diagnostic claire. Le meilleur run, SAC HER v2, ouvre la porte jusqu'à 0.133 rad en validation, ce qui marque un progrès réel par rapport aux runs précédents. La prochaine étape naturelle est de transformer cette ouverture partielle en succès stable.

## Références

- Nasiriany et al., 2024. *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*.
- Haarnoja et al., 2018. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.
- Schulman et al., 2017. *Proximal Policy Optimization Algorithms*.
- Zhu et al., 2020. *RoboSuite: A Modular Simulation Framework and Benchmark for Robot Learning*.
- Todorov et al., 2012. *MuJoCo: A physics engine for model-based control*.
