(.venv) PS C:\Users\talha\Downloads\Butterfly Intervention Architecture (BIA)> python main.py
Figure saved -> figures/divergence_experiment.png
Saved -> results/divergence_20260704_164442.json

Lyapunov Exponent: 0.693173
Theoretical (ln2): 0.693147
Difference: 0.000026
Saved -> results/lyapunov_20260704_164442.json

Starting BSA...
Running BSA sweep...
Figure saved -> figures/bsa_sensitivity_map.png
Saved -> results/bsa_20260704_164450.json

Starting IWA...
Figure saved -> figures/iwa_intervention_window.png
Saved -> results/iwa_20260704_164458.json
Optimal intervention window: iterations 5 to 19
Peak leverage point: iteration 14

Starting Reactive Supervisor...
[Supervisor] Intervened at t=7 | distance=1.17e-07 | cost=1.17e-07

Final distance (supervised): 0.000000
Final distance (baseline): 0.063371
Improvement: 100.0%
Figure saved -> figures/reactive_supervisor.png
Saved -> results/reactive_supervisor_20260704_164503.json

Starting Perturbation Study...
Baseline final distance: 0.063371
Sweeping 200 perturbation scales...
True stable effective scale: 0.4372
At scale 1.0: final distance = 0.00000000
Figure saved -> figures/perturbation_study.png
Saved -> results/perturbation_study_20260704_164527.json
Critical threshold: 0.4372
Below 0.4372 — chaotic response zone
Above 0.4372 — stable recovery zone

Starting Predictive Supervisor...
[Predictive] Intervened at t=1 | distance=0.00e+00 | cost=1.50e-09 | trigger=λ-spike
[Predictive] Intervened at t=13 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=24 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=36 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=42 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=51 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=58 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=71 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=85 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike
[Predictive] Intervened at t=91 | distance=0.00e+00 | cost=0.00e+00 | trigger=λ-spike

Final distance (predictive): 0.00000000
Final distance (reactive): 0.00000000
Final distance (baseline): 0.06337129
Improvement (predictive): 100.0%
Improvement (reactive): 100.0%
Predictive interventions: 10
Reactive interventions: 1
Predictive total cost: 1.50e-09
Reactive total cost: 1.17e-07
Figure saved -> figures/predictive_supervisor.png
Saved -> results/predictive_supervisor_20260704_164533.json

==================================================
FINAL SUMMARY
==================================================
Reactive — interventions: 1 | improvement: 100.0%
Predictive — interventions: 10 | improvement: 100.0% | cost: 1.50e-09

Starting Henon Map Experiment...
Henon Lyapunov exponent: 0.417276
Theoretical value: ~0.418000

Predicted intervention window:
Opens at: t = 11.0
Closes at: t = 44.1

Observed intervention window:
Opens at: t = 15
Closes at: t = 47

[Supervisor] Intervened at t=15 | distance=1.83e-07
Final supervised distance: 0.000000
Final baseline distance: 1.334975
Improvement: 100.0%

=============================================
PREDICTION vs OBSERVATION
=============================================
Window open: predicted=11.0 observed=15
Window close: predicted=44.1 observed=47
Figure saved -> figures/henon_experiment.png
Saved -> results/henon_experiment_20260704_164539.json

Starting Lorenz System Experiment...
Computing Lorenz Lyapunov exponent...
Lorenz Lyapunov exponent: 0.889771
Theoretical value: ~0.906000

Predicted intervention window:
Opens at: t = 2.6
Closes at: t = 15.5

Distance samples:
t= 1: distance = 9.5280e-07
t= 5: distance = 1.4416e-06
t= 10: distance = 2.7300e-06
t= 15: distance = 4.8996e-06
t= 20: distance = 8.5145e-06
t= 25: distance = 1.3835e-05
t= 30: distance = 1.9004e-05
t= 50: distance = 6.6656e-06
t=100: distance = 1.1160e-06
t=200: distance = 1.1963e-06
Max distance: 4.7015e+01
Min distance: 7.3729e-07

Observed intervention window:
Opens at: t = 22
Closes at: t = 2456

[Supervisor] Intervened at t=22 | distance=1.05e-05
Final supervised distance: 0.000000
Final baseline distance: 35.216412
Improvement: 100.0%

==================================================
PREDICTION vs OBSERVATION — LORENZ
==================================================
Window open: predicted=2.6 observed=22
Window close: predicted=15.5 observed=2456
Figure saved -> figures/lorenz_experiment.png
Saved -> results/lorenz_experiment_20260704_164548.json

Starting Neural Network Drift Experiment...
Generating training data...
Training SinePredictor...
Epoch 50 | Loss: 0.003620
Epoch 100 | Loss: 0.001039
Epoch 150 | Loss: 0.000750
Epoch 200 | Loss: 0.000685
Epoch 250 | Loss: 0.000617
Epoch 300 | Loss: 0.000549
Clean MSE: 0.000547
[BIA] Reference input mean = 0.0005

Running unsupervised drift experiment...
Running supervised drift experiment...

Clean MSE (no drift): 0.000547
Unsupervised MSE (drift): 0.322098
Supervised MSE (BIA): 0.006484
Improvement: 98.0%
Interventions: 474
Total cost: 248.8500
Model weights modified: False
Figure saved -> figures/neural_drift_experiment.png
Saved -> results/neural_drift_experiment_20260704_164709.json

Neural Network Results:
Unsupervised MSE: 0.322098
Supervised MSE: 0.006484
Improvement: 98.0%
Interventions: 474
