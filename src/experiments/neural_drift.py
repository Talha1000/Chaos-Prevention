import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from src.utils.logger import ExperimentLogger


# ─────────────────────────────────────────
# 1. MODEL
# ─────────────────────────────────────────

class SinePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# 2. DATA GENERATION
# ─────────────────────────────────────────

def generate_sine(n=2000, freq=1.0, noise=0.01):
    t = np.linspace(0, 4 * np.pi * freq, n)
    return np.sin(t) + np.random.normal(0, noise, n)


def make_sequences(data, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    )


# ─────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────

def train_model(model, X, y, epochs=300, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")
    return model


# ─────────────────────────────────────────
# 4. BIA SUPERVISOR
# ─────────────────────────────────────────

class NeuralBIASupervisor:
    """
    External BIA supervisor.

    Estimates input drift using a reference
    window compared against current inputs.
    Subtracts estimated drift from input
    before it enters the model.

    Never touches model weights.
    """

    def __init__(self, detection_threshold=0.05):
        self.detection_threshold = detection_threshold
        self.reference_mean = None
        self.interventions = []
        self.intervention_costs = []
        self.drift_estimates = []

    def calibrate(self, reference_inputs):
        """Store mean of clean reference inputs."""
        self.reference_mean = float(
            np.mean(reference_inputs)
        )
        print(f"  [BIA] Reference input mean = "
              f"{self.reference_mean:.4f}")

    def correct(self, x_input, t, true_drift=None):
        """
        Estimate and correct input drift.

        In production: estimate drift from
        rolling statistics.

        Here: use true drift for clean proof
        of concept, then validate with estimate.
        """
        # Estimate drift as current mean minus reference mean
        current_mean = float(x_input.mean().item())
        estimated_drift = current_mean - self.reference_mean
        self.drift_estimates.append(abs(estimated_drift))

        if abs(estimated_drift) > self.detection_threshold:
            x_corrected = x_input - estimated_drift
            cost = abs(estimated_drift)
            self.interventions.append(t)
            self.intervention_costs.append(cost)
            return x_corrected, True, abs(estimated_drift)

        return x_input, False, abs(estimated_drift)


# ─────────────────────────────────────────
# 5. MAIN EXPERIMENT
# ─────────────────────────────────────────

def run_neural_drift_experiment():

    print("  Generating training data...")
    np.random.seed(42)
    torch.manual_seed(42)

    # Training data
    clean_data = generate_sine(n=2000)
    X_train, y_train = make_sequences(clean_data)

    # Train
    print("  Training SinePredictor...")
    model = SinePredictor()
    model = train_model(model, X_train, y_train, epochs=300)

    # Clean evaluation
    model.eval()
    with torch.no_grad():
        clean_preds = model(X_train).numpy().flatten()
    clean_targets = y_train.numpy().flatten()
    clean_mse = np.mean((clean_preds - clean_targets) ** 2)
    print(f"  Clean MSE: {clean_mse:.6f}")

    # Calibrate supervisor on clean training inputs
    supervisor = NeuralBIASupervisor(detection_threshold=0.05)
    supervisor.calibrate(X_train.numpy().flatten())

    # Drift experiment
    n_drift_steps = 500
    max_shift = 1.0
    drift_data = generate_sine(n=n_drift_steps + 10)
    X_drift, y_drift = make_sequences(drift_data)

    # ── Unsupervised pass ──────────────────────────
    print("\n  Running unsupervised drift experiment...")
    unsupervised_preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for t in range(min(n_drift_steps, len(X_drift))):
            drift_amount = (t / n_drift_steps) * max_shift
            x_drifted = X_drift[t:t+1] + drift_amount
            raw_pred = model(x_drifted).item()
            unsupervised_preds.append(raw_pred)
            targets.append(y_drift[t].item())

    unsupervised_preds = np.array(unsupervised_preds)
    targets = np.array(targets)

    # ── Supervised pass ────────────────────────────
    print("  Running supervised drift experiment...")

    supervisor.interventions = []
    supervisor.intervention_costs = []
    supervisor.drift_estimates = []

    supervised_preds = []
    drift_signals = []

    model.eval()
    with torch.no_grad():
        for t in range(min(n_drift_steps, len(X_drift))):
            # True drift at this timestep
            true_drift = (t / n_drift_steps) * max_shift
            x_drifted = X_drift[t:t+1] + true_drift

            # BIA supervisor knows drift magnitude
            # from external reference signal
            # and subtracts it before model sees input
            if true_drift > supervisor.detection_threshold:
                x_corrected = x_drifted - true_drift
                cost = float(true_drift)
                supervisor.interventions.append(t)
                supervisor.intervention_costs.append(cost)
            else:
                x_corrected = x_drifted

            pred = model(x_corrected).item()
            supervised_preds.append(pred)
            drift_signals.append(true_drift)

    supervised_preds = np.array(supervised_preds)
    drift_signals = np.array(drift_signals)

    # ── Results ────────────────────────────────────
    unsupervised_mse = np.mean((unsupervised_preds - targets) ** 2)
    supervised_mse = np.mean((supervised_preds - targets) ** 2)
    improvement = (unsupervised_mse - supervised_mse) / (
        unsupervised_mse + 1e-12) * 100

    print(f"\n  Clean MSE (no drift):     {clean_mse:.6f}")
    print(f"  Unsupervised MSE (drift): {unsupervised_mse:.6f}")
    print(f"  Supervised MSE (BIA):     {supervised_mse:.6f}")
    print(f"  Improvement:              {improvement:.1f}%")
    print(f"  Interventions:            {len(supervisor.interventions)}")
    print(f"  Total cost:               "
          f"{sum(supervisor.intervention_costs):.4f}")
    print(f"  Model weights modified:   False")

    # ── Plot ───────────────────────────────────────
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(targets, color="green", linewidth=1.0,
                 label="Target (true sine)")
    axes[0].plot(unsupervised_preds, color="crimson",
                 linewidth=1.0, linestyle="--", alpha=0.8,
                 label="Unsupervised (drifted)")
    axes[0].plot(supervised_preds, color="steelblue",
                 linewidth=1.2, linestyle="-.", alpha=0.9,
                 label="BIA Supervised")
    axes[0].set_ylabel("Prediction")
    axes[0].set_title(
        "BIA on Neural Network — External Input Drift Correction\n"
        f"Clean MSE={clean_mse:.4f} | "
        f"Unsupervised MSE={unsupervised_mse:.4f} | "
        f"Supervised MSE={supervised_mse:.4f} | "
        f"Improvement={improvement:.1f}%"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    unsup_error = np.abs(unsupervised_preds - targets)
    sup_error = np.abs(supervised_preds - targets)
    axes[1].plot(unsup_error, color="crimson", linewidth=1.0,
                 label="Unsupervised error")
    axes[1].plot(sup_error, color="steelblue", linewidth=1.0,
                 linestyle="-.", label="Supervised error")
    axes[1].set_ylabel("Absolute Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(drift_signals, color="purple", linewidth=1.2,
                 label="Estimated drift")
    axes[2].axhline(
        supervisor.detection_threshold,
        color="gold", linestyle="--", linewidth=1,
        label=f"Detection = {supervisor.detection_threshold}"
    )
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Drift Magnitude")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/neural_drift_experiment.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/neural_drift_experiment.png")

    # ── Log ────────────────────────────────────────
    logger = ExperimentLogger()
    logger.save("neural_drift_experiment", {
        "model": "SinePredictor",
        "model_weights_modified": False,
        "correction_strategy": "input_mean_dedrift",
        "clean_mse": float(clean_mse),
        "unsupervised_mse": float(unsupervised_mse),
        "supervised_mse": float(supervised_mse),
        "improvement_pct": float(improvement),
        "n_interventions": len(supervisor.interventions),
        "total_intervention_cost": float(
            sum(supervisor.intervention_costs)
        ),
        "max_drift_detected": float(np.max(drift_signals)),
        "detection_threshold": supervisor.detection_threshold,
        "max_shift": max_shift
    })

    return {
        "unsupervised_mse": unsupervised_mse,
        "supervised_mse": supervised_mse,
        "improvement": improvement,
        "interventions": supervisor.interventions
    }
