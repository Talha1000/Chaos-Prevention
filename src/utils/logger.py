import os
import json
from datetime import datetime


class ExperimentLogger:

    def __init__(self):

        os.makedirs("results", exist_ok=True)

    def save(self, name, results):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"results/{name}_{timestamp}.json"

        with open(filename, "w") as f:

            json.dump(results, f, indent=4)

        print(f"Saved -> {filename}")
