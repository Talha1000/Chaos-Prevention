import matplotlib.pyplot as plt

from .style import apply_style


apply_style()


class Plotter:

    @staticmethod
    def trajectory(data):

        plt.figure()

        plt.plot(data)

        plt.title("Logistic Map Trajectory")

        plt.xlabel("Iteration")

        plt.ylabel("State")

        plt.tight_layout()

        plt.savefig(
            "figures/trajectory.png"
        )

        plt.show()
