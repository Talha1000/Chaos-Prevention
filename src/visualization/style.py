import matplotlib.pyplot as plt


def apply_style():

    plt.style.use("ggplot")

    plt.rcParams["figure.figsize"] = (12, 6)

    plt.rcParams["font.size"] = 12

    plt.rcParams["axes.labelsize"] = 14

    plt.rcParams["axes.titlesize"] = 16

    plt.rcParams["savefig.dpi"] = 300
