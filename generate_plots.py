import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: generate_plots.py <costs_file> <params_file> <output_folder>", file=sys.stderr)
        exit(-1)

    with open(sys.argv[1]) as file_in:
        cost = []
        for line in file_in:
            cost.append(float(line.strip()))


    f1, ax1 = plt.subplots()
    ax1.plot(cost)
    plt.title("Cost per Epoch")
    plt.legend(['Cost'])
    plt.savefig(sys.argv[3] + '/cost_history.png')

    if sys.argv[2] != 'None':

        with open(sys.argv[2]) as file_in:
            acc = []
            for line in file_in:
                acc.append(float(line.strip()))

        f2, ax2 = plt.subplots()
        ax2.plot(acc)
        plt.title("Accuracy per Epoch")
        plt.legend(['Acc %'])
        plt.savefig(sys.argv[3] + '/accuracy_history.png')

    print("Generate at " + sys.argv[3] + '*.png')
