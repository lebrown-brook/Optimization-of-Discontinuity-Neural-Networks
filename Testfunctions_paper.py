import numpy as np
import matplotlib.pyplot as plt


def g_l(x: np.ndarray) -> np.ndarray:

    #input is a 2D numpy array with shape (n_samples, 2)
    #return 2sin(1.25pi*||x||) + 4  if x_2 <= 2x_1
    #else 2sin(0.75pi*||x||)
    return np.where(x[:, 1] <= 2 * x[:, 0],
                    2 * np.sin(1.25 * np.pi * np.linalg.norm(x, axis=1)) + 4,
                    2 * np.sin(0.75 * np.pi * np.linalg.norm(x, axis=1))
                   ).reshape(-1, 1)




def g_s(x: np.ndarray) -> np.ndarray:
    #return -2x_1^2 +6 if x in [-1,1]x[0,+infinity)
    #else 4*e^((1-x_1^2)/2)

    return np.where((x[:, 0] >= -1) & (x[:, 0] <= 1) & (x[:, 1] >= 0),
                    -2 * x[:, 0]**2 + 6,
                    4 * np.exp((1 - x[:, 0]**2) / 2)
                   ).reshape(-1, 1)

def g_eta(x: np.ndarray) -> np.ndarray:
    #return sin(0.4pi(x_1 +x_2))  if x_2 >= e^x_1
    #        sin(0.7pi(x_1 +x_2)) -4 if x_2 < e^x_1 -1
    #       sin(pi(x_1 +x_2)) +4 otherwise
    return np.where(x[:, 1] >= np.exp(x[:, 0]),
                    np.sin(0.4 * np.pi * (x[:, 0] + x[:, 1])),
                    np.where(x[:, 1] < np.exp(x[:, 0]) - 1,
                             np.sin(0.7 * np.pi * (x[:, 0] + x[:, 1])) - 4,
                             np.sin(np.pi * (x[:, 0] + x[:, 1])) + 4
                            )
                   ).reshape(-1, 1)


def g_gamma(x: np.ndarray) -> np.ndarray:
    #sin(pi (x_1 + x_2)) + 4 if ||x||^2 <= 1
    #sin(0.4pi(x_1 + x_2)) otherwise
    return np.where(np.linalg.norm(x, axis=1)**2 <= 1,
                    np.sin(np.pi * (x[:, 0] + x[:, 1])) + 4,
                    np.sin(0.4 * np.pi * (x[:, 0] + x[:, 1]))
                   ).reshape(-1, 1)



#run a simple test to visualize the functions by heatmaps on the grid [-1,1]x[-1,1]
if __name__ == "__main__":

    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-2, 2, 400)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    functions = {
        "g_l": g_l,
        "g_s": g_s,
        "g_eta": g_eta,
        "g_gamma": g_gamma
    }

    for name, func in functions.items():
        Y = func(X_grid).reshape(X1.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(X1, X2, Y, levels=100, cmap='viridis')
        plt.colorbar()
        plt.title(f"Heatmap of {name}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.savefig(f"{name}_heatmap.png")
        plt.close()