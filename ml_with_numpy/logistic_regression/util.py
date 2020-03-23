import matplotlib.pyplot as plt


# --- Visualize Utils ---

def visualize_points(X, y, title=None, x_lim=None, y_lim=None):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    plt.scatter(X_0[:, 0], X_0[:, 1], c='r')
    plt.scatter(X_1[:, 0], X_1[:, 1], c='b')
    plt.grid(True); 
    if title is not None:
        plt.title(title)
    
    if x_lim is not None:
        plt.xlim(x_lim)
    
    if y_lim is not None:
        plt.ylim(y_lim)
        
        
def get_lim_range(X, axis, margin_ratio):
    if margin_ratio <= 0 or margin_ratio >= 1:
        raise ValueError("`margin_ratio` has to be between 0 and 1")
    X_min = X[:, axis].min()
    X_max = X[:, axis].max()
    r = X_max - X_min
    margin = r * margin_ratio
    return X_min - margin / 2.0, X_max + margin / 2.0


def get_points(x1, x2, w, b):
    y1 = (-b.flatten() - x1 * w.flatten()[0]) / (w.flatten()[1])
    y2 = (-b.flatten() - x2 * w.flatten()[0]) / (w.flatten()[1])
    return y1[0], y2[0]


def visualize_line(x1, x2, y1, y2, linestyle=None, linewidth=None, color=None):
    x_values = [x1, x2]
    y_values = [y1, y2]
    plt.plot(x_values, y_values, linestyle=linestyle, linewidth=linewidth, c=color)
    