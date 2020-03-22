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
        
        

        
    