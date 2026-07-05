import numpy as np

class ChaosGenerator:
    """
    A utility class for generating synthetic multivariate time series 
    from coupled chaotic systems, ideal for testing the Systemic Tau framework.
    """
    
    @staticmethod
    def logistic_map_coupled(n_steps: int, n_comp: int, r: float = 3.8, coupling: float = 0.05, noise: float = 0.0) -> np.ndarray:
        """
        Generates a multivariate time series using coupled logistic maps.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps.
        n_comp : int
            Number of coupled components/variables.
        r : float, optional
            The growth rate parameter (default 3.8 for chaos).
        coupling : float, optional
            The strength of coupling between components (default 0.05).
        noise : float, optional
            The standard deviation of Gaussian noise added to the output (default 0.0).
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_steps, n_comp).
        """
        X = np.zeros((n_steps, n_comp))
        
        # Initial conditions in (0, 1)
        X[0, :] = np.random.uniform(0.2, 0.8, n_comp)
        
        for t in range(1, n_steps):
            for i in range(n_comp):
                # Calculate mean of all other components
                if n_comp > 1:
                    others = np.delete(X[t-1, :], i)
                    mean_others = np.mean(others)
                else:
                    mean_others = 0.0
                    
                # Coupled logistic map step
                x_prev = X[t-1, i]
                val = (1 - coupling) * r * x_prev * (1 - x_prev) + coupling * r * mean_others * (1 - mean_others)
                X[t, i] = np.clip(val, 0, 1)
                
        if noise > 0:
            X += np.random.normal(0, noise, X.shape)
            
        return X

    @staticmethod
    def lorenz_coupled(n_steps: int, dt: float = 0.01, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0, noise: float = 0.0) -> np.ndarray:
        """
        Generates a multivariate time series using the Lorenz attractor (X, Y, Z components).
        Uses simple Euler integration.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps.
        dt : float, optional
            Integration time step (default 0.01).
        sigma, rho, beta : float, optional
            Lorenz system parameters (default chaotic regime).
        noise : float, optional
            Noise standard deviation.
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_steps, 3) representing X, Y, Z coordinates.
        """
        X = np.zeros((n_steps, 3))
        # Initial conditions near the attractor
        X[0] = [1.0, 1.0, 1.0]
        
        for t in range(1, n_steps):
            x, y, z = X[t-1]
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            X[t, 0] = x + dx * dt
            X[t, 1] = y + dy * dt
            X[t, 2] = z + dz * dt
            
        if noise > 0:
            X += np.random.normal(0, noise, X.shape)
            
        return X
        
    @staticmethod
    def rossler_coupled(n_steps: int, dt: float = 0.01, a: float = 0.2, b: float = 0.2, c: float = 5.7, noise: float = 0.0) -> np.ndarray:
        """
        Generates a multivariate time series using the Rössler attractor (X, Y, Z components).
        Uses simple Euler integration.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps.
        dt : float, optional
            Integration time step (default 0.01).
        a, b, c : float, optional
            Rössler system parameters (default chaotic regime).
        noise : float, optional
            Noise standard deviation.
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_steps, 3).
        """
        X = np.zeros((n_steps, 3))
        # Initial conditions
        X[0] = [0.1, 0.0, 0.1]
        
        for t in range(1, n_steps):
            x, y, z = X[t-1]
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            X[t, 0] = x + dx * dt
            X[t, 1] = y + dy * dt
            X[t, 2] = z + dz * dt
            
        if noise > 0:
            X += np.random.normal(0, noise, X.shape)
            
        return X
