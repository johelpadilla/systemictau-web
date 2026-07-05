import pluggy
from systemictau.plugins import hookspecs

def get_plugin_manager():
    """
    Initializes the pluggy manager for Systemic Tau, enabling third-party 
    researchers to inject custom math algorithms via Python EntryPoints.
    """
    pm = pluggy.PluginManager("systemictau")
    pm.add_hookspecs(hookspecs)
    
    # Load plugins registered via setuptools entrypoints
    pm.load_setuptools_entrypoints("systemictau")
    
    return pm

# Singleton instance
plugin_manager = get_plugin_manager()
