import yaml, os

def initialize_project():
    
    # make sure projects are set to the root directory of the project
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT_DIR)
    
    # Function to load yaml configuration file
    def load_config(config_name):
        with open(os.path.join(config_path, config_name), 'r') as file:
            config = yaml.safe_load(file)

        return config

    config_path = "conf/base"

    config = load_config("catalog.yml")
       
    return ROOT_DIR, config

if __name__ == "__main__":

    root_path, config = initialize_project()