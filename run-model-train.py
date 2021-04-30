import os, yaml, time
from project_init import initialize_project
from src.data.make_dataset import make_dataset
from src.visual.generate_plots import generate_plots
from src.model.SARIMA_model import model_train_predict

def main():
    
    make_dataset()
        
    eda_summary = generate_plots()
    
    model_train_predict(eda_summary)  

if __name__ == "__main__":
    
    start_time = time.time()
    
    root_path, config  = initialize_project()
    
    main()

    end_time = time.time()
    processing_time = round((end_time - start_time)/60,2)
    
    print(f"\nOverall process completed in {processing_time} minutes") 