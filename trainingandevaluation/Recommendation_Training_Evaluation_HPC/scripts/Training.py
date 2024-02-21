import os
import sys

from Preprocessing import data_preprocessing as pre
from model_handler import train_and_save_model, load_and_retrain_model

def main():

    # Get the input directory and output directory from command-line arguments
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    model_directory = sys.argv[3]
    train_num = int(sys.argv[4])
    

    # Check if the input directory exists
    if not os.path.isdir(input_directory):
        print("Input directory does not exist.")
        # Define the text file name
        text_file_name = "test.txt"
         
        # Write the text content to the file
        text_file_path = os.path.join(output_directory, text_file_name)
        
        with open(text_file_path, "w") as f:
          f.write(input_directory)   
          f.write(os.listdir(input_directory))
        return
        
    # Get the path of the input files
    trainPath = os.path.join(input_directory, "A_train.npz")
    testPath = os.path.join(input_directory, "A_test.npz")
    mappingPath = os.path.join(input_directory, "init_mappings.npz")
    
    # Load the train and test matrices
    test, train, mappings = pre.load_train_and_test_matrix(dataPathTest = testPath, dataPathTrain = trainPath, dataPathMappings = mappingPath)
    user_index_map, track_index_map = mappings
    

    # Train the model and save it, each number is combined with a certain model and parameters to ease the process
    if train_num == 0:
        train_and_save_model('RankALStest', train, modelPath=output_directory, iterations=1, factors=32) # this times 30 (try for 2 hours at first)
    if train_num == 1:
        train_and_save_model('RankALS1', train, modelPath=output_directory, iterations=5, factors=16)
    if train_num == 2:
        train_and_save_model('RankALS2', train, modelPath=output_directory, iterations=5, factors=32)
    if train_num == 3:
        train_and_save_model('RankALS3', train, modelPath=output_directory, iterations=5, factors=64)
    if train_num == 4:
        load_and_retrain_model('RankALS1', 'RankALS4', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it10
    if train_num == 5:
        load_and_retrain_model('RankALS2', 'RankALS5', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it10
    if train_num == 6:
        load_and_retrain_model('RankALS3', 'RankALS6', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it10
    if train_num == 7:
        load_and_retrain_model('RankALS4', 'RankALS7', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it15
    if train_num == 8:
        load_and_retrain_model('RankALS5', 'RankALS8', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it15
    if train_num == 9:
        load_and_retrain_model('RankALS6', 'RankALS9', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it15
    if train_num == 10:
        load_and_retrain_model('RankALS7', 'RankALS10', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it20
    if train_num == 11:
        load_and_retrain_model('RankALS8','RankALS11', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it20
    if train_num == 12:
        load_and_retrain_model( 'RankALS9','RankALS12', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it20
    if train_num == 13:
        load_and_retrain_model('RankALS10', 'RankALS13', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it25
    if train_num == 14:
        load_and_retrain_model('RankALS11', 'RankALS14', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it25
    if train_num == 15:
        load_and_retrain_model('RankALS12','RankALS15', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it25
    if train_num == 16:
        load_and_retrain_model('RankALS13', 'RankALS16', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it30
    if train_num == 17:
        load_and_retrain_model('RankALS14', 'RankALS17', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it30
    if train_num == 18:
        load_and_retrain_model('RankALS15','RankALS18', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it30
    if train_num == 19:
        load_and_retrain_model( 'RankALS16','RankALS19', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it35
    if train_num == 20:
        load_and_retrain_model( 'RankALS17','RankALS20', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it35
    if train_num == 21:
        load_and_retrain_model( 'RankALS18','RankALS21', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it35
    if train_num == 31:
        train_and_save_model('RankALS128_5', train, modelPath=output_directory, iterations=5, factors=128)
    if train_num == 32:
        load_and_retrain_model('RankALS128_5', 'RankALS128_10', train, modelPath=model_directory, savePath=output_directory, iterations=5)#it10
    if train_num == 33:
        load_and_retrain_model('RankALS128_10', 'RankALS128_20', train, modelPath=model_directory, savePath=output_directory, iterations=10)#it20
    if train_num == 34:
        load_and_retrain_model('RankALS128_20', 'RankALS128_30', train, modelPath=model_directory, savePath=output_directory, iterations=10)#it30
    if train_num == 101:
        train_and_save_model('Popularity', train, modelPath=output_directory, iterations=30, factors=32)
    if train_num == 102:
        train_and_save_model('Random', train, modelPath=output_directory, iterations=30, factors=64)
if __name__ == '__main__':
    main()
