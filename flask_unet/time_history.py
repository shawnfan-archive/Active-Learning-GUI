import keras
import time

class TimeHistory(keras.callbacks.Callback):
    
    #Once training begins...
    def on_train_begin(self, logs={}):
        
        #Create array
        self.batch_times = []
        self.epoch_times = []
        
        #Creates textfile we will update with time
        file = open("testfile.txt", "w")
        file.write("Training Started...\n")
        file.close()

    #Once batch begins...
    def on_batch_begin(self, batch, logs={}):
        
        self.batch_time_start = time.time()
        
    #Once batch finishes...
    def on_batch_end(self, batch, logs={}):
        
        #Calculate and append elapsed time
        elapsed = (time.time() - self.batch_time_start)
        self.batch_times.append(elapsed)
        
        #Write to file
        with open("testfile.txt", "a") as file:
            file.write("Elapsed Time (Batch): " + str(elapsed) + "\n")
            file.close()

        
    #Once epoch begins...
    def on_epoch_begin(self, batch, logs={}):
        
        self.epoch_time_start = time.time()

    #Once epoch finishes...
    def on_epoch_end(self, batch, logs={}):
        
        #Calculate and append elapsed time
        elapsed = (time.time() - self.epoch_time_start)
        self.epoch_times.append(elapsed)
        
        #Write to file
        with open("testfile.txt", "a") as file:
            file.write("Elapsed Time (Epoch): " + str(elapsed) + "\n")
            file.close()
