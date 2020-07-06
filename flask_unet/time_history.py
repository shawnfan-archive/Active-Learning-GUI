import keras
import time

class TimeHistory(keras.callbacks.Callback):
    
    current_epoch = 0
    current_batch = 0
    time_remaining = "Calculating..."
    total_epochs = 0
    total_batches = 0
    finished = False

    def __init__(self, total_epochs, total_batches):
        TimeHistory.total_epochs = total_epochs
        TimeHistory.total_batches = total_batches

    #Once training begins...
    def on_train_begin(self, logs={}):

        #Reset current epoch
        TimeHistory.current_epoch = 0
        TimeHistory.time_remaining = "Calculating..."
        
        #Create array
        self.batch_times = []
        self.epoch_times = []
        
        #Creates textfile we will update with time
        file = open("testfile.txt", "w")
        file.write("Training Started...\n")
        file.close()

    def on_train_end(self, logs={}):
        
        TimeHistory.finished = True

    #Once batch begins...
    def on_batch_begin(self, batch, logs={}):
        
        self.batch_time_start = time.time()
        
    #Once batch finishes...
    def on_batch_end(self, batch, logs={}):

        TimeHistory.current_batch += 1
        
        #Calculate and append elapsed time
        elapsed = (time.time() - self.batch_time_start)
        self.batch_times.append(elapsed)

        #Update time remaining 
        #Remaining batches for current epoch
        remaining_batches = TimeHistory.total_batches - TimeHistory.current_batch
        remaining_epochs = TimeHistory.total_epochs - TimeHistory.current_epoch
        TimeHistory.time_remaining = time.strftime(
            "%H:%M:%S",
            time.gmtime(elapsed * ((TimeHistory.total_batches * remaining_epochs) + remaining_batches)))
        
        #Write to file
        with open("testfile.txt", "a") as file:
            file.write("Elapsed Time (Batch): " + str(elapsed) + "\n")
            file.close()

        
    #Once epoch begins...
    def on_epoch_begin(self, batch, logs={}):

        #Reset current batch to 1 
        TimeHistory.current_batch = 1
        
        self.epoch_time_start = time.time()

    #Once epoch finishes...
    def on_epoch_end(self, batch, logs={}):

        TimeHistory.current_epoch += 1
        
        #Calculate and append elapsed time
        elapsed = (time.time() - self.epoch_time_start)
        self.epoch_times.append(elapsed)

        #Update time remaining
        TimeHistory.time_remaining = time.strftime(
            "%H:%M:%S",
            time.gmtime(elapsed * (TimeHistory.total_epochs - TimeHistory.current_epoch)))
        
        #Write to file
        with open("testfile.txt", "a") as file:
            file.write("Elapsed Time (Epoch): " + str(elapsed) + "\n")
            file.close()

class PredictionProgress():
    
    progress = 0
    total = 0
    finished = False

    def __init__(self, total):
        PredictionProgress.total = total
        PredictionProgress.progress = 0
    
    def on_prediction_end(self):
        PredictionProgress.total -= 1
        PredictionProgress.progress += 1
    
    def on_completion(self):
        PredictionProgress.finished = True