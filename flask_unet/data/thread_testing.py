from flask import Flask, request
import threading
import time

class myThread (threading.Thread):
   def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
   def run(self):
        time.sleep(5)
        print("Running " + self.name)

#Instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/home', methods=['GET'])
def test():
    thread1 = myThread('Test Thread')
    thread1.run()
    print("Done!")
    return "Done!"

if __name__ == "__main__":
    app.run(debug=True)