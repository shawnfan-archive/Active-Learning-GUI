import requests

def request_test():
    response = requests.get('http://localhost:5000/home')

if __name__ == "__main__":
    request_test()