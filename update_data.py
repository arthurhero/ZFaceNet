from multiprocessing import Process
import dataloader as dl

def d():
    while True:
        dl.download_random_data()
    
for num in range(10):
    Process(target=d, args=()).start()
