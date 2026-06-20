import multiprocessing
import time

def compute_square(number):
    """A heavy or time-consuming CPU task."""
    time.sleep(0.5)  # Simulate a slight delay
    return number * number

if __name__ == "__main__":
    numbers = range(100)
    
    # Track performance
    start_time = time.perf_counter()
    
    # Context manager automatically handles opening and closing the pool
    # It defaults to the number of CPU cores available on your machine
    with multiprocessing.Pool() as pool:
        # pool.map distributes the list items across worker processes
        results = pool.map(compute_square, numbers)
        worker_count = pool._processes
        print(f"Total workers in pool: {worker_count}")  # Outputs: 4
        end_time = time.perf_counter()
    
    print(f"Results: {results}")
    print(f"Finished in {end_time - start_time:.2f} seconds.")
    
# Track performance
# start_time = time.perf_counter()
# for i in range(100): 
#     compute_square(i)
# end_time = time.perf_counter()    
# print(f"Results: {results}")
# print(f"Finished in {end_time - start_time:.2f} seconds.")