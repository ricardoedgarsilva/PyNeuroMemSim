import time

for i in range(10):
    print(f"\rProgress: {i * 10}%", end='')
    time.sleep(1)
print("\rCompleted!")  # Overwrite with the final message
