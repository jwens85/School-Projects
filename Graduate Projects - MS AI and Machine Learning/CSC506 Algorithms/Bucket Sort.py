import math

def bucket_sort(numbers, bucket_count):
    numbers_size = len(numbers)
    if numbers_size < 1:
        return

    # Create buckets as empty lists
    buckets = [[] for _ in range(bucket_count)]

    # Find the maximum value in the list to normalize bucket indices
    max_value = max(numbers)

    # Distribute each number into a bucket
    for number in numbers:
        # Determine the bucket index
        index = math.floor(number * bucket_count / (max_value + 1))
        buckets[index].append(number)

    # Sort each bucket and concatenate all sorted buckets
    sorted_numbers = []
    for bucket in buckets:
        sorted_numbers.extend(sorted(bucket))

    # Copy sorted numbers back to original list
    numbers[:] = sorted_numbers
