num_integers = int(input())
numbers = []

for _ in range(num_integers):
    number = int(input())
    numbers.append(number)

min_value = min(numbers)

for number in numbers:
    adjusted_number = number - min_value
    print(adjusted_number)