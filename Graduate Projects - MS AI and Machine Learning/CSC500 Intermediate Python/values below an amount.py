num_integers = int(input())
numbers = []

for _ in range(num_integers):
    number = int(input())
    numbers.append(number)

threshold = int(input())

for number in numbers:
    if number <= threshold:
        print(number)