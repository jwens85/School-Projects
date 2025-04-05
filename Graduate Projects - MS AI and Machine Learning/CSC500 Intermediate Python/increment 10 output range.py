first_int = int(input())
second_int = int(input())

if second_int < first_int:
    print("Second integer can't be less than the first.")
else:
    for i in range(first_int, second_int + 1, 10):
        print(i, end=' ')
    print()