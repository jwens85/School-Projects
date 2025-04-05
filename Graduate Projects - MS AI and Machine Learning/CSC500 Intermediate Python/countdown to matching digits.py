num = int(input())
if 20 <= num <= 98:
    while num // 10 != num % 10:
        print(num)
        num = num - 1
    print(num)
else:
    print("Input must be 20-98")