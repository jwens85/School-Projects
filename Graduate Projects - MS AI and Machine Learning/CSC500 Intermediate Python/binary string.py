num = int(input())
binary_string = ""
while num > 0:
    binary_string += str(num % 2)
    num //= 2
print(binary_string)