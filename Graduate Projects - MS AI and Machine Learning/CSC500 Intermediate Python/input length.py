user_text = input()
count = 0
for char in user_text:
    if char not in [' ', '.', ',']:
        count += 1

print(count)