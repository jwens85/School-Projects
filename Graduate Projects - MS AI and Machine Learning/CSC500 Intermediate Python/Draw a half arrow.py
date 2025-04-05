arrow_base_height = int(input("Enter arrow base height:\n"))
arrow_base_width = int(input("Enter arrow base width:\n"))
arrow_head_width = int(input("Enter arrow head width:\n"))

while arrow_head_width <= arrow_base_width:
    arrow_head_width = int(input("Enter arrow head width:\n"))

print()

for _ in range(arrow_base_height):
    print('*' * arrow_base_width)

for width in range(arrow_head_width, 0, -1):
    print('*' * width)