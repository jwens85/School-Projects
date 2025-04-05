text = input()
while text not in ["Quit", "quit", "q"]:
    print(text[::-1])
    text = input()