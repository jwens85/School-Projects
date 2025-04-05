  numbers = []
  num = int(input())
  while num > 0:
      numbers.append(num)
      try:
          num = int(input())
      except EOFError:
          break
  print(min(numbers), max(numbers))
