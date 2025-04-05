age=int(input("What is your age?"))
if age<18:
  print("You are a child grow up")
else:
  print("You are an adult go to work")
brains=bool(input("Are you smart?"))
if brains==True:
  print("Go to school")
else:
  print("Go to work")
lazy=int(input("How lazy are you from 1-10?"))
if lazy<3:
  print("You are going to be homeless")
if lazy <= 7:  # Corrected comparison
  print("You are going to be a bum")
else:
  print("You are going to be a millionaire")