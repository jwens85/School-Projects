#Welcome to week 3, working on feedback form last week:
#I will insert more comments
#I will consolodate the CT secctions into one file
#I will work on defining my  main function so that it is callable in of my work

#def welcome():
#   print("Welcome to Week 3!")
#   return
# welcome()

# def meal():
#   food_charge=float(input("Enter Food Cost"))
#   tip_percent=1.18
#   sales_tax=1.07
#   print("With a tip of ", (food_charge * tip_percent) - food_charge) 
#   print("and a sales tax of ", (food_charge * sales_tax) - food_charge), 
#   print("your total is ", (food_charge * tip_percent) + (food_charge * sales_tax))
#   return
#We will now call the 'meal' function
# meal()
#The program is returning the wrong total due to incorrect operation, and percentages

# def meal():
#   food_charge=float(input("Enter Food Cost: "))
#   tax_rate=0.07
#   tip_rate=0.18
#   tax_amount=food_charge * tax_rate
#   tip_amount=food_charge * tip_rate
#   total_amount= food_charge + tax_amount + tip_amount
#   print("If your food cost is ", food_charge)
#   print("and your tax rate is ", tax_rate)
#   print("and your tip rate is ", tip_rate)
#   print("Your meal's total is ", total_amount)
#   return
# meal()
#Let's clean up the output a bit

# def meal():
#     food_charge = float(input("Enter Food Cost: "))
#     tax_rate = 0.07
#     tip_rate = 0.18
#     tax_amount = food_charge * tax_rate
#     tip_amount = food_charge * tip_rate
#     total_amount = food_charge + tax_amount + tip_amount
#     print(f"If your food cost is ${food_charge:.2f}")
#     print(f"and your tax rate is {tax_rate * 100:.2f}%")
#     print(f"and your tip rate is {tip_rate * 100:.2f}%")
#     print(f"Your meal's total is ${total_amount:.2f}")
#     return
# meal()

#Lets see how part 2 looks, time in 24h integer format so
#1100 is 11AM, 1630 is 430PM


# def alarm_clock():
#   time=int(input("Enter the time in 24h format: "))
#   wait=int(input("Number of hours to wait: "))
#   alarm_time=(time + wait)
#   print(alarm_time)
#   return
# alarm_clock()

#This is returning the wrong time, let's fix it

# def alarm_clock():
#   time=int(input("Enter the time in 24h format: "))
#   wait=int(input("Number of hours to wait: "))
#   alarm_time=(time + wait) % 24
#   print(alarm_time)
#   return
# alarm_clock()

#I would like it to return the time in a human readable sentance

# def alarm_clock():
#   time=int(input("Enter the time in 24h format: "))
#   wait=int(input("Number of hours to wait: "))
#   alarm_time=(time + wait) % 24
#   if alarm_time < 1200:
#     day_night="AM"
#   else:
#     day_night="PM"
#   print("It will be", alarm_time, day_night, "when the alarm rings")
#   return
# alarm_clock()

#This doesn't work at all

# def alarm_clock():
#     time = int(input("Enter the time in 24h format (0-23): "))
#     wait = int(input("Number of hours to wait: "))
#     alarm_time = (time + wait) % 24
#     if alarm_time < 12:
#         day_night = "AM"
#     else:
#         day_night = "PM"
#     print(f"It will be {alarm_time}:00 {day_night} when the alarm rings")

# alarm_clock()

#This is working but let's account for minutes

# def alarm_clock():
#   current_hour=int(input("Enter the current hour in a 24h format: "))
#   current_minute=int(input("Enter the current minute: "))
#   total_current_minutes = current_hour * 60 + current_minute
#   wait_hours=int(input("Enter the number of hours to wait: "))
#   wait_minutes=int(input("Enter the number of minutes to wait: "))
#   total_wait_minutes=wait_hours * 60 + wait_minutes
#   total_alarm_minutes=total_current_minutes + total_wait_minutes
#   new_hour, new_minute = divmod(total_alarm_minutes, 60)
#   new_hour %= 24
#   if new_hour <12:
#     phase = "AM"
#   else:
#     phase = "PM"
#     if new_hour > 12:
#       new_hour -= 12
#     elif new_hour == 0:
#       new_hour = 12
#   print(f"It will be {new_hour:02d}:{new_minute:02d} {phase} when the alarm rings.")
#   return
# alarm_clock()

#This seems to be working well, let's combine the two parts of the assignment

#Part 1
def meal():
    food_charge = float(input("Enter Food Cost: "))
    tax_rate = 0.07
    tip_rate = 0.18
    tax_amount = food_charge * tax_rate
    tip_amount = food_charge * tip_rate
    total_amount = food_charge + tax_amount + tip_amount
    print(f"If your food cost is ${food_charge:.2f}")
    print(f"and your tax rate is {tax_rate * 100:.2f}%")
    print(f"and your tip rate is {tip_rate * 100:.2f}%")
    print(f"Your meal's total is ${total_amount:.2f}")
    return
meal()
#Part 2
def alarm_clock():
  current_hour=int(input("Enter the current hour in a 24h format: "))
  current_minute=int(input("Enter the current minute: "))
  total_current_minutes = current_hour * 60 + current_minute
  wait_hours=int(input("Enter the number of hours to wait: "))
  wait_minutes=int(input("Enter the number of minutes to wait: "))
  total_wait_minutes=wait_hours * 60 + wait_minutes
  total_alarm_minutes=total_current_minutes + total_wait_minutes
  new_hour, new_minute = divmod(total_alarm_minutes, 60)
  new_hour %= 24
  if new_hour == 0 and new_minute == 0:
    time_str = "12 Midnight"
  elif new_hour == 12 and new_minute == 0:
    time_str = "12 Noon"
  else:
    if new_hour < 12 or new_hour == 24:
      phase = "AM"
    else:
      phase = "PM"
    if new_hour > 12:
      new_hour -= 12
    elif new_hour == 0 or new_hour == 24:
      new_hour = 12
    time_str = f"{new_hour:02d}:{new_minute:02d} {phase}"
  hour_str = "hour" if wait_hours == 1 else "hours"
  minute_str = "minute" if wait_minutes == 1 else "minutes"
  print(f"It will be {time_str} when the alarm rings, after {wait_hours} {hour_str} and {wait_minutes} {minute_str}.")

alarm_clock()