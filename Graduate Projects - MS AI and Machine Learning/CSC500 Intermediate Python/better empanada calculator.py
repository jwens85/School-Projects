# # # # empanada_cost = 3
# # # # taco_cost = 4

# # # # user_money = int(input('Enter money for meal: '))

# # # # max_empanadas = user_money // empanada_cost
# # # # max_tacos = user_money // taco_cost

# # # # meal_cost = 0
# # # # for num_tacos in range(max_tacos + 1):
# # # #     for num_empanadas in range(max_empanadas + 1):
# # # #         meal_cost = (num_empanadas * empanada_cost) + (num_tacos * taco_cost)

# # # #         # Find first meal option that exactly matches user money
# # # #         if meal_cost == user_money:
# # # #             breakempanada_cost = 3
# # # #     if meal_cost == user_money:
# # # #         break

# # # # if meal_cost == user_money:
# # # #     print('${} buys {} empanadas and {} tacos without change.'
# # # #         .format(meal_cost, num_empanadas, num_tacos))
# # # # else:
# # # #     print('You cannot buy a meal without having change left over.')
# # # #_______________________________________________________
# # # taco_cost = 4

# # # user_money = float(input('Enter money for meal: '))

# # # max_empanadas = int(user_money / empanada_cost)
# # # max_tacos = int(user_money / taco_cost)

# # # meal_cost = 0
# # # for num_tacos in range(max_tacos + 1):
# # #     for num_empanadas in range(max_empanadas + 1):
# # #         meal_cost = (num_empanadas * empanada_cost) + (num_tacos * taco_cost)

# # #         if meal_cost == user_money:
# # #             break

# # #     if meal_cost == user_money:
# # #         break

# # # if meal_cost == user_money:
# # #     print('${} buys {} empanadas and {} tacos without change.'
# # #           .format(meal_cost, num_empanadas, num_tacos))
# # # else:
# # #     print('You cannot buy a meal without having change left over.')
# # #_______________________________________________________
# # empanada_cost = 3
# # taco_cost = 4

# # user_money = float(input('Enter money for meal: '))

# # max_empanadas = user_money // empanada_cost
# # max_tacos = user_money // taco_cost

# # meal_cost = 0
# # for num_tacos in range(max_tacos + 1):
# #     for num_empanadas in range(max_empanadas + 1):
# #         meal_cost = (num_empanadas * empanada_cost) + (num_tacos * taco_cost)

# #         if meal_cost == user_money:
# #             break

# #     if meal_cost == user_money:
# #         break

# # if meal_cost == user_money:
# #     print('${} buys {} empanadas and {} tacos without change.'
# #           .format(meal_cost, num_empanadas, num_tacos))
# # else:
# #     print('You cannot buy a meal without having change left over.')
# #_______________________________________________________
# empanada_cost = 3
# taco_cost = 4

# user_money = float(input('Enter money for meal: '))

# max_empanadas = user_money / empanada_cost
# max_tacos = user_money / taco_cost

# meal_cost = 0
# for num_tacos in range(int(max_tacos) + 1):
#     for num_empanadas in range(int(max_empanadas) + 1):
#         meal_cost = (num_empanadas * empanada_cost) + (num_tacos * taco_cost)

#         if meal_cost == user_money:
#             break

#     if meal_cost == user_money:
#         break

# if meal_cost == user_money:
#     print('${} buys {} empanadas and {} tacos without change.'
#           .format(meal_cost, num_empanadas, num_tacos))
# else:
#     print('You cannot buy a meal without having change left over.')
# #________________________________________________________
# dollars_and_cents = float(input('Enter the amount of money: '))

# empanada_cost = 3
# taco_cost = 4

# total_empanadas = int(dollars_and_cents // empanada_cost)
# remaining_cents1 = dollars_and_cents % empanada_cost

# total_tacos = int(remaining_cents1 // taco_cost)
# remaining_cents2 = remaining_cents1 % taco_cost

# print('You can buy {} empanadas and {} tacos with ${:.2f} left over.'.format(total_empanadas, total_tacos, remaining_cents2))

#:-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-)
if food_choice.lower() == 'empanadas':
    user_money = float(input('Enter money for meal: '))

    empanada_cost = 3
    taco_cost = 4
    max_empanadas = user_money // empanada_cost
    max_tacos = user_money // taco_cost
    meal_cost = 0
    for num_tacos in range(max_tacos + 1):
        for num_empanadas in range(max_empanadas + 1):
            meal_cost = (num_empanadas * empanada_cost) + (num_tacos * taco_cost)
            if meal_cost == user_money:
                break
        if meal_cost == user_money:
            break
    if meal_cost == user_money:
        print('${} buys {} empanadas and {} tacos without change.'
              .format(meal_cost, num_empanadas, num_tacos))
    else:
        print('You cannot buy a meal without having change left over.')
elif food_choice.lower() == 'tacos':
    user_money = float(input('Enter money for meal: '))

    taco_cost = 4
    empanada_cost = 3

    max_tacos = user_money // taco_cost
    remaining_money = user_money % taco_cost
    max_empanadas = remaining_money // empanada_cost

    print('You can buy {} tacos and {} empanadas with ${:.2f} left over.'
          .format(int(max_tacos), int(max_empanadas), remaining_money % empanada_cost))
else:
    print("Invalid choice of food. Please choose either tacos or empanadas.")