# # user_input = input('Enter phone number: ')
# # index = 0
# # for character in user_input:
# #     print('Element {} is: {}'.format(index, character))
# #     index += 1
#____________________________________________
# user_input = input('Enter phone number: ')
# phone_number = ''
# for character in user_input:
#     if '0' <= character <= '9':
#         phone_number += character
#     else:
#         #FIXME: Add elif branches for letters and hyphen
#         phone_number += '?'
# print('Numbers only: {}'.format(phone_number))
#____________________________________________
# user_input = input('Enter phone number: ')
# phone_number = ''

# for character in user_input:
#     if ('0' <= character <= '9') or (character == '-'):
#         phone_number += character
#     elif ('a' <= character <= 'c') or ('A' <= character <= 'C'):
#         phone_number += '2'
#     #FIXME: Add remaining elif branches
#     else:
#         phone_number += '?'

# print('Numbers only: {}'.format(phone_number))
#____________________________________________
user_input = input('Enter phone number: ')
phone_number = ''

for character in user_input:
    if ('0' <= character <= '9') or (character == '-'):
        phone_number += character
    elif ('a' <= character <= 'c') or ('A' <= character <= 'C'):
        phone_number += '2'
    elif ('d' <= character <= 'f') or ('D' <= character <= 'F'):
        phone_number += '3'
    else:
        phone_number += '?'

print('Numbers only: {}'.format(phone_number))