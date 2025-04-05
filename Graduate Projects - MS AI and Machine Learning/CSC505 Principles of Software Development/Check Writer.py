# Check Writer V1.0
# def number_to_words(n):
#     ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
#     teens = ["Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
#     tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
#     thousands = ["", "Thousand"]

#     if n == 0:
#         return "Zero"

#     words = ""

#     if n // 1000 > 0:
#         words += ones[n // 1000] + " Thousand "
#         n %= 1000

#     if n // 100 > 0:
#         words += ones[n // 100] + " Hundred "
#         n %= 100

#     if n > 10 and n < 20:
#         words += teens[n - 11] + " "
#         n = 0

#     if n >= 20:
#         words += tens[n // 10] + " "
#         n %= 10

#     if n > 0:
#         words += ones[n] + " "

#     return words.strip()

# def check_writer(amount):
#     amount_in_words = number_to_words(amount)
#     return f"${amount}.00 ({amount_in_words} Dollars)"

# amount = int(input("Enter the check amount (0-9999): "))
# if 0 <= amount <= 9999:
#     print(check_writer(amount))
# else:
#     print("Please enter a number between 0 and 9999.")

# ###################################################################################################

# # Check Writer V1.1
# def number_to_words(n):
#     ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
#     teens = ["Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
#     tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
#     thousands = ["", "Thousand"]

#     if n == 0:
#         return "Zero"

#     words = ""

#     if n // 1000 > 0:
#         words += ones[n // 1000] + " Thousand "
#         n %= 1000

#     if n // 100 > 0:
#         words += ones[n // 100] + " Hundred "
#         n %= 100

#     if n > 10 and n < 20:
#         words += teens[n - 11] + " "
#         n = 0

#     if n >= 20:
#         words += tens[n // 10] + " "
#         n %= 10

#     if n > 0:
#         words += ones[n] + " "

#     return words.strip()

# def check_writer(amount):
#     dollars = int(amount)
#     cents = int(round((amount - dollars) * 100))

#     if dollars == 0:
#         dollar_part = "Zero Dollars"
#     else:
#         dollar_part = f"{number_to_words(dollars)} Dollar" + ("s" if dollars > 1 else "")

#     if cents == 0:
#         cent_part = "Zero Cents"
#     else:
#         cent_part = f"{number_to_words(cents)} Cent" + ("s" if cents > 1 else "")

#     return f"{dollars}.{cents:02d} ({dollar_part} and {cent_part})"


# amount = float(input("Enter the check amount (0.00-9999.99): "))
# if 0 <= amount <= 9999.99:
#     print(check_writer(amount))
# else:
#     print("Please enter an amount between 0.00 and 9999.99.")

# ####################################################################################################

# Check Writer V1.2
def number_to_words(n):
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    thousands = ["", "Thousand"]

    if n == 0:
        return "Zero"

    words = ""

    if n // 1000 > 0:
        words += ones[n // 1000] + " Thousand "
        n %= 1000

    if n // 100 > 0:
        words += ones[n // 100] + " Hundred "
        n %= 100

    if n > 10 and n < 20:
        words += teens[n - 11] + " "
        n = 0

    if n >= 20:
        words += tens[n // 10] + " "
        n %= 10

    if n > 0:
        words += ones[n] + " "

    return words.strip()

def check_writer(amount):
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    if dollars == 0:
        dollar_part = "Zero Dollars"
    else:
        dollar_part = f"{number_to_words(dollars)} Dollar" + ("s" if dollars > 1 else "")

    if cents == 0:
        cent_part = "Zero Cents"
    else:
        cent_part = f"{number_to_words(cents)} Cent" + ("s" if cents > 1 else "")

    return f"{dollar_part} and {cent_part}"

def create_text_check(check_number, payee, date, amount, memo):
    amount_in_words = check_writer(amount)

    lines = [
        "John Wensink                           Check No: {}".format(check_number),
        "                                            Date: {}".format(date),
        "",
        "PAY TO THE ORDER OF: {}".format(payee),
        "",
        "                                           ${:,.2f}".format(amount),
        "{}".format(amount_in_words),
        "",
        "Memo: {}                         Signature: ____________".format(memo)
    ]

    max_length = max(len(line) for line in lines)
    border = "-" * (max_length + 4)

    print(border)
    for line in lines:
        print("| {:<{}} |".format(line, max_length))
    print(border)

check_number = input("Enter the check number: ")
payee = input("Enter the payee: ")
date = input("Enter the date (MM/DD/YYYY): ")

while True:
    amount = float(input("Enter the check amount (0.00-9999.99): "))
    if 0.00 <= amount <= 9999.99:
        break
    else:
        print("Please enter a valid amount between 0.00 and 9999.99.")

memo = input("Enter the memo: ")

create_text_check(check_number, payee, date, amount, memo)
