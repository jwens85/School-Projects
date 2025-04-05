def print_menu():
    print("MENU\n\
c - Number of non-whitespace characters\n\
w - Number of words\n\
f - Fix capitalization\n\
r - Replace punctuation\n\
s - Shorten spaces\n\
q - Quit\n")


def execute_menu_2(user_choice, sample_text):
    if user_choice != 'q':
        if user_choice == 'c':
            print('Number of non-whitespace characters: {}\n'.format(get_num_of_non_WS_characters(sample_text)))
        if user_choice == 'w':
            print('Number of words: {:d}\n'.format(get_num_of_words(sample_text)))
        if user_choice == 'f':
            letter_cap_counter, edited_sample_text = fix_capitalization_2(sample_text)
            print('Number of letters capitalized: {:d}'.format(letter_cap_counter))
            print('Edited text: {:s}\n'.format(edited_sample_text))
        if user_choice == 'r':
            edited_sample_text, exclamation_count, semicolon_count = replace_punctuation_2(sample_text)
            print('Punctuation replaced')
            print('exclamation_count: {:d}'.format(exclamation_count))
            print('semicolon_count: {:d}'.format(semicolon_count))
            print('Edited text: {:s}\n'.format(edited_sample_text))
        if user_choice == 's':
            print('Edited text: {:s}\n'.format(shorten_space(sample_text)))
        print_menu()
        print('Choose an option:')


def get_num_of_non_WS_characters(sample_text):
    characters = 0
    for char in sample_text:
        if char != ' ':
            characters += 1
    return characters


def get_num_of_words(sample_text):
    num_words = int(len(sample_text.split()))
    return num_words


def fix_capitalization_2(sample_text):
    capitalize = True
    letter_cap_counter = 0
    edited_sample_text = ''
    for char in sample_text:
        if capitalize == False:
            edited_sample_text += char
            if char == '.':
                capitalize = True
        elif char != ' ' and capitalize == True and char.islower():
            char = char.upper()
            edited_sample_text += char
            letter_cap_counter += 1
            capitalize = False
        else:
            edited_sample_text += char
    return letter_cap_counter, edited_sample_text


def replace_punctuation_2(sample_text, exclamation_count=0, semicolon_count=0):
    edited_sample_text = ''
    for char in sample_text:
        if char != '!' and char != ';':
            edited_sample_text += char
        if char == '!':
            edited_sample_text += '.'
            exclamation_count += 1
        if char == ';':
            edited_sample_text += ','
            semicolon_count += 1
    return edited_sample_text, exclamation_count, semicolon_count


def shorten_space(sample_text):
    edited_sample_text = ''
    sample_text_list = sample_text.split()
    for word in sample_text_list:
        if word != sample_text_list[len(sample_text_list) - 1]:
            edited_sample_text += word + ' '
        elif word == sample_text_list[len(sample_text_list) - 1]:
            edited_sample_text += word
    return edited_sample_text


if __name__ == '__main__':
    sample_text = str(input('Enter a sample text:\n'))
    print('\nYou entered: {:s}\n'.format(sample_text))
    print_menu()
    user_choice = str(input('Choose an option:\n'))
    execute_menu_2(user_choice, sample_text)
