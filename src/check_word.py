def check_word_in_file(word, word_set):
    return word.lower() in word_set

def calculate_accuracy(text, word_set):
    total_words = len(text.split())
    found_words = sum(1 for word in text.split() if check_word_in_file(word.lower(), word_set))
    return found_words / total_words

def read_eng_dictionary():
    file_name = "words.txt"
    try:
        with open(file_name, 'r') as file:
            word_set = set(word.strip().lower() for word in file)

        return word_set
    except FileNotFoundError:
        print("File not found!")


if __name__ == "__main__":
    file_name = "words.txt"
    try:
        with open(file_name, 'r') as file:
            word_set = set(word.strip().lower() for word in file)

        text = input("Enter the text to check: ")

        accuracy = calculate_accuracy(text, word_set)
        print(f"Accuracy: {accuracy:.2%}")
    
    except FileNotFoundError:
        print("File not found!")