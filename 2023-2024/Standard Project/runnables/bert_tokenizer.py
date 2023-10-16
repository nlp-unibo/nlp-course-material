from transformers import BertTokenizerFast


def find_search_ids(text_ids, search_ids):
    search_length = len(search_ids)
    found_positions = []
    for pos in range(len(text_ids)):
        if text_ids[pos: pos + search_length] == search_ids:
            found_positions.append(list(range(pos, pos + search_length)))

    return found_positions


def word_to_ids_mapping(text, search_word, tokenizer):
    encoded_text = tokenizer.encode_plus(text)
    encoded_search = tokenizer.encode_plus(search_word, add_special_tokens=False)

    text_ids = encoded_text.input_ids
    search_ids = encoded_search.input_ids

    search_positions = find_search_ids(text_ids=text_ids, search_ids=search_ids)
    if not len(search_positions):
        raise RuntimeError(f'Could not find word {search_word} in {text}')

    print(search_positions)


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    example = "avevo chiesto a mia madre di andare al mc ma dice che sono una balena!"
    search_word = "balena"

    word_to_ids_mapping(text=example,
                        search_word=search_word,
                        tokenizer=tokenizer)
    # word_to_ids_mapping(encoded=tokenizer(search_word))
