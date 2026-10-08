
import re
from typing import List, Tuple
from collections import Counter
import json
import jsonlines
from utils import cached

count_split_success = 0
count_split_failed = 0

def split_book(book: dict) -> dict:
    """
    Split the book content into chapters and update the book dictionary.

    Args:
    book (dict): The book dictionary containing 'content' key.

    Returns:
    dict: Updated book dictionary with 'content' as a list of chapter dictionaries.
    """
    content = book['content']

    # Regular expressions to match common chapter headings
    chapter_patterns = [
        r'\n\s*(#{1,5}\s+)?(?=.{1,50}\n)((?:Chapter|CHAPTER|Prologue|Epilogue|Afterword|Preface|Introduction|Conclusion|Appendix|Interlude|Part|PART|part|Book)|#{1,6})\s+(?:\d+|[IVXLCDM]+|(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred))\.?\s*\n',

        r'\n\s*(#{1,5}\s+)?(?=.{1,40}\n)((?:Chapter|CHAPTER|Prologue|Epilogue|Afterword|Preface|Introduction|Conclusion|Appendix|Interlude|Part|PART|part))\s+.*\n',


        r'\n\s*(?=.{1,50}\n)(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty|Thirty|Forty|Fifty|Sixty|Seventy|Eighty|Ninety|Hundred)(?:\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine))?\s*\n',

        r'\n\s*(#{1,5}\s+)?(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX|XXI|XXII|XXIII|XXIV|XXV|XXVI|XXVII|XXVIII|XXIX|XXX|XXXI|XXXII|XXXIII|XXXIV|XXXV|XXXVI|XXXVII|XXXVIII|XXXIX|XL|XLI|XLII|XLIII|XLIV|XLV|XLVI|XLVII|XLVIII|XLIX|L|LI|LII|LIII|LIV|LV|LVI|LVII|LVIII|LIX|LX|LXI|LXII|LXIII|LXIV|LXV|LXVI|LXVII|LXVIII|LXIX|LXX|LXXI|LXXII|LXXIII|LXXIV|LXXV|LXXVI|LXXVII|LXXVIII|LXXIX|LXXX|LXXXI|LXXXII|LXXXIII|LXXXIV|LXXXV|LXXXVI|LXXXVII|LXXXVIII|LXXXIX|XC|XCI|XCII|XCIII|XCIV|XCV|XCVI|XCVII|XCVIII|XCIX|C)\s*\n'


        r'\n\s*(?=.{1,50}\n)\d+\s*\n',

        r'\n\s*Chapter\s+(\d+)\.?\s*\n',
    ]

    chapter_regex = '|'.join(f'({pattern})' for pattern in chapter_patterns)
    chapter_pattern = re.compile(chapter_regex, re.IGNORECASE)

    # Find all chapter headings
    chapters = list(chapter_pattern.finditer(content))

    # Process chapters given the boundaries
    chapter_splits = []
    for i, match in enumerate(chapters):
        start = match.start()
        end = chapters[i+1].start() if i+1 < len(chapters) else len(content)

        if i == 0:
            # add the content before the first chapter
            chapter_content = content[:match.start()].strip()
            chapter_title = None
            chapter_splits.append({"title": chapter_title, "content": chapter_content})

        chapter_title = match.group(0).strip()
        chapter_content = content[start:end].strip()
        chapter_splits.append({"title": chapter_title, "content": chapter_content})

    for split in chapter_splits:
        print('===\n', split['content'][:100].split('\n')[0])
        from utils import num_tokens_from_string
        print(f'Num tokens: {num_tokens_from_string(split["content"])}')


    # now merge chapter_splits. If a split < 1000 char, merge it with the NEXT split.
    merged_chapter_splits = []
    chunk = {'title': '', 'content': ''}
    for split in chapter_splits:
        if not chunk['title']:
            chunk['title'] = split['title']
        chunk['content'] += ('' if not chunk['content'] else '\n') + split['content']
        if len(chunk['content']) >= 2000:
            merged_chapter_splits.append(chunk)
            chunk = {'title': '', 'content': ''}
    if chunk['content']:
        merged_chapter_splits.append(chunk)
    chapter_splits = merged_chapter_splits

    from utils import logger
    for split in chapter_splits:

        logger.info('===\n' + split['content'][:100].split('\n')[0])
        from utils import num_tokens_from_string
        logger.info(f'Num tokens: {num_tokens_from_string(split["content"])}')


    logger.info(f'Splitting {book["title"]} ({book.get("num_tokens", -1)} tokens) into {len(chapter_splits)} chapters')


    # If we have successfully split the book
    if len(chapter_splits) > 5:
        global count_split_success
        count_split_success += 1

        # Update the book dictionary
        return chapter_splits
    else:

        # print the first 100 lines of content, each line with 100 characters
        lines = content.split('\n')[:200]

        global count_split_failed
        count_split_failed += 1
        # Split the content into chunks of 8000 characters
        return None



if __name__ == '__main__':

    with jsonlines.open('dataset-construction/data-extraction/source-books/books_example.jsonl', mode='r') as reader:
        books_data = list(reader)

    # Process all books
    split_books = [split_book(book) for book in books_data]

    # Print the count
    print(f"Split {count_split_success} books, failed {count_split_failed} books")
    # Update books_data with the processed books
    books_data = split_books

    print(f"Processed {len(books_data)} books, splitting their content into chapters.")
