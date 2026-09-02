


def recursive_split(text, separators, chunk_size):
    if len(text) <= chunk_size:
        return [text]
    # Fixed chunking for recursive exit condition
    if not separators:
        return [
            text[i : i + chunk_size]
            for i in range(0,len(text), chunk_size)
        ]
    current_separator = separators[0]
    smaller_separators = separators[1:]
    splits = text.split(current_separator)
    chunks = []

    buffer = ""
    for split in splits:
        candidate = buffer + (current_separator if buffer else "") + split
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.extend(
                    recursive_split(buffer, smaller_separators, chunk_size)
                )
            buffer = split
    if buffer:
        chunks.extend(
            recursive_split(buffer,separators,chunk_size)
        )
    return chunks
