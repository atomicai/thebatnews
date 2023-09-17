def clean_text(data, columns = ['text', 'title']):
  """
  data : polars.DataFrame

  Removes URLs, telegram links, special_characters from given DataFrame columns
  """
  for column in columns:
    data = data.with_columns(pl.col(column).apply(lambda s: re.sub("https?:[^\s]+|_|\n|\*|(t\.me.*)|\[|\]|\(|\)", "", s)))