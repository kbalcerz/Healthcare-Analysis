from ipywidgets import interact

# Function borrowed from the book:
# Principles and techniques of data science. Lau, Gonzalez, Nolan. 
# found @ https://www.textbook.ds100.org/ch/05/cleaning_calls.html
def df_interact(df):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + 5, col:col + 10]
    interact(peek, row=(0, len(df), 5), col=(0, len(df.columns) - 10))
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))