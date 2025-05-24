student_url = (
    'https://archive.ics.uci.edu/' +
    'ml/machine-learning-databases/00320/student.zip'
)

def grab_student_numeric_discrete():
    import urllib.request
    import zipfile
    import pandas as pd

    # Download and extract
    urllib.request.urlretrieve(student_url, 'port_student.zip')
    with zipfile.ZipFile('port_student.zip') as zf:
        zf.extract('student-mat.csv')

    # Load and preprocess
    df = pd.read_csv('student-mat.csv', sep=';')
    df = df.drop(columns=['G1', 'G2']).select_dtypes(include=['number'])
    df['grade'] = pd.cut(
        df['G3'], [0, 11, 14, 20],
        labels=['low', 'mid', 'high'],
        include_lowest=True
    )
    df.drop(columns=['G3'], inplace=True)

    # Save cleaned file
    df.to_csv('portugese_student_numeric_discrete.csv', index=False)
