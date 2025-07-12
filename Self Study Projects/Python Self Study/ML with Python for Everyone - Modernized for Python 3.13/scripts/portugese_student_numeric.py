student_url = ('https://archive.ics.uci.edu/' +
               'ml/machine-learning-databases/00320/student.zip')

def grab_student_numeric():
    import urllib.request, zipfile
    urllib.request.urlretrieve(student_url, 'port_student.zip')
    zipfile.ZipFile('port_student.zip').extract('student-mat.csv')

    df = pd.read_csv('student-mat.csv', sep=';')
    df = df.drop(columns=['G1', 'G2']).select_dtypes(include=['number'])
    df.to_csv('portugese_student_numeric.csv', index=False)

# grab_student_numeric()
