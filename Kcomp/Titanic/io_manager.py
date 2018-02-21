import csv

def readfile(filename):
    raw_data = []
    with open(filename, newline='') as csvfile:
        contents = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in contents:
            raw_data.append(row)

    print("Format: " + str(raw_data[0]))
    return raw_data[1:]

def writefile(filename, contents):
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=', ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["PassengerId", "Survived"])
        for c in contents:
            spamwriter.writerow(c)
