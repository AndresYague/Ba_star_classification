import sys

def main():
    if len(sys.argv) < 2:
        print("Give me a file!")
        return 1

    with open(sys.argv[1], "r") as fread:
        for line in fread:
            if "Label" in line:
                lnlst = line.split()
                print(lnlst[1], lnlst[7])
            elif "star" in line:
                print(line[:-1])

main()
