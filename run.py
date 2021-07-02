import sys
import GenAlg

# tours
tours = ["./tour29.csv", "./tour194.csv", "./tour929.csv", "./tour100.csv"]

## Argument parsing
if len (sys.argv) > 2:
    print("Usage: python run.py [tourname|tour29|tour194|tour929|tour100]")
    sys.exit(1)
if len(sys.argv) == 2:
    tourarg = sys.argv[1]

if tourarg == "tour29": file = tours[0]
elif tourarg == "tour194": file = tours[1]
elif tourarg == "tour929": file = tours[2]
elif tourarg == "tour100": file = tours[3]
else:
    print("Could not recognize argument: '%s', expected one of %s" %
    (tourarg, str(["tour29", "tour194", "tour929", "tour100"])))
    sys.exit(2)


if __name__ == "__main__":
    a = GenAlg.GenAlg("GenAlg")
    #a.optimize("./tour29.csv")
    a.optimize(file)
